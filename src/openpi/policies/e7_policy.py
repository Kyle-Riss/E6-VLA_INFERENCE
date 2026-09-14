import ast
import dataclasses
import functools
import json
import pathlib

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_e7_example() -> dict:
    """Creates a random input example for an E7-style policy (xArm 6)."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),  # xArm 6: [j1..j6, gripper] = 7D
        "prompt": "approach the target object",
    }


def make_e7_label_example() -> dict:
    """Input example for the 3-slot, label-grounded variant."""
    return make_e7_example() | {
        "observation/label_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "carry the science book",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


_SIDES = ("left", "center", "right")


@functools.cache
def _shelf_layouts(repo_id: str) -> dict[int, int] | None:
    """Which shelf position each episode's own category sits at, keyed by episode index.

    Read from the dataset's own ``meta/e7_context.json`` rather than from a column, because
    the layout is a property of the episode and the recorded frames already carry the episode
    index. Nothing is re-converted to add this.

    Returns None when the file is absent -- datasets without shelf signs simply have no
    grounding target, and the model raises only if a run actually asks for one.
    """
    path = pathlib.Path.home() / ".cache/huggingface/lerobot" / repo_id / "meta/e7_context.json"
    if not path.exists():
        return None
    episodes = json.loads(path.read_text())["episodes"]
    out = {}
    for key, meta in episodes.items():
        layout = meta.get("shelf_layout")
        category = meta.get("category")
        if not layout or not category:
            continue
        if isinstance(layout, str):
            layout = ast.literal_eval(layout)
        side = layout.get(category)
        if side in _SIDES:
            out[int(key)] = _SIDES.index(side)
    return out or None


@dataclasses.dataclass(frozen=True)
class E7Inputs(transforms.DataTransformFn):
    """xArm 6 (E7) inputs — 7D state/action, same contract as E6 v16+.

    Collection note: the xArm6 teleop path commands Cartesian velocity
    (``/xarm/vc_set_cartesian_velocity`` @ 20 Hz), but the training label is
    ``q_measured[t+1] - q_measured[t]`` from the 16 Hz joint recording, so the
    joint-delta contract holds regardless of how the arm was driven.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    # Dataset to read shelf layouts from when a run trains with the query-grounding
    # auxiliary term. None (the default) emits no target, which is every other run.
    grounding_repo_id: str | None = None

    def __call__(self, data: dict) -> dict:
        hik_image = _parse_image(data["observation/exterior_image_1_left"])
        zed_image = _parse_image(data["observation/exterior_image_2_left"])
        # Present only for label-grounded runs. Its absence is what separates the
        # 2-slot baseline from the 3-slot condition, and both are built from the
        # same recorded episodes -- so this is a plain key check, not an error.
        label_image = data.get("observation/label_image")

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # HIK → base slot, ZED → left_wrist slot. Unlike E6, an empty
                # third slot is NOT emitted: a zeros+mask-False slot still costs
                # 256 sequence tokens and a full SigLIP forward while
                # contributing nothing. When a label view IS supplied it takes
                # that slot for real. Requires the matching ``image_keys`` on
                # :class:`Pi0Config` (see ``pi05_e7_v1_lora``).
                names = ("base_0_rgb", "left_wrist_0_rgb")
                images = (hik_image, zed_image)
                image_masks = (np.True_, np.True_)
                if label_image is not None:
                    names += ("right_wrist_0_rgb",)
                    images += (_parse_image(label_image),)
                    image_masks += (np.True_,)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (hik_image, zed_image, np.zeros_like(hik_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # xArm 6: [j1..j6, gripper] = 7D — same shape as E6.
        # No DROID 8D alignment (gripper stays at index 6, NOT 7). This matches
        # E6 v23 (``align_droid_state=False``), which is the reference run for the
        # E6→E7 cross-embodiment comparison; inserting a dummy j7 here would change
        # the action contract and break that comparison.
        state = np.asarray(data["observation/state"])  # (7,)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # LeRobot marks the chunk steps it had to invent when the horizon ran past the end
        # of the episode; it fills them by repeating the last real action. Carried through
        # as its complement so the loss can drop them. Absent at inference and on datasets
        # that predate the flag, and absent means every step is real.
        if "action_is_pad" in data:
            inputs["action_valid"] = ~np.asarray(data["action_is_pad"], dtype=bool)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        if self.grounding_repo_id is not None and "observation/episode_index" in data:
            layouts = _shelf_layouts(self.grounding_repo_id)
            if layouts is not None:
                episode = int(np.asarray(data["observation/episode_index"]))
                # -1 for an episode the metadata does not describe. The loss masks those out
                # rather than scoring a guess.
                inputs["grounding_target"] = np.int32(layouts.get(episode, -1))

        return inputs


@dataclasses.dataclass(frozen=True)
class E7Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        acts = np.asarray(data["actions"])
        # action contract: [Δj1..Δj6, gripper] = 7D (same layout as E6 v16+)
        # joint(6D): velocity delta (deg/frame) — target = q_measured + Δ, see
        #            ``delta_reference`` in the executor config (NOT q_target_prev + Δ)
        # gripper(1D): absolute command, 0.0 = open .. 1.0 = close.
        #   ⚠ OPEN: binary vs continuous is undecided. E6 was a vacuum (true {0,1});
        #   the xArm6 G2 is a parallel gripper with continuous aperture. Record the
        #   raw continuous value at collection — binarising later is lossless, the
        #   reverse is not — and fix the contract in the conversion script, not here.
        #   Binary executor rule (cmd = 1 if action[6] > 0.5) applies only if the
        #   conversion thresholds it.
        return {"actions": acts[:, :7]}
