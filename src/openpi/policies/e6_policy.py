import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_e6_example() -> dict:
    """Creates a random input example for an E6-style policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),
        "prompt": "approach red object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class E6Inputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # v14+: insert dummy 0 at state index 6 → [j1..j6, 0, gripper] 8D
    use_dummy_joint: bool = False

    def __call__(self, data: dict) -> dict:
        hik_image = _parse_image(data["observation/exterior_image_1_left"])
        zed_image = _parse_image(data["observation/exterior_image_2_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                # E6 2cam: HIK (base) + ZED (left_wrist slot), right_wrist zeros.
                images = (hik_image, zed_image, np.zeros_like(hik_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (hik_image, zed_image, np.zeros_like(hik_image))
                # FAST models do not use image masking for padded views.
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        state = np.asarray(data["observation/state"])
        if self.use_dummy_joint and state.shape[0] == 7:
            # v14: 7D [j1..j6, gripper] → 8D [j1..j6, 0, gripper]
            state = np.insert(state, 6, 0.0)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class E6Outputs(transforms.DataTransformFn):
    # v14+: remove dummy at action index 6 → [j1..j6, gripper] 7D
    use_dummy_joint: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if self.use_dummy_joint and actions.shape[1] == 8:
            # v14: 8D [j1..j6, dummy, gripper] → 7D [j1..j6, gripper]
            actions = np.concatenate([actions[:, :6], actions[:, 7:8]], axis=1)
        else:
            # v13 and earlier: take first 7 columns
            actions = actions[:, :7]
        return {"actions": actions}
