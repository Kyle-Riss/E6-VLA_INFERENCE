import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Optional LoRA on the SigLIP vision tower (separate from
    # ``paligemma_variant`` which controls the Gemma LLM side). Set
    # ``vision_lora_rank`` to a positive int to enable; layers outside
    # ``vision_lora_layer_range`` (inclusive, 0-indexed) are masked out.
    # The base SigLIP weights remain unchanged regardless — LoRA is added
    # as a parallel residual.
    vision_lora_rank: int | None = None
    vision_lora_alpha: float = 16.0
    vision_lora_layer_range: tuple[int, int] | None = None

    # Optional LoRA layer scoping for the action expert (Gemma 300m, depth 18).
    # Only meaningful when ``action_expert_variant`` already enables LoRA
    # (e.g. ``"gemma_300m_lora"``). Inclusive 0-indexed range; layers outside
    # have their LoRA contribution multiplied by 0 (no forward effect, no
    # gradient flow). When ``None``, LoRA is active on all 18 layers (legacy
    # v2/v3 behavior). Used by v4 to constrain which expert layers adapt.
    action_expert_lora_layer_range: tuple[int, int] | None = None

    # Per-dimension loss weights applied to the squared flow-matching error
    # before averaging over action_dim. Length must equal action_dim (32).
    # E.g. set index 3 to 3.0 to up-weight j4 during training (v7).
    # When None, all dimensions are weighted equally (legacy behavior).
    action_loss_weights: tuple[float, ...] | None = None

    # Path to a task-metric spec (see scripts/make_task_metric_spec.py). When set, the loss
    # gains a term that scores the flow-matching residual by what it does to the tool rather
    # than by its size in normalized action coordinates.
    #
    # It is a file rather than a set of fields because the term needs the dataset's own
    # normalization constants, and the model cannot see those — they live in the transform
    # stage. Resolving them once and writing them down also makes the calibration auditable
    # instead of something to take on faith.
    #
    # None reproduces the standard objective exactly, which is the identity the unit tests
    # check. Only the continuous arm joints enter this term: the suction channel is a binary
    # actuator whose effect is not a displacement, so it keeps the ordinary objective.
    task_metric_spec: str | None = None

    # Keys that receive ONLY color augmentation (no spatial crop/rotate) during training.
    # When None (default), any key containing "wrist" is treated as wrist-only.
    # Set to () so that ALL camera slots receive spatial augmentation — correct for E6
    # where both base_0_rgb (HIK) and left_wrist_0_rgb (ZED) are exterior cameras.
    wrist_image_keys: tuple[str, ...] | None = None

    # Camera slots the model consumes, in token order. Defaults to the DROID
    # 3-slot layout that pi0/pi0.5 were pretrained with.
    #
    # Setups that fill a slot with zeros (E6/E7 pass ``right_wrist_0_rgb`` as
    # zeros + mask False) can drop it here instead: a masked slot contributes
    # nothing to the loss but still costs 256 tokens of sequence and a full
    # SigLIP forward. Dropping it is numerically a no-op for the surviving
    # tokens — ``positions`` is ``cumsum(input_mask)-1`` so masked tokens add 0,
    # and ``make_attn_mask``'s ``valid_mask`` blocks them in both directions —
    # and parameter shapes are unaffected (SigLIP weights are reused per image),
    # so checkpoints stay loadable across the change.
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS

    # ── M2' (Deployment Contract v1) ──────────────────────────────────────────
    # 🔴 기본값이 전부 inert 해야 한다. `query_grounding: bool = True` 같은 실수 하나로
    #    E6 v16~v26 · E7 v1 · v2_60 · v2_120 서빙이 **조용히** 바뀐다(모듈이 생기고
    #    strict 로드가 거부한다). 비트 동일 게이트로 매번 확인할 것.
    #
    # 추론이 실제로 읽는 것은 아래 넷뿐이다(`pi0_pytorch.py` 실측):
    #   camera_role_embed · query_grounding · qg_rank · qg_slot
    camera_role_embed: bool = False
    query_grounding: bool = False
    # None 이면 마지막 슬롯 — image_keys 순서상 라벨 뷰(right_wrist_0_rgb).
    # ⚠️ 순서가 다르면 엉뚱한 카메라에 주입되고 **에러가 안 난다.** 로드 시 대조할 것.
    qg_slot: int | None = None
    qg_rank: int = 64

    # 아래는 **학습 전용**이다. 추론 경로는 읽지 않는다(`pi0_pytorch.py` 는 bounded
    # 형태를 무조건 쓴다). 계약 문서화를 위해 자리만 두고, **기본값은 학습서버 원본
    # 정의와 글자 단위로 같게** 맞춘다 — 처음엔 내가 임의로 1.0/1.0/0.0 을 적었는데,
    # 추론에서 안 읽힌다는 이유로 원본과 다른 값을 남기면 이 파일을 계약 문서로 읽는
    # 쪽이 학습 설정을 잘못 읽는다. `qg_roi` 는 학습 쪽 `e7_label_card_rois()` 의
    # 산출물이고 그 함수가 이 트리에 없다 — 추론에 불필요하므로 옮기지 않았다.
    qg_lambda: float = 0.0
    qg_tau: float = 0.07
    qg_direct: bool = False
    qg_smooth: float = 0.05
    qg_bounded: bool = False
    qg_roi: tuple[tuple[int, ...], ...] | None = None
    # ─────────────────────────────────────────────────────────────────────────

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={k: image_spec for k in self.image_keys},
                image_masks={k: image_mask_spec for k in self.image_keys},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


# 🔴 E6 전용 — 2026-08-12 코드 드롭 병합 때 **되살린 함수**다.
# 학습서버의 ad3685e 판 pi0_config.py 에는 이 함수가 없는데(그쪽에서 쓰지 않으므로)
# 우리 트리의 E6 TrainConfig **20개**가 참조한다. 통째로 덮어썼더니
# `AttributeError: module 'openpi.models.pi0_config' has no attribute ...` 로
# config.py import 자체가 실패했다 — E7 뿐 아니라 E6 서빙도 같이 죽는다.
# ⚠️ 다음에 학습서버 파일로 이 모듈을 교체할 때 **반드시 다시 확인할 것.**
def freeze_filter_vlm_frozen_vision_and_action_lora() -> nnx.filterlib.Filter:
    """Freeze base weights; train vision LoRA + action-expert LoRA + action heads.

    Use with ``vision_lora_rank`` set and ``action_expert_variant`` containing ``lora``.
    Freezes SigLIP non-LoRA params and LLM non-expert-LoRA params.
    """
    llm = nnx_utils.PathRegex("PaliGemma/llm/.*")
    img = nnx_utils.PathRegex("PaliGemma/img/.*")
    has_lora = nnx_utils.PathRegex(".*lora.*")
    has_1 = nnx_utils.PathRegex(".*_1.*")
    expert_lora = nnx.All(has_lora, has_1)
    freeze_img = nnx.All(img, nnx.Not(has_lora))
    freeze_llm = nnx.All(llm, nnx.Not(expert_lora))
    return nnx.Any(freeze_img, freeze_llm)


def freeze_filter_vlm_frozen_action_expert_lora_only() -> nnx.filterlib.Filter:
    """Freeze all of ``PaliGemma`` except action-expert LoRA, for narrow π0.5 fine-tuning.

    Use with ``paligemma_variant="gemma_2b"`` and ``action_expert_variant="gemma_300m_lora"``.

    :meth:`Pi0Config.get_freeze_filter` does not freeze SigLIP (``PaliGemma/img``) or the main Gemma stack when only
    the action expert has LoRA, which would train the full vision-language tower. This filter freezes both the image
    encoder and all LLM base weights, while keeping trainable only (1) LoRA tensors on the action expert (paths
    matching both ``lora`` and ``_1``) and (2) the small action-side heads (``action_in_proj``, ``time_mlp_*``,
    ``action_out_proj``), which live outside ``PaliGemma``.
    """
    llm = nnx_utils.PathRegex("PaliGemma/llm/.*")
    img = nnx_utils.PathRegex("PaliGemma/img/.*")
    has_lora = nnx_utils.PathRegex(".*lora.*")
    has_1 = nnx_utils.PathRegex(".*_1.*")
    expert_lora = nnx.All(has_lora, has_1)
    freeze_llm = nnx.All(llm, nnx.Not(expert_lora))
    return nnx.Any(img, freeze_llm)


def freeze_filter_v3_vision_late_lora() -> nnx.filterlib.Filter:
    """Freeze base PaliGemma, train only (action-expert LoRA + vision LoRA + action heads).

    Use with ``paligemma_variant="gemma_2b"``, ``action_expert_variant="gemma_300m_lora"``,
    AND ``vision_lora_rank`` set on :class:`Pi0Config`.

    Trainable:
      - Action-expert LoRA tensors  (paths matching both ``lora`` and ``_1``)
      - Vision LoRA tensors          (paths under ``PaliGemma/img/...`` containing ``lora``)
      - Action-side heads outside ``PaliGemma`` (``action_in_proj``, ``time_mlp_*``, ``action_out_proj``)

    Frozen:
      - All LLM base weights (``PaliGemma/llm/...`` minus action-expert LoRA)
      - All SigLIP base weights (``PaliGemma/img/...`` minus vision LoRA)

    Note: layers excluded by ``vision_lora_layer_range`` still allocate LoRA params
    (because :func:`scan` stacks them along the depth axis), but their contribution
    is multiplied by a zero mask so they never influence the loss and never receive
    gradient. They effectively stay at init values throughout training.
    """
    llm = nnx_utils.PathRegex("PaliGemma/llm/.*")
    img = nnx_utils.PathRegex("PaliGemma/img/.*")
    has_lora = nnx_utils.PathRegex(".*lora.*")
    has_1 = nnx_utils.PathRegex(".*_1.*")
    expert_lora = nnx.All(has_lora, has_1)
    img_lora = nnx.All(img, has_lora)
    freeze_llm = nnx.All(llm, nnx.Not(expert_lora))
    freeze_img = nnx.All(img, nnx.Not(img_lora))
    return nnx.Any(freeze_img, freeze_llm)


def freeze_filter_v4_combined_lora() -> nnx.filterlib.Filter:
    """Same gradient mask as v3: vision LoRA + action-expert LoRA + small action heads trainable.

    v4 adds layer-range scoping on the action expert via
    :attr:`Pi0Config.action_expert_lora_layer_range`, but that scoping happens
    at forward time inside ``gemma.Module`` (via per-layer mask × LoRA output).
    The freeze filter — which decides which params get a gradient slot in the
    optimizer — does not need to change: out-of-range expert LoRA params are
    still allocated, still listed as "trainable" by the filter, and the mask
    zeroes their forward contribution so their gradient is exactly 0. This
    function therefore delegates to :func:`freeze_filter_v3_vision_late_lora`
    and is kept as an alias for clarity in v4 ``TrainConfig``s.
    """
    return freeze_filter_v3_vision_late_lora()


def freeze_filter_vision_full_finetune() -> nnx.filterlib.Filter:
    """Full SigLIP fine-tuning: freeze only LLM base weights, everything else trainable.

    Trainable:
      - All SigLIP (PaliGemma/img) base weights — no LoRA, direct weight update
      - Action-expert LoRA tensors (paths matching both ``lora`` and ``_1``)
      - Action-side heads (``action_in_proj``, ``time_mlp_*``, ``action_out_proj``)

    Frozen:
      - All LLM base weights (``PaliGemma/llm/...`` minus action-expert LoRA)

    Use with ``vision_lora_rank=None`` (no LoRA adapters on SigLIP).
    """
    llm = nnx_utils.PathRegex("PaliGemma/llm/.*")
    has_lora = nnx_utils.PathRegex(".*lora.*")
    has_1 = nnx_utils.PathRegex(".*_1.*")
    expert_lora = nnx.All(has_lora, has_1)
    freeze_llm = nnx.All(llm, nnx.Not(expert_lora))
    return freeze_llm
