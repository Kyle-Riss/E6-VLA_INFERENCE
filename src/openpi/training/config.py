"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.e6_policy as e6_policy

# 🔴 2026-08-12 — `config_snippets.py` 에 이 import 가 빠져 있었다.
# `LeRobotE7DataConfig.create()` 가 `e7_policy.E7Inputs` 를 부르는데 `NameError` 로
# 죽는다. ⚠️ **README 의 `984` 게이트는 이걸 못 잡는다** — 그 게이트는 `t.model.*` 만
# 보고 `data.create()` 를 부르지 않기 때문이다(실제로 통과한 뒤에 발견했다).
import openpi.policies.e7_policy as e7_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Caps how much more a low-variance dimension can be amplified than the median
    # dimension under quantile normalization. See `transforms.quantile_bounds`.
    # None keeps the unmodified quantile rule (bit-for-bit). Only used when
    # `use_quantile_norm` is true.
    #
    # 🔴 필수 필드다 — `policy_config.py` 가 `data_config.quantile_range_floor` 를
    #    무조건 읽으므로 없으면 AttributeError 로 **E6 서빙까지 죽는다**
    #    (2026-08-12 코드 드롭 병합 시 발견. 학습서버 패키지에 빠져 있었다).
    # ⚠️ E7 grounded 는 **None** 이다(학습서버 확인). 트리에 있는 4.0/6.0/8.0 은
    #    E6 v34/v37/v38 정규화 대조군 값이므로 **가져오지 말 것**.
    quantile_range_floor: float | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotE6DataConfig(DataConfigFactory):
    """Data config for a custom E6 LeRobot dataset."""

    # v14+: state/action에 dummy index 6 삽입/제거 (8D ↔ 7D 변환)
    use_dummy_joint: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/state": "state",
                        # Map LeRobot ``action`` -> openpi ``actions`` for downstream transforms.
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # E6 v1 uses 6 joint deltas + 1 gripper command directly in the dataset actions.
        data_transforms = _transforms.Group(
            inputs=[e6_policy.E6Inputs(model_type=model_config.model_type, use_dummy_joint=self.use_dummy_joint)],
            outputs=[e6_policy.E6Outputs(use_dummy_joint=self.use_dummy_joint)],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")



@dataclasses.dataclass(frozen=True)
class LeRobotE7DataConfig(DataConfigFactory):
    """Data config for xArm 6 (E7) LeRobot dataset.

    State/action are 7D: [j1..j6, gripper] — identical in shape and semantics to
    E6, so the same LeRobot schema and conversion contract apply:
        state  = [j1..j6 (t), gripper_command (t)]        degrees, absolute
        action = [Δj1..Δj6 (t→t+1), gripper_command (t+1)] deg/frame, gripper absolute

    Deliberately NO DROID 8D alignment (no dummy j7, gripper stays at index 6):
    E6 v23 — the reference run for the E6→E7 cross-embodiment comparison — used
    ``align_droid_state=False``, and the padded dims cost ~1% of the loss anyway
    (measured on v23), so there is nothing to gain from realigning here.
    """

    # Feed the dedicated shelf-label view into the third image slot.
    #
    # This is the manipulated variable of the main ablation, not a convenience:
    # the 2-slot and 3-slot conditions are trained from the SAME converted
    # dataset and differ only here (plus the matching ``image_keys`` on the
    # model). Leaving it False on a dataset that has ``label_image`` silently
    # drops the column, which is exactly the 2-slot baseline.
    use_label_view: bool = False

    # Photometric jitter on the label slot. TRAINING ONLY -- accepted so a v2
    # TrainConfig resolves, and deliberately not acted on here: the robot must see its
    # own cameras unmodified. The training tree assembles it into a group that
    # `create_trained_policy` never reads, so there is nothing to reproduce.
    label_jitter: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack = {
            "observation/exterior_image_1_left": "exterior_image_1_left",
            "observation/exterior_image_2_left": "exterior_image_2_left",
            "observation/state": "state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.use_label_view:
            repack["observation/label_image"] = "label_image"
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack)]
        )
        data_transforms = _transforms.Group(
            inputs=[e7_policy.E7Inputs(model_type=model_config.model_type)],
            outputs=[e7_policy.E7Outputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi05_e7_grounded_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            # On. With pi05 the continuous state path does not exist -- `state_proj`
            # is only built in the pi0 branch and `embed_suffix` skips the state
            # token -- so this flag is the ONLY way proprioception reaches the
            # network. Left at the inherited False (upstream `pi05_libero`, copied
            # through every E6 config) the policy sees images and text and nothing
            # else: it reads its own arm off the ZED view. E6 was a top-down suction
            # pick and place and could afford that; E7 pushes a book along +x into a
            # 350 mm slot, and depth along the camera axis is exactly what a 224 px
            # view resolves worst.
            #
            # It is free in sequence budget. The text slot is padded to
            # `max_token_len` either way, so the length stays 984; what changes is
            # how much of the 200 is real -- measured 11 -> 43 tokens on
            # "insert the liberal arts book into the appropriate shelf". Only the
            # true 7 dims are written, because TokenizePrompt runs BEFORE
            # PadStatesAndActions in ModelTransformFactory; the 32-dim padding never
            # reaches the tokenizer.
            discrete_state_input=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(18, 26),
            action_expert_lora_layer_range=None,
            wrist_image_keys=(),
            image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        ),
        data=LeRobotE7DataConfig(
            repo_id="local/e7_books_v4",
            use_label_view=True,
            base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),
            assets=AssetsConfig(assets_dir="assets/pi05_e7_grounded_lora", asset_id="local/e7_books_v4"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=8,
        log_interval=50,
        save_interval=2500,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000),
        freeze_filter=pi0_config.freeze_filter_v4_combined_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_e7_grounded_v2_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            # On. With pi05 the continuous state path does not exist -- `state_proj`
            # is only built in the pi0 branch and `embed_suffix` skips the state
            # token -- so this flag is the ONLY way proprioception reaches the
            # network. Left at the inherited False (upstream `pi05_libero`, copied
            # through every E6 config) the policy sees images and text and nothing
            # else: it reads its own arm off the ZED view. E6 was a top-down suction
            # pick and place and could afford that; E7 pushes a book along +x into a
            # 350 mm slot, and depth along the camera axis is exactly what a 224 px
            # view resolves worst.
            #
            # It is free in sequence budget. The text slot is padded to
            # `max_token_len` either way, so the length stays 984; what changes is
            # how much of the 200 is real -- measured 11 -> 43 tokens on
            # "insert the liberal arts book into the appropriate shelf". Only the
            # true 7 dims are written, because TokenizePrompt runs BEFORE
            # PadStatesAndActions in ModelTransformFactory; the 32-dim padding never
            # reaches the tokenizer.
            discrete_state_input=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(18, 26),
            action_expert_lora_layer_range=None,
            wrist_image_keys=(),
            image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        ),
        data=LeRobotE7DataConfig(
            repo_id="local/e7_books_v4",
            use_label_view=True,
            label_jitter=True,
            base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),
            assets=AssetsConfig(assets_dir="assets/pi05_e7_grounded_v2_lora", asset_id="local/e7_books_v4"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=8,
        log_interval=50,
        save_interval=2500,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000),
        freeze_filter=pi0_config.freeze_filter_v4_combined_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        # Serves the 2026-08-19 bundle (`e7_v2_60_bundle`, step 19999), trained on
        # the corpus collected AFTER the 08-15 camera move. Same model as
        # `pi05_e7_grounded_lora` in every field; the only differences are the
        # dataset identifiers.
        #
        # It has to exist as its own entry rather than reusing the grounded config,
        # because `--policy.config` resolves the TrainConfig from THIS source tree
        # and `create_trained_policy` reads norm stats from
        # `<bundle>/assets/<data_config.asset_id>/norm_stats.json`. The bundle ships
        # them under `local/e7_books_v2_60`; pointed at `local/e7_books_v4` the load
        # dies with FileNotFoundError, and pointed at v1's stats it would normalise
        # against a different camera rig with a different divisor (47 train episodes,
        # holdout excluded) -- which does not raise at all.
        name="pi05_e7_v2_60_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            # On, for the reasons spelled out on `pi05_e7_grounded_lora` above. The
            # bundle manifest carries `discrete_state_input: true`, and the startup
            # gate compares it, so a False here would be caught -- but only if the
            # gate runs; keep the two in step.
            discrete_state_input=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(18, 26),
            action_expert_lora_layer_range=None,
            wrist_image_keys=(),
            image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        ),
        data=LeRobotE7DataConfig(
            repo_id="local/e7_books_v2_60",
            use_label_view=True,
            base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),
            assets=AssetsConfig(assets_dir="assets/pi05_e7_v2_60_lora", asset_id="local/e7_books_v2_60"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=8,
        log_interval=50,
        save_interval=2500,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000),
        freeze_filter=pi0_config.freeze_filter_v4_combined_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        # Serves the 2026-08-20 bundle (`e7_v2_120_step20000_bundle`, step 20000),
        # trained on the 119-episode corpus with FOUR sign layouts. Same model as
        # `pi05_e7_v2_60_lora` in every field; only the dataset identifiers differ.
        #
        # Why the four layouts matter: in v2_60 `science` sat on the right in both
        # layouts, so position memory alone scored 66.7% and science had to be
        # excluded from any grounding claim. Here all three categories reach all
        # three shelves (20/10/10 each), which drops the position-memory ceiling to
        # 50% and the brightness-only baseline to chance (32.5%). All three
        # categories become usable.
        #
        # As with the v2_60 entry, this must exist as its own TrainConfig because
        # `--policy.config` resolves the name from THIS source tree and
        # `create_trained_policy` reads norm stats from
        # `<bundle>/assets/<asset_id>/norm_stats.json`. The stats here were
        # recomputed from the new train split (95 episodes, 24 held out); reusing
        # v2_60's would normalise against a different divisor and would NOT raise.
        name="pi05_e7_v2_120_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            # On, for the reasons spelled out on `pi05_e7_grounded_lora` above.
            discrete_state_input=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(18, 26),
            action_expert_lora_layer_range=None,
            wrist_image_keys=(),
            image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        ),
        data=LeRobotE7DataConfig(
            repo_id="local/e7_books_v2_120",
            use_label_view=True,
            base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),
            assets=AssetsConfig(assets_dir="assets/pi05_e7_v2_120_lora", asset_id="local/e7_books_v2_120"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=8,
        log_interval=50,
        save_interval=2500,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=40_000),
        freeze_filter=pi0_config.freeze_filter_v4_combined_lora(),
        ema_decay=None,
    ),
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        # E6 v1: single exterior camera, 7D state/action contract, primitive task strings from LeRobot tasks.
        name="pi05_e6_v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # Keep pi05 internal action dimension.
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotE6DataConfig(
            # Same repo_id as examples/e6/convert_e6_episode_to_lerobot.py DEFAULT_REPO_ID.
            repo_id="billy/dobot_e6_pick_place_random_v1",
            base_config=DataConfig(
                prompt_from_task=True,
                # LeRobot datasets usually use the ``action`` column name (not ``actions``).
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        # π0.5 + LoRA (E6): freeze SigLIP + full Gemma stacks; train only action-expert LoRA + small action heads.
        # See :func:`openpi.models.pi0_config.freeze_filter_vlm_frozen_action_expert_lora_only` (not
        # :meth:`Pi0Config.get_freeze_filter`, which would leave the vision tower and main LLM trainable).
        name="pi05_e6_v1_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotE6DataConfig(
            repo_id="billy/dobot_e6_pick_place_random_v1",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            # Reuse norm stats from ``compute_norm_stats.py --config-name pi05_e6_v1`` (same dataset).
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v1",
                asset_id="billy/dobot_e6_pick_place_random_v1",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        # Default to 1 for OOM-safe smoke runs; ramp batch on the CLI (2, 4, 8, …) after checking loss/checkpoints.
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_action_expert_lora_only(),
        ema_decay=None,
    ),
    TrainConfig(
        # π0.5 + LoRA (E6 v2): orange box pick-and-place, episode-level prompts (left↔right).
        name="pi05_e6_v2_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotE6DataConfig(
            repo_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v2",
                asset_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_action_expert_lora_only(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_e6_v3_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v2",
                asset_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_action_expert_lora_only(),
        ema_decay=None,
    ),
    TrainConfig(
        # π0.5 + vision LoRA (SigLIP 22-26 r16) + action expert LoRA (r32). WandB: 1v6ufvih
        name="pi05_e6_v4_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v2",
                asset_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        # π0.5 + vision LoRA (SigLIP 22-26 r16) + action expert LoRA r16 (gemma_300m_lora_r16). WandB: 5f7jjoze
        name="pi05_e6_v5_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v2",
                asset_id="kyle-riss/dobot_e6_pick_place_orange_v2",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        # π0.5 + vision LoRA (SigLIP 22-26 r16) + action expert LoRA r32 (gemma_300m_lora). v6 dataset.
        name="pi05_e6_v6_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v6",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v6",
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v6",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_e6_v8_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v8",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v8_lora",
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v8",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_e6_v9_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v8",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v8_lora",
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v8",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_e6_v10_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v10",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v10_lora",
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v10",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v11: action rank=16 (gemma_300m_lora_r16), scope 11~15 layer (학습 시 freeze filter로 제어), vision LoRA 동일
    TrainConfig(
        name="pi05_e6_v11_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v10",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v10_lora",
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v10",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v12: action rank=16 (gemma_300m_lora_r16), scope 전체 18 layer, vision LoRA 동일
    TrainConfig(
        name="pi05_e6_v12_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v10",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v10_lora",
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v10",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_e6_v13_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v13",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                assets_dir="assets/pi05_e6_v13_lora",
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v13",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_e6_v14_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v14",
            use_dummy_joint=True,
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v14",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=22_500,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v16: vision LoRA 22~26 (same as v13/v14), 7D state/action, gripper absolute
    TrainConfig(
        name="pi05_e6_v16_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=17_500,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v18: vision LoRA 0~25 (full range), same dataset/action contract as v16/v17, 20k steps
    TrainConfig(
        name="pi05_e6_v18_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(0, 25),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v19: vision LoRA mid only (14~18, 5L), same dataset/action contract as v16
    TrainConfig(
        name="pi05_e6_v19_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(14, 18),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v17: vision LoRA 14~25 (wider range), same dataset/action contract as v16
    TrainConfig(
        name="pi05_e6_v17_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(14, 25),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=15_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    #
    # Dobot E6 custom fine-tune configs.
    # These mirror the configs in move-one/openpi/src/openpi/training/config.py so that
    # serve_policy.py can load user-trained checkpoints without modifying move-one.
    # All three configs use DroidInputs/DroidOutputs (8-D action contract: Δq[0..5], pad, gripper).
    #
    TrainConfig(
        name="pi0_e6_freeze_vlm",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
    ),
    TrainConfig(
        # Primitive-176 local fine-tune (DROID or BASE init; same DroidInputs contract).
        name="pi0_e6_freeze_vlm_primitive_176_local",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="local/primitive_tagged_v1_full"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
    ),
    TrainConfig(
        # Primitive-176 UR5-style run (exterior only; wrist slot zeroed at inference).
        # Pass --input_layout ur5_style to the client so wrist_image is zeros.
        name="pi0_e6_freeze_vlm_primitive_176_local_ur5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="local/primitive_tagged_v1_full"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
    ),
    # v21: vision LoRA Early (0~8), same dataset/action contract as v16
    TrainConfig(
        name="pi05_e6_v21_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(0, 8),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v22: vision LoRA Mid (9~17), same dataset/action contract as v16
    TrainConfig(
        name="pi05_e6_v22_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(9, 17),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v23: vision LoRA Late (18~26), same dataset/action contract as v16
    TrainConfig(
        name="pi05_e6_v23_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(18, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v24: vision LoRA 전체 (0~26), same dataset/action contract as v16
    TrainConfig(
        name="pi05_e6_v24_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(0, 26),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v25: vision LoRA Mid/Late 경계 (15~19), v19(14~18) 대비 1칸 shift
    TrainConfig(
        name="pi05_e6_v25_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(15, 19),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
    # v26: vision LoRA Late 상위 (22~25), 7500step 미완료
    TrainConfig(
        name="pi05_e6_v26_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora_r16",
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            vision_lora_layer_range=(22, 25),
        ),
        data=LeRobotE6DataConfig(
            repo_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=AssetsConfig(
                asset_id="Kyle-Riss/dobot_e6_pick_place_orange_v16",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=7_500,
        batch_size=1,
        log_interval=50,
        freeze_filter=pi0_config.freeze_filter_vlm_frozen_vision_and_action_lora(),
        ema_decay=None,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
