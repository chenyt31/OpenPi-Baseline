"""See _CONFIGS for the list of available configs."""

from collections.abc import Sequence
import dataclasses
import difflib
import pathlib
from typing import Any, Protocol, runtime_checkable

import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


def default_dataset_root() -> str:
    """Default location for the dataset cache."""
    return str(download.get_cache_dir() / "datasets")


ModelType = _model.ModelType


@dataclasses.dataclass
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
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

    # Indicates where the cached dataset should be stored. If None, the default directory will be used.
    dataset_root: str | None = dataclasses.field(default_factory=default_dataset_root)

    # If true, will disable syncing the dataset from the huggingface hub. Allows training on local-only datasets.
    local_files_only: bool = False


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
            case ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            default_prompt=self.default_prompt,
                        ),
                    ],
                )
            case ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            default_prompt=self.default_prompt,
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@runtime_checkable
class DataConfigFactory(Protocol):
    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def load_norm_stats(self, metadata_dir: pathlib.Path, repo_id: str) -> dict[str, _transforms.NormStats] | None:
        norm_stats_path = metadata_dir / repo_id / "norm_stats.json"
        return _normalize.deserialize_json(norm_stats_path.read_text()) if norm_stats_path.exists() else None


class FakeDataConfig(DataConfigFactory):
    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id="fake")


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory]
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if (repo_id := self.repo_id) is not tyro.MISSING:
            norm_stats = self.load_norm_stats(metadata_dir, repo_id)
        else:
            repo_id, norm_stats = None, None

        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            norm_stats=norm_stats,
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = False
    # If provided, will determine the default prompt that be used by the model.
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

    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    def create(self, metadata_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if (repo_id := self.repo_id) is not tyro.MISSING:
            norm_stats = self.load_norm_stats(metadata_dir, repo_id)
        else:
            repo_id, norm_stats = None, None

        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
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
            self.base_config or DataConfig(),
            repo_id=repo_id,
            norm_stats=norm_stats,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
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
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for metadata (e.g., norm stats).
    metadata_base_dir: str = "./assets"
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
    def metadata_dir(self) -> pathlib.Path:
        """Get the metadata directory for this config."""
        return (pathlib.Path(self.metadata_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(action_dim=24),
        data=LeRobotAlohaDataConfig(use_delta_joint_actions=True),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_aloha_towel_diverse",
        model=pi0.Pi0Config(action_dim=24),
        data=LeRobotAlohaDataConfig(use_delta_joint_actions=True),
        num_train_steps=30_000,
        policy_metadata={
            # Adjust the reset pose to align better with the internal training data.
            "reset_pose": [0, -1.5, 1.5, 0, 0, 0]
        },
    ),
    #
    # DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=15, max_token_len=180),
        data=SimpleDataConfig(
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
        ),
        num_train_steps=30_000,
    ),
    #
    # Simulation configs.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(action_dim=24),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            # TODO(ury): Retrain the aloha sim model and remove the adapt_to_pi flag.
            adapt_to_pi=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_aloha_sim",
        model=pi0_fast.Pi0FASTConfig(action_horizon=50, action_dim=14, max_token_len=256),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero",
        model=pi0.Pi0Config(action_dim=7, action_horizon=10),
        data=SimpleDataConfig(
            data_transforms=lambda model: _transforms.Group(
                inputs=[libero_policy.LiberoInputs(action_dim=model.action_dim)],
                outputs=[libero_policy.LiberoOutputs()],
            ),
        ),
        num_train_steps=30_000,
    ),
    #
    # Additional configs.
    #
    TrainConfig(
        name="pi0_paligemma",
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
    ),
    #
    # Example configs.
    #
    TrainConfig(
        name="aloha_static_cups_open",
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_static_cups_open",
            use_delta_joint_actions=True,
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
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            base_config=DataConfig(local_files_only=False),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
        ),
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
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
