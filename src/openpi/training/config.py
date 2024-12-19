from collections.abc import Sequence
import dataclasses
import difflib
import os
import pathlib
from typing import Annotated, Any, Protocol, Union

import namer
import tyro

import openpi.models.common as common
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_small as pi0_small
import openpi.models.tokenizer as _tokenizer
from openpi.policies import aloha_policy
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


def default_dataset_root() -> str | None:
    # TODO(ury): Temporary, remove this once the default works well.
    if os.path.exists("/mnt/weka"):  # noqa: PTH110
        return "/mnt/weka/lerobot"
    return None


@dataclasses.dataclass
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Contains precomputed normalization stats.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Indicates where the cached dataset should be stored.
    # This can also be controlled by setting the LEROBOT_HOME environment variable.
    dataset_root: str | None = dataclasses.field(default_factory=default_dataset_root)


class DataConfigFactory(Protocol):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        """Create a data config."""


class FakeDataConfig(DataConfigFactory):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        return DataConfig(repo_id="fake")


class LeRobotRepack(_transforms.DataTransformFn):
    def __call__(self, item) -> dict:
        return {
            "images": {"cam_high": item["observation.images.top"]},
            "state": item["observation.state"],
            "actions": item["action"],
        }


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # The LeRobot repo id.
    repo_id: str
    # The delta action mask. Each value corresponds to an action dimension and indicates if it should be converted to a delta action.
    # If None, absolute actions are used.
    delta_action_mask: Sequence[bool] | None = None
    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # If true, will adapt the joint and gripper values to match the pi runtime. This useful when
    # fine-tuning a pretrained model.
    adopt_to_pi: bool = False

    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        norm_stats_path = metadata_dir / self.repo_id / "norm_stats.json"
        norm_stats = _normalize.deserialize_json(norm_stats_path.read_text()) if norm_stats_path.exists() else None

        return DataConfig(
            repo_id=self.repo_id,
            norm_stats=norm_stats,
            repack_transforms=_transforms.Group(
                inputs=[
                    LeRobotRepack(),
                ]
            ),
            data_transforms=_transforms.Group(
                inputs=[
                    aloha_policy.AlohaInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=self.delta_action_mask,
                        adapt_to_pi=self.adopt_to_pi,
                    ),
                ],
                outputs=[
                    aloha_policy.AlohaOutputs(
                        delta_action_mask=self.delta_action_mask,
                        adapt_to_pi=self.adopt_to_pi,
                    ),
                ],
            ),
            model_transforms=_transforms.Group(
                inputs=[
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model.max_token_len),
                        default_prompt=self.default_prompt,
                    ),
                ]
            ),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = namer.generate(category=["food", "technology"], suffix_length=3)  # noqa: RUF009

    action_dim: int = 24
    action_horizon: int = 50
    max_token_len: int = 48

    module: common.BaseModule = dataclasses.field(default_factory=pi0.Module)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = None

    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    metadata_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 16
    num_train_steps: int = 100_000

    log_interval: int = 100
    save_interval: int = 1000
    keep_interval: int = 5000

    overwrite: bool = False
    resume: bool = False

    # Keyword arguments to pass to the policy's sample method.
    sample_kwargs: dict[str, Any] | None = None

    @property
    def metadata_dir(self) -> pathlib.Path:
        """Get the metadata directory for this config."""
        return (pathlib.Path(self.metadata_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def create_model(self) -> _model.Model:
        return _model.Model(
            module=self.module,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
        )


_CONFIGS = [
    #
    # pi0 configs.
    #
    TrainConfig(
        name="pi0",
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            delta_action_mask=None,
        ),
    ),
    TrainConfig(
        name="pi0_pretrained",
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            delta_action_mask=None,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi0_paligemma",
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            delta_action_mask=None,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
    ),
    #
    # pi0_small configs.
    #
    TrainConfig(
        name="pi0_small",
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        batch_size=2,
        module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
    ),
    TrainConfig(
        name="debug_restore",
        batch_size=2,
        module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
        resume=True,
        exp_name="debug",
        num_train_steps=10,
    ),
]

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.cli(
        Union.__getitem__(  # type: ignore
            tuple(
                Annotated.__class_getitem__(  # type: ignore
                    (
                        Annotated.__class_getitem__((type(v), tyro.conf.AvoidSubcommands)),  # type: ignore
                        tyro.conf.subcommand(k, default=v),
                    )
                )
                for k, v in _CONFIGS_DICT.items()
            )
        ),
    )


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f"Did you mean '{closest[0]}'? " if closest else ""
        if closest:
            raise ValueError(f"Config '{config_name}' not found. Did you mean '{closest_str}'?")
        raise ValueError(f"Config '{config_name}' not found.")

    return _CONFIGS_DICT[config_name]
