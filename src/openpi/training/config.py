import dataclasses
from typing import Annotated, Union

import namer
import tyro

from openpi.models import common
import openpi.models.pi0 as pi0
import openpi.models.pi0_small as pi0_small
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    keep_interval: int = 5000
    module: common.BaseModule = dataclasses.field(default_factory=pi0.Module)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = None
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    project_name: str = "openpi"
    exp_name: str = namer.generate(category=["food", "technology"], suffix_length=3)  # noqa: RUF009
    checkpoint_dir: str = "/tmp/openpi/checkpoints"
    seed: int = 42
    batch_size: int = 16
    num_train_steps: int = 2_000_000
    log_interval: int = 100
    save_interval: int = 1000

    overwrite: bool = False
    resume: bool = False


CONFIGS = {
    "default": TrainConfig(),
    "debug": TrainConfig(batch_size=2, module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy")),
    "debug_restore": TrainConfig(
        batch_size=2,
        module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader(""),
    ),
    "paligemma": TrainConfig(
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        # module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
    ),
    "pi0_small": TrainConfig(
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
    ),
}


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
                for k, v in CONFIGS.items()
            )
        ),
    )
