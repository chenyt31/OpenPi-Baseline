import dataclasses
from typing import Annotated, Union

import tyro

import openpi.training.optimizer as _optimizer


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    keep_interval: int = 5000
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = None
    load_pretrained_weights: str | None = None
    checkpoint_dir: str = "/tmp/openpi/checkpoints"
    seed: int = 42
    batch_size: int = 16
    num_train_steps: int = 2_000_000
    log_interval: int = 100
    save_interval: int = 1000

    overwrite: bool = False
    resume: bool = False


_CONFIGS = {
    "default": TrainConfig(),
    "large": TrainConfig(batch_size=128),
}


def cli() -> TrainConfig:
    return tyro.cli(
        Union.__getitem__(  # type: ignore
            tuple(
                Annotated.__class_getitem__(  # type: ignore
                    (
                        Annotated.__class_getitem__((type(v), tyro.conf.AvoidSubcommands)),
                        tyro.conf.subcommand(k, default=v),
                    )
                )
                for k, v in _CONFIGS.items()
            )
        ),
    )
