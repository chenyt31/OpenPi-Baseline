import dataclasses

import openpi.training.optimizer as _optimizer


@dataclasses.dataclass
class TrainConfig:
    keep_interval: int = 5000
    lr_schedule: str | _optimizer.ScheduleProvider = "cosine_decay"
    optimizer: str | _optimizer.OptimizerProvider = "adamw"
    load_pretrained_weights: str | None = None
    checkpoint_dir: str = "/tmp/openpi/checkpoints"
    seed: int = 42
    batch_size: int = 16
    num_train_steps: int = 2_000_000
    log_interval: int = 100
    save_interval: int = 1000
