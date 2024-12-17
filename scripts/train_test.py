import dataclasses
import pathlib

from openpi.training import config as _config

from . import train


def test_train(tmp_path: pathlib.Path):
    config = dataclasses.replace(
        _config.CONFIGS["debug"],
        checkpoint_dir=tmp_path / "checkpoint",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
