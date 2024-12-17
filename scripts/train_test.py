import dataclasses
import pathlib

import pytest

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug", "pi0_small"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config.CONFIGS[config_name],
        batch_size=2,
        checkpoint_dir=tmp_path / "checkpoint",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
