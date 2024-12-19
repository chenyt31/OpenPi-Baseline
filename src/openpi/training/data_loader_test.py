import dataclasses

import jax

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class LeRobotRepack(_transforms.DataTransformFn):
    def __call__(self, item) -> dict:
        return {
            "images": {"cam_high": item["observation.images.top"]},
            "state": item["observation.state"],
            "actions": item["action"],
        }


def test_simple_data_loader():
    model = _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(model, 10)

    loader = _data_loader.simple_data_loader(dataset, local_batch_size=4, max_batches=2)
    batches = list(loader)

    assert len(batches) == 2
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[0]))
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[1]))


def test_data_with_fake_dataset():
    config = _config.get_config("debug")
    model = config.create_model()

    loader = _data_loader.create_data_loader(config, model, skip_norm_stats=True, max_batches=2)

    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.action_horizon, config.action_dim)


def test_data_loader():
    config = _config.get_config("pi0_pretrained")
    config = dataclasses.replace(config, batch_size=4)

    model = config.create_model()

    loader = _data_loader.create_data_loader(
        config,
        model,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        max_batches=2,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)
    assert len(batches) == 2
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[0]))
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[1]))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.action_horizon, config.action_dim)
