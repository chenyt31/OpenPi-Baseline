import jax
from lerobot.common.datasets import lerobot_dataset

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0
from openpi.policies import aloha_policy
from openpi.training import data_loader as _data_loader


class LeRobotRepack(_transforms.DataTransformFn):
    def __call__(self, item) -> dict:
        return {
            "images": {"cam_high": item["observation.images.top"]},
            "state": item["observation.state"],
            "actions": item["action"],
        }


def test_data_loader():
    model = _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(model, 10)

    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    loader = _data_loader.data_loader(dataset, local_batch_size=4, sharding=sharding, max_batches=2)

    batches = list(loader)

    assert len(batches) == 2
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[0]))
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[1]))


def test_data_loader_lerobot():
    repo_id = "lerobot/aloha_sim_transfer_cube_human"
    action_horizon = 50

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id, delta_timestamps={"action": [t / dataset_meta.fps for t in range(action_horizon)]}
    )

    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    batch_size = 4
    loader = _data_loader.data_loader(
        dataset,
        local_batch_size=batch_size,
        sharding=sharding,
        max_batches=2,
        transforms=[
            LeRobotRepack(),
            aloha_policy.AlohaInputs(action_dim=24),
        ],
    )

    batches = list(loader)
    assert len(batches) == 2
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[0]))
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(batches[1]))

    assert batches[0]["actions"].shape == (batch_size, action_horizon, 24)
