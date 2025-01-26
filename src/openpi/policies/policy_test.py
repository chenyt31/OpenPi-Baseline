from openpi_client import action_chunk_broker
import pytest

from openpi.models import exported as _exported
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def create_exported_config() -> _policy_config.PolicyConfig:
    model = _exported.PiModel("s3://openpi-assets/exported/pi0_base/model")

    return _policy_config.PolicyConfig(
        model=model,
        norm_stats=model.norm_stats("trossen_biarm_single_base_cam_24dim"),
        input_layers=[aloha_policy.AlohaInputs(action_dim=model.action_dim)],
        output_layers=[aloha_policy.AlohaOutputs()],
    )


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "s3://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_infer_exported():
    config = create_exported_config()
    policy = _policy_config.create_policy(config)

    example = aloha_policy.make_aloha_example()
    outputs = policy.infer(example)

    assert outputs["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "s3://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
