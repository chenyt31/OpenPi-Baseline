from openpi.policies import aloha_policy
from openpi.policies import policy as _policy


def test_infer():
    model = aloha_policy.load_pi0_model()
    policy = aloha_policy.create_aloha_policy(model)

    outputs = policy.infer(aloha_policy.make_aloha_example())
    assert outputs["action/qpos"].shape == (model.action_horizon, 14)


def test_broker():
    model = aloha_policy.load_pi0_model()
    policy = _policy.ActionChunkBroker(
        aloha_policy.create_aloha_policy(model),
        # Only execute the first half of the chunk.
        action_horizon=model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(model.action_horizon):
        outputs = policy.infer(example)
        assert outputs["action/qpos"].shape == (14,)
