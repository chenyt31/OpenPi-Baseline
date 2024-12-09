from etils import epath

from openpi.models import exported as _exported
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy


def test_infer():
    model = aloha_policy.load_pi0_model()

    policy = aloha_policy.create_aloha_policy(
        model,
        aloha_policy.PolicyConfig(norm_stats=aloha_policy.make_aloha_norm_stats()),
    )

    example = aloha_policy.make_aloha_example()
    outputs = policy.infer(example)

    assert outputs["qpos"].shape == (model.action_horizon, 14)


def test_exported():
    ckpt_path = epath.Path("checkpoints/pi0_sim/model").resolve()
    model = _exported.PiModel.from_checkpoint(ckpt_path)

    policy = aloha_policy.create_aloha_policy(
        model,
        aloha_policy.PolicyConfig(
            norm_stats=_exported.import_norm_stats(ckpt_path, "huggingface_aloha_sim_transfer_cube"),
            adapt_to_pi=False,
        ),
    )

    example = aloha_policy.make_aloha_example()
    outputs = policy.infer(example)

    assert outputs["qpos"].shape == (model.action_horizon, 14)


def test_broker():
    model = aloha_policy.load_pi0_model()

    policy = _policy.ActionChunkBroker(
        aloha_policy.create_aloha_policy(
            model,
            aloha_policy.PolicyConfig(norm_stats=aloha_policy.make_aloha_norm_stats()),
        ),
        # Only execute the first half of the chunk.
        action_horizon=model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(model.action_horizon):
        outputs = policy.infer(example)
        assert outputs["qpos"].shape == (14,)
