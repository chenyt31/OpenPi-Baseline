import jax

from openpi.models import exported
from openpi.models import pi0


def test_sample_actions():
    model = exported.PiModel.from_checkpoint("s3://openpi-assets-internal/checkpoints/pi0_sim/model")
    actions = model.sample_actions(jax.random.key(0), model.fake_obs(), num_steps=10)

    assert actions.shape == (1, model.action_horizon, model.action_dim)


def test_exported_as_pi0():
    model = exported.model_from_checkpoint(
        pi0.Module(),
        "s3://openpi-assets-internal/checkpoints/pi0_sim/model",
        param_path="decoder",
    )
    actions = model.sample_actions(jax.random.key(0), model.fake_obs(), num_steps=10)

    assert actions.shape == (1, model.action_horizon, model.action_dim)


def test_exported_droid():
    model = exported.PiModel.from_checkpoint(
        "s3://openpi-assets-internal/checkpoints/gemmamix_dct_dec5_droid_dec8_1008am/340000/model"
    )
    actions = model.sample_actions(jax.random.key(0), model.fake_obs(), num_denoising_steps=10)

    assert actions.shape == (1, model.action_horizon, model.action_dim)
