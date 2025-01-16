import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.shared import download


def make_from_spec(spec: jax.ShapeDtypeStruct):
    return jnp.zeros(shape=spec.shape, dtype=spec.dtype)


def create_pi0_config():
    return pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)


def test_model():
    config = create_pi0_config()
    model = config.create(jax.random.key(0))

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = model.compute_loss(jax.random.key(0), obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(jax.random.key(0), obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


@pytest.mark.manual
def test_model_restore():
    config = create_pi0_config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(jax.random.key(0), obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(jax.random.key(0), obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


if __name__ == "__main__":
    test_model_restore()
