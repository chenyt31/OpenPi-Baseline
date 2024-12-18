import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0


def make_from_spec(spec: jax.ShapeDtypeStruct):
    return jnp.zeros(shape=spec.shape, dtype=spec.dtype)


def test_model():
    model = _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)

    batch_size = 2
    observation_spec, action_spec = _model.create_inputs_spec(model, batch_size=batch_size)

    observation = jax.tree.map(make_from_spec, observation_spec)
    actions = jax.tree.map(make_from_spec, action_spec)

    rng = jax.random.key(0)
    params = model.init_params(rng, observation, actions)
    model = model.set_params(params)

    loss = model.compute_loss(rng, observation, actions)
    assert loss.shape == ()

    actions = model.sample_actions(rng, observation, num_steps=10)
    assert actions.shape == (batch_size, 50, 24)
