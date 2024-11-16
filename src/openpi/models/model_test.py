import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0


def make_from_spec(spec: jax.ShapeDtypeStruct):
    return jnp.zeros(shape=spec.shape, dtype=spec.dtype)


def test_model():
    model = _model.Model(pi0.Module())

    batch_size = 8
    observation_spec, action_spec = _model.create_inputs_spec(model, batch_size=batch_size)

    observation = jax.tree.map(make_from_spec, observation_spec)
    actions = jax.tree.map(make_from_spec, action_spec)

    rng = jax.random.key(0)
    model = model.init_params(rng, observation, actions)

    loss = model.compute_loss(rng, observation, actions)
    assert loss.shape == (batch_size, 50)

    actions = model.sample_actions(rng, observation)
    assert actions.shape == (batch_size, 50, 24)
