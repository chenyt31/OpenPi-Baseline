import jax

from openpi.models import common
from openpi.models import model as _model
from openpi.models import pi0


def test_model():
    batch_size = 8
    batch = _model.make_example_batch(batch_size)

    assert batch["actions"].shape == (batch_size, 50, 24)

    observation = common.Observation.from_dict(batch)

    rng = jax.random.key(0)

    model = _model.Model(pi0.Module())
    model = model.init_params(rng, observation, batch["actions"])

    loss = model.compute_loss(rng, observation, batch["actions"])
    assert loss.shape == (batch_size, 50)

    actions = model.sample_actions(rng, observation)
    assert actions.shape == (batch_size, 50, 24)
