from etils import epath
import jax

from openpi.models import exported


def test_sample_actions():
    path = epath.Path("checkpoints/pi0_sim/model").resolve()

    model = exported.PiModel.from_checkpoint(path)
    actions = model.sample_actions(jax.random.key(0), model.fake_obs())

    assert actions.shape == (1, model.action_horizon, model.action_dim)
