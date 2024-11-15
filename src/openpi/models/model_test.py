import jax

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import tokenizer as _tokenizer


def test_model():
    batch_size = 8
    batch = _model.make_example_batch(batch_size)

    assert batch["actions"].shape == (batch_size, 50, 24)

    rng = jax.random.key(0)
    model = _model.Model(pi0.Module(), tokenizer=_tokenizer.PaligemmaTokenizer())
    model = model.init_params(rng, batch)

    loss = model.compute_loss(rng, batch)
    assert loss.shape == (batch_size, 50)

    actions = model.sample_actions(rng, batch)
    assert actions.shape == (batch_size, 50, 24)
