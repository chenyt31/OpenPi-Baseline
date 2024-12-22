import pathlib

import jax
import jax.numpy as jnp

from openpi.models import exported
from openpi.models import pi0


def test_sample_actions():
    model = exported.PiModel.from_checkpoint("s3://openpi-assets-internal/checkpoints/pi0_sim/model")
    actions = model.sample_actions(jax.random.key(0), model.fake_obs(), num_steps=10)

    assert actions.shape == (1, model.action_horizon, model.action_dim)


def test_exported_as_pi0():
    pi_model = exported.PiModel.from_checkpoint("s3://openpi-assets-internal/checkpoints/pi0_sim/model")
    model = pi_model.set_module(pi0.Module(), param_path="decoder")

    key = jax.random.key(0)
    obs = model.fake_obs()

    pi_actions = pi_model.sample_actions(key, obs, num_steps=10)
    actions = model.sample_actions(key, obs, num_steps=10)

    assert pi_actions.shape == (1, model.action_horizon, model.action_dim)
    assert actions.shape == (1, model.action_horizon, model.action_dim)

    diff = jnp.max(jnp.abs(pi_actions - actions))
    assert diff < 10.0


def test_convert_to_openpi(tmp_path: pathlib.Path):
    output_dir = tmp_path / "output"

    exported.convert_to_openpi(
        "s3://openpi-assets-internal/checkpoints/pi0_sim/model",
        "huggingface_aloha_sim_transfer_cube",
        output_dir,
    )

    assert (output_dir / "params").exists()
    assert (output_dir / "assets").exists()


def test_exported_droid():
    model = exported.PiModel.from_checkpoint(
        "s3://openpi-assets-internal/checkpoints/gemmamix_dct_dec5_droid_dec8_1008am/340000/model"
    )
    actions = model.sample_actions(jax.random.key(0), model.fake_obs(), num_denoising_steps=10)

    assert actions.shape == (1, model.action_horizon, model.action_dim)
