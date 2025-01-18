import pathlib

import jax
import jax.numpy as jnp
import pytest

import openpi.models.exported as exported
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.training.checkpoints as _checkpoints


@pytest.mark.manual
def test_sample_actions():
    model = exported.PiModel("s3://openpi-assets/exported/pi0_base/model")
    actions = model.sample_actions(jax.random.key(0), model.fake_obs(), num_steps=10)

    assert actions.shape == (1, model.action_horizon, model.action_dim)


@pytest.mark.manual
def test_exported_as_pi0():
    exported_model = exported.PiModel("s3://openpi-assets/exported/pi0_base/model")
    live_model = pi0.Pi0Config().load(exported_model.params["decoder"])

    key = jax.random.key(0)
    obs = exported_model.fake_obs()

    exported_actions = exported_model.sample_actions(key, obs, num_steps=10)
    live_actions = live_model.sample_actions(key, obs, num_steps=10)

    assert exported_actions.shape == (1, exported_model.action_horizon, exported_model.action_dim)
    assert live_actions.shape == (1, live_model.action_horizon, live_model.action_dim)

    diff = jnp.max(jnp.abs(exported_actions - live_actions))
    assert diff < 10.0


@pytest.mark.manual
def test_processor_loading():
    pi_model = exported.PiModel("s3://openpi-assets/exported/pi0_base/model")
    assert "trossen_biarm_single_base_cam_24dim" in pi_model.processor_names()

    norm_stats = pi_model.norm_stats("trossen_biarm_single_base_cam_24dim")
    assert sorted(norm_stats) == ["actions", "state"]


@pytest.mark.manual
def test_convert_to_openpi(tmp_path: pathlib.Path):
    output_dir = tmp_path / "output"

    exported.convert_to_openpi(
        "s3://openpi-assets/exported/pi0_base/model",
        output_dir,
        processor="trossen_biarm_single_base_cam_24dim",
    )

    # Make sure that we can load the params and norm stats.
    _ = _model.restore_params(output_dir / "params")
    _ = _checkpoints.load_norm_stats(output_dir / "assets")
