import pathlib

from openpi.models import exported
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_make_bool_mask():
    assert _policy_config.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _policy_config.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_create_trained_policy(tmp_path: pathlib.Path):
    ckpt_dir = tmp_path / "ckpt"

    # TODO(ury): Replace with an openpi checkpoint once we stop using exported models.
    exported.convert_to_openpi(
        "s3://openpi-assets/exported/pi0_aloha_sim/model",
        "huggingface_aloha_sim_transfer_cube",
        ckpt_dir,
    )

    # Make sure that we can load the policy.
    policy = _policy_config.create_trained_policy(_config.get_config("debug"), ckpt_dir)
    assert policy is not None
