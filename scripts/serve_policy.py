import enum
import logging

import tyro

from openpi import transforms
from openpi.models import exported as _exported
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import calvin_policy
from openpi.policies import droid_policy
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    CALVIN = "calvin"
    LIBERO = "libero"


def repack_from_env(env: EnvMode) -> transforms.Group:
    """Creates environment specific repack transforms."""
    # TODO(ury): Move this to the runtime.
    match env:
        case EnvMode.ALOHA:
            return transforms.Group(
                inputs=[aloha_policy.ActInputsRepack()],
                outputs=[aloha_policy.ActOutputsRepack()],
            )
        case EnvMode.ALOHA_SIM:
            return transforms.Group(
                inputs=[aloha_policy.ActInputsRepack()],
                outputs=[aloha_policy.ActOutputsRepack()],
            )
        case _:
            return transforms.Group()


def create_default_policy(env: EnvMode, default_prompt: str | None, crop_ratio: float | None = None) -> _policy.Policy:
    model: _model.BaseModel
    config: _policy_config.PolicyConfig

    match env:
        case EnvMode.ALOHA:
            logging.info("Loading model...")
            model = _exported.PiModel.from_checkpoint("s3://openpi-assets-internal/checkpoints/pi0_real/model")

            logging.info("Creating policy...")
            delta_action_mask = _policy_config.make_bool_mask(6, -1, 6, -1)
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=model.norm_stats("trossen_biarm_single_base_cam_24dim"),
                default_prompt=default_prompt,
                input_layers=[
                    aloha_policy.ActInputsRepack(),
                    aloha_policy.AlohaInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=delta_action_mask,
                        adapt_to_pi=True,
                        crop_ratio=crop_ratio,
                    ),
                ],
                output_layers=[
                    aloha_policy.AlohaOutputs(
                        delta_action_mask=delta_action_mask,
                        adapt_to_pi=True,
                    ),
                    aloha_policy.ActOutputsRepack(),
                ],
            )
        case EnvMode.ALOHA_SIM:
            logging.info("Loading model...")
            model = _exported.PiModel.from_checkpoint("s3://openpi-assets-internal/checkpoints/pi0_sim/model")

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=model.norm_stats("huggingface_aloha_sim_transfer_cube"),
                default_prompt=default_prompt,
                input_layers=[
                    aloha_policy.ActInputsRepack(),
                    aloha_policy.AlohaInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=None,
                        adapt_to_pi=False,
                    ),
                ],
                output_layers=[
                    aloha_policy.AlohaOutputs(
                        delta_action_mask=None,
                        adapt_to_pi=False,
                    ),
                    aloha_policy.ActOutputsRepack(),
                ],
            )
        case EnvMode.DROID:
            logging.info("Loading model...")
            model = _exported.PiModel.from_checkpoint(
                "s3://openpi-assets-internal/checkpoints/gemmamix_nov4_droid_no22_1056am/290000/model"
            )

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=model.norm_stats("openx_droid"),
                default_prompt=default_prompt,
                input_layers=[
                    droid_policy.DroidInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=None,
                    ),
                ],
                output_layers=[
                    droid_policy.DroidOutputs(
                        delta_action_mask=None,
                    ),
                    transforms.SubsampleActions(stride=5),
                ],
                sample_kwargs={"num_steps": 10},
            )
        case EnvMode.CALVIN:
            logging.info("Loading model...")
            model = _exported.PiModel.from_checkpoint(
                "s3://openpi-assets-internal/checkpoints/release_gemmamix_calvin_nov24_2053/40000/model"
            )

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=model.norm_stats("calvin"),
                default_prompt=default_prompt,
                input_layers=[
                    calvin_policy.CalvinInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    calvin_policy.CalvinOutputs(),
                ],
                sample_kwargs={"num_steps": 10},
            )
        case EnvMode.LIBERO:
            logging.info("Loading model...")
            model = _exported.PiModel.from_checkpoint(
                "s3://openpi-assets-internal/checkpoints/release_gemmamix_libero_nov23_1443/40000/model"
            )

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=model.norm_stats("libero"),
                default_prompt=default_prompt,
                input_layers=[
                    libero_policy.LiberoInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    libero_policy.LiberoOutputs(),
                ],
                sample_kwargs={
                    "num_steps": 10,
                },
            )
        case _:
            raise ValueError(f"Unknown environment mode: {env}")

    return _policy_config.create_policy(config)


def main(
    port: int = 8000,
    *,
    env: EnvMode = EnvMode.ALOHA_SIM,
    config_name: str | None = None,
    checkpoint_path: str | None = None,
    default_prompt: str | None = None,
    record: bool = False,
    crop_ratio: float | None = None,
) -> None:
    """Serve a policy.

    Args:
        env: The environment to serve the policy for.
        config_name: If provided, loads the policy from a training config. Otherwise, loads the default pi0 policy.
        checkpoint_path: Required if `config_name` is provided. Specifies the path to the checkpoint to load.
        default_prompt: If provided, overrides the default prompt for the policy.
        record: Whether to record the policy's behavior.
        crop_ratio: If provided, sets the crop ratio for image preprocessing in ALOHA policy. Only used when env=EnvMode.ALOHA.
    """
    if config_name:
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required if config_name is provided")
        config = _config.get_config(config_name)
        policy = _policy_config.create_trained_policy(
            config,
            checkpoint_path,
            repack_transforms=repack_from_env(env),
            sample_kwargs=config.sample_kwargs,
        )
    else:
        policy = create_default_policy(env, default_prompt, crop_ratio)

    # Record the policy's behavior.
    if record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    logging.info("Creating server...")
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=port)

    logging.info("Serving...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
