import enum
import logging

import tyro

from openpi.models import exported as _exported
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import calvin_policy
from openpi.policies import droid_policy
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server


class ModelMode(enum.Enum):
    LIVE = "live"
    REF = "ref"
    SIM = "sim"
    DROID = "droid"
    CALVIN = "calvin"
    LIBERO = "libero"


def create_policy(mode: ModelMode, default_prompt: str) -> _policy.Policy:
    model: _model.BaseModel
    config: _policy_config.PolicyConfig

    match mode:
        case ModelMode.LIVE:
            logging.info("Loading model...")
            model = aloha_policy.load_pi0_model()

            logging.info("Creating policy...")
            delta_action_mask = _policy_config.make_bool_mask(6, -1, 6, -1)
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=aloha_policy.make_aloha_norm_stats(),
                default_prompt=default_prompt,
                input_layers=[
                    aloha_policy.ActInputsRepack(),
                    aloha_policy.AlohaInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=delta_action_mask,
                        adapt_to_pi=True,
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
        case ModelMode.REF:
            logging.info("Loading model...")
            ckpt_path = "checkpoints/pi0_real/model"
            model = _exported.PiModel.from_checkpoint(ckpt_path)

            logging.info("Creating policy...")
            delta_action_mask = _policy_config.make_bool_mask(6, -1, 6, -1)
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=_exported.import_norm_stats(ckpt_path, "trossen_biarm_single_base_cam_24dim"),
                default_prompt=default_prompt,
                input_layers=[
                    aloha_policy.ActInputsRepack(),
                    aloha_policy.AlohaInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=delta_action_mask,
                        adapt_to_pi=True,
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
        case ModelMode.SIM:
            logging.info("Loading model...")
            ckpt_path = "checkpoints/pi0_sim/model"
            model = _exported.PiModel.from_checkpoint(ckpt_path)

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=_exported.import_norm_stats(ckpt_path, "huggingface_aloha_sim_transfer_cube"),
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
        case ModelMode.DROID:
            logging.info("Loading model...")
            ckpt_path = "checkpoints/gemmamix_dct_dec5_droid_dec8_1008am/340000/model"
            model = _exported.PiModel.from_checkpoint(ckpt_path)

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=_exported.import_norm_stats(ckpt_path, "openx_droid"),
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
                ],
                sample_kwargs={"num_denoising_steps": 10},
            )
        case ModelMode.CALVIN:
            logging.info("Loading model...")
            ckpt_path = "checkpoints/release_gemmamix_calvin_nov24_2053/40000/model"
            model = _exported.PiModel.from_checkpoint(ckpt_path)

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=_exported.import_norm_stats(ckpt_path, "calvin"),
                default_prompt=default_prompt,
                input_layers=[
                    calvin_policy.CalvinInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    calvin_policy.CalvinOutputs(),
                ],
                sample_kwargs={"num_denoising_steps": 10},
            )
        case ModelMode.LIBERO:
            logging.info("Loading model...")
            ckpt_path = "checkpoints/release_gemmamix_libero_nov23_1443/40000/model"
            model = _exported.PiModel.from_checkpoint(ckpt_path)

            logging.info("Creating policy...")
            config = _policy_config.PolicyConfig(
                model=model,
                norm_stats=_exported.import_norm_stats(ckpt_path, "libero"),
                default_prompt=default_prompt,
                input_layers=[
                    libero_policy.LiberoInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    libero_policy.LiberoOutputs(),
                ],
                sample_kwargs={"num_denoising_steps": 10},
            )
        case _:
            raise ValueError(f"Unknown model mode: {mode}")

    return _policy_config.create_policy(config)


def main(
    port: int = 8000,
    *,
    record: bool = False,
    mode: ModelMode = ModelMode.DROID,
    default_prompt: str = "transfer cube",
) -> None:
    policy = create_policy(mode, default_prompt)

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
