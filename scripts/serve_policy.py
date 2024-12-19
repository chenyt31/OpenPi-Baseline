import enum
import logging
from typing import Any

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
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class ModelMode(enum.Enum):
    LIVE = "live"
    REF = "ref"
    SIM = "sim"
    DROID = "droid"
    CALVIN = "calvin"
    LIBERO = "libero"


def load_trained_policy(
    config_name: str,
    checkpoint_path: str,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
) -> _policy.Policy:
    repack_transforms = repack_transforms or transforms.Group()
    config = _config.get_config(config_name)

    logging.info("Loading model...")
    model = config.create_model()
    model = model.set_params(_model.restore_params(checkpoint_path))

    data_config = config.data.create(config.metadata_dir, model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_path)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
    )


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
            ckpt_path = "checkpoints/gemmamix_nov4_droid_no22_1056am/290000/model"
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
                    transforms.SubsampleActions(stride=5),
                ],
                sample_kwargs={"num_steps": 10},
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
                sample_kwargs={"num_steps": 10},
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
                sample_kwargs={
                    "num_steps": 10,
                },
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
