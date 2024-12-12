import enum
import logging

from etils import epath
import tyro

from openpi.models import exported as _exported
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server


class ModelMode(enum.Enum):
    LIVE = "live"
    REF = "ref"
    SIM = "sim"


def create_model(mode: ModelMode) -> tuple[_model.BaseModel, aloha_policy.PolicyConfig]:
    model: _model.BaseModel
    config: aloha_policy.PolicyConfig

    match mode:
        case ModelMode.LIVE:
            model = aloha_policy.load_pi0_model()
            config = aloha_policy.PolicyConfig(
                norm_stats=aloha_policy.make_aloha_norm_stats(),
                delta_action_mask=aloha_policy.make_bool_mask(6, -1, 6, -1),
            )
        case ModelMode.REF:
            ckpt_path = epath.Path("checkpoints/pi0_real/model").resolve()
            model = _exported.PiModel.from_checkpoint(ckpt_path)
            config = aloha_policy.PolicyConfig(
                norm_stats=_exported.import_norm_stats(ckpt_path, "trossen_biarm_single_base_cam_24dim"),
                delta_action_mask=aloha_policy.make_bool_mask(6, -1, 6, -1),
            )
        case ModelMode.SIM:
            ckpt_path = epath.Path("checkpoints/pi0_sim/model").resolve()
            model = _exported.PiModel.from_checkpoint(ckpt_path)
            config = aloha_policy.PolicyConfig(
                norm_stats=_exported.import_norm_stats(ckpt_path, "huggingface_aloha_sim_transfer_cube"),
                # The model was fine-tuned on the original aloha data.
                adapt_to_pi=False,
            )

    return model, config


def main(
    port: int = 8000,
    *,
    record: bool = False,
    mode: ModelMode = ModelMode.SIM,
    default_prompt: str = "transfer cube",
) -> None:
    logging.info("Loading model...")
    model, config = create_model(mode)
    config.default_prompt = default_prompt

    logging.info("Creating policy...")
    policy: _policy.BasePolicy = _policy.ActionChunkBroker(
        aloha_policy.create_aloha_policy(model, config),
        action_horizon=model.action_horizon // 2,
    )

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
