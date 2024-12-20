import enum
import logging

from etils import epath
import tyro

from openpi.models import exported as _exported
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.serving import http_policy_server


class Mode(enum.Enum):
    LIVE = "live"
    REF = "ref"
    SIM = "sim"


def create_model(mode: Mode) -> tuple[_model.BaseModel, dict]:
    model: _model.BaseModel
    match mode:
        case Mode.LIVE:
            model = aloha_policy.load_pi0_model()
            norm_stats = aloha_policy.make_aloha_norm_stats()
        case Mode.REF:
            ckpt_path = epath.Path("checkpoints/pi0_real/model").resolve()
            # ckpt_path = epath.Path("checkpoints/myckpt/30000/model").resolve()
            # ckpt_path = epath.Path("checkpoints/gemmamix_scratch_aloha_delta_actions30k/model").resolve() 
            ckpt_path = epath.Path("checkpoints/gemmamix_all_aloha_tasks_delta_actions70k/model").resolve()
            # ckpt_path = epath.Path("checkpoints/gemmamix_finetuned_towelfold20k/model").resolve()
            # ckpt_path = epath.Path("checkpoints/gemmamix_finetuned_buildabox20k/model").resolve()
            model = _exported.PiModel.from_checkpoint(ckpt_path)
            # norm_stats = _exported.import_norm_stats(ckpt_path, "trossen_biarm_single_base_cam_24dim")
            # norm_stats = _exported.import_norm_stats(ckpt_path, "berkeley_aloha")
            # norm_stats = _exported.import_norm_stats(ckpt_path, "all_berkeley_aloha_data") # works great
            norm_stats = _exported.import_norm_stats(ckpt_path, "all_berkeley_aloha_data_delta_actions")
            # norm_stats = _exported.import_norm_stats(ckpt_path, "trossen_bimanual_diverse_small_towel_folding_random_None_joint")
            # norm_stats = _exported.import_norm_stats(ckpt_path, "trossen_bimanual_build_a_box_random_None")
        case Mode.SIM:
            ckpt_path = epath.Path("checkpoints/pi0_sim/model").resolve()
            model = _exported.PiModel.from_checkpoint(ckpt_path)
            norm_stats = _exported.import_norm_stats(ckpt_path, "huggingface_aloha_sim_transfer_cube")

    return model, norm_stats


def main(
    port: int = 8000,
    *,
    record: bool = False,
    mode: Mode = Mode.REF,
    default_prompt: str = "uncap the pen",
    # default_prompt: str = "place corn on the pink tray",
    # default_prompt: str = "toast_out_of_toaster",
) -> None:

    logging.info("Loading model...")
    model, norm_stats = create_model(mode)

    logging.info("Creating policy...")
    policy: _policy.BasePolicy = _policy.ActionChunkBroker(
        aloha_policy.create_aloha_policy(model, norm_stats, default_prompt=default_prompt),
        # Only execute the first half of the chunk.
        action_horizon=model.action_horizon // 2,
    )

    # Record the policy's behavior.
    if record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    logging.info("Creating server...")
    server = http_policy_server.HttpPolicyServer(policy=policy, host="0.0.0.0", port=port)

    logging.info("Serving...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
