import logging

from etils import epath
import tyro

from openpi.models import exported as _exported
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.serving import http_policy_server


def main(port: int = 8000, *, record: bool = False, exported: bool = True) -> None:
    logging.info("Loading model...")
    model: _model.BaseModel
    if exported:
        ckpt_path = epath.Path("checkpoints/pi0_sim/model").resolve()
        model = _exported.PiModel.from_checkpoint(ckpt_path)
        norm_stats = _exported.import_norm_stats(ckpt_path, "huggingface_aloha_sim_transfer_cube")
    else:
        model = aloha_policy.load_pi0_model()
        norm_stats = aloha_policy.make_aloha_norm_stats()

    logging.info("Creating policy...")
    policy: _policy.BasePolicy = _policy.ActionChunkBroker(
        aloha_policy.create_aloha_policy(model, norm_stats),
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
