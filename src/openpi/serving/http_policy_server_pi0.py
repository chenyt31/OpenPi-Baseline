import logging

import tyro

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.serving import http_policy_server


def main(
    port: int = 8000,
) -> None:
    logging.info("Loading model...")
    model = aloha_policy.load_pi0_model()

    logging.info("Creating policy...")
    policy = _policy.ActionChunkBroker(
        aloha_policy.create_aloha_policy(model),
        # Only execute the first half of the chunk.
        action_horizon=model.action_horizon // 2,
    )

    logging.info("Creating server...")
    server = http_policy_server.HttpPolicyServer(policy=policy, host="0.0.0.0", port=port)

    logging.info("Serving...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
