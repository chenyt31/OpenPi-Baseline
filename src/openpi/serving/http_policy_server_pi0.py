import logging
import pathlib

import jax
import numpy as np
import tyro

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import tokenizer
from openpi.policies import policy as _policy
from openpi.serving import http_policy_server


def load_pi0_model() -> _model.Model:
    model = _model.Model(pi0.Module(), rng=jax.random.key(0), tokenizer=tokenizer.PaligemmaTokenizer())
    return _model.restore_params(model, pathlib.Path("checkpoints/pi0_base/model").absolute())


def make_aloha_norm_stats():
    """Define the normalization stats for ALOHA."""
    return {
        "actions": transforms.NormStats(
            mean=np.array(
                [
                    -0.0,
                    0.01,
                    0.02,
                    0.0,
                    -0.01,
                    0.0,
                    0.44,
                    -0.0,
                    0.02,
                    0.02,
                    0.0,
                    -0.01,
                    0.0,
                    0.43,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            std=np.array(
                [
                    0.11,
                    0.17,
                    0.16,
                    0.11,
                    0.17,
                    0.15,
                    0.41,
                    0.12,
                    0.19,
                    0.18,
                    0.14,
                    0.19,
                    0.19,
                    0.39,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
        "state": transforms.NormStats(
            mean=np.array(
                [
                    0.11,
                    0.45,
                    -0.6,
                    0.13,
                    0.5,
                    -0.2,
                    0.45,
                    -0.07,
                    0.31,
                    -0.48,
                    -0.13,
                    0.55,
                    0.21,
                    0.44,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            std=np.array(
                [
                    0.3,
                    0.54,
                    0.55,
                    0.28,
                    0.47,
                    0.44,
                    0.38,
                    0.32,
                    0.57,
                    0.56,
                    0.3,
                    0.49,
                    0.54,
                    0.36,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    }


def main(
    port: int = 8000,
) -> None:
    logging.info("Loading model...")
    model = load_pi0_model()

    norm_stats = make_aloha_norm_stats()

    logging.info("Creating policy...")
    policy = _policy.ActionChunkBroker(
        _policy.Policy(
            model,
            transforms=[
                transforms.AlohaInputs(action_dim=model.action_dim),
                transforms.Normalize(norm_stats),
            ],
            output_transforms=[
                transforms.Unnormalize(norm_stats),
                transforms.AlohaOutputs(),
            ],
        ),
        action_horizon=model.action_horizon,
    )

    logging.info("Creating server...")
    server = http_policy_server.HttpPolicyServer(policy=policy, host="0.0.0.0", port=port)

    logging.info("Serving...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
