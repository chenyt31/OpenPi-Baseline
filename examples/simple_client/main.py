import dataclasses
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    example: str = "aloha"


def main(args: Args) -> None:
    obs_fn = {
        "aloha": _random_observation_aloha,
    }[args.example]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # Send 1 observation to make sure the model is loaded.
    policy.infer(obs_fn())

    start = time.time()
    for _ in range(100):
        policy.infer(obs_fn())
    end = time.time()

    print(f"Total time taken: {end - start}")
    # Note that each inference returns many action chunks.
    print(f"Inference rate: {100 / (end - start)} Hz")


def _random_observation_aloha() -> dict:
    return {
        "qpos": np.ones((14,)),
        "image": np.random.rand(4, 3, 480, 640).astype(np.float32),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
