import dataclasses
import logging
import pathlib

import tyro

from openpi.runtime import runtime as _runtime
from openpi.runtime.agents import policy_agent as _policy_agent
from openpi.runtime.environments.aloha_sim import env as _env
from openpi.runtime.environments.aloha_sim import saver as _saver
from openpi.serving import http_policy_client as _http_policy_client


@dataclasses.dataclass
class Args:
    out_path: pathlib.Path = pathlib.Path("out.mp4")

    task: str = "gym_aloha/AlohaInsertion-v0"
    seed: int = 0

    host: str = "0.0.0.0"
    port: int = 8000


def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=_http_policy_client.HttpClientPolicy(
                host=args.host,
                port=args.port,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_path),
        ],
        max_hz=50,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
