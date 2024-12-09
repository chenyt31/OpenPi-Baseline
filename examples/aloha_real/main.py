import dataclasses
import logging

import tyro

from examples.aloha_real import env as _env

# from examples.aloha_real import video_display as _video_display
from openpi.runtime import runtime as _runtime
from openpi.runtime.agents import policy_agent as _policy_agent
from openpi.serving import http_policy_client as _http_policy_client


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000


def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(),
        agent=_policy_agent.PolicyAgent(
            policy=_http_policy_client.HttpClientPolicy(
                host=args.host,
                port=args.port,
            )
        ),
        subscribers=[
            # _video_display.VideoDisplay(),
        ],
        max_hz=50,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
