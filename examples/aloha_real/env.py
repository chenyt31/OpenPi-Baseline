import einops
import numpy as np
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import real_env as _real_env


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(self, render_height: int = 480, render_width: int = 640) -> None:
        self._env = _real_env.make_real_env(init_node=True)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def done(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            obs["images"][cam_name] = einops.rearrange(obs["images"][cam_name], "h w c -> c h w").astype(np.uint8)

        return {
            "state": obs["qpos"],
            "images": obs["images"],
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
