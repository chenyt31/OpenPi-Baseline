import pathlib

import imageio
import numpy as np
from typing_extensions import override

from openpi.runtime import subscriber as _subscriber


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(self, out_path: pathlib.Path, subsample: int = 1) -> None:
        self._out_path = out_path
        self._images: list[np.ndarray] = []
        self._subsample = subsample

    @override
    def on_episode_start(self) -> None:
        self._images = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        im = observation["image"][0]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im)

    @override
    def on_episode_end(self) -> None:
        imageio.mimwrite(
            self._out_path,
            np.array(self._images[:: self._subsample]),
            duration=0.2,
            loop=0,
        )
