import matplotlib.pyplot as plt
from typing_extensions import override

from openpi.runtime import subscriber as _subscriber


class VideoDisplay(_subscriber.Subscriber):
    """Displays video frames."""

    def __init__(self) -> None:
        self._ax: plt.Axes | None = None
        self._plt_img: plt.Image | None = None

    @override
    def on_episode_start(self) -> None:
        plt.ion()
        self._ax = plt.subplot()
        self._plt_img = None

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        assert self._ax is not None

        image = observation["image"][0, ...]

        if self._plt_img is None:
            self._plt_img = self._ax.imshow(image)
        else:
            self._plt_img.set_data(image)
        plt.pause(0.001)

    @override
    def on_episode_end(self) -> None:
        plt.ioff()
        plt.close()
