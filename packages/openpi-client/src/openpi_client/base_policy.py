import abc


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        """Infer actions from observations."""
