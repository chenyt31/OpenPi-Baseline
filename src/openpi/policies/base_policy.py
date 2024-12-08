import abc


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> object:
        """Infer actions from observations."""
