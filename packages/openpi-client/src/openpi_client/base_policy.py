import abc
from typing import Dict  # noqa: UP035


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Infer actions from observations."""
