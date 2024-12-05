from typing_extensions import override

from openpi.policies import policy as _policy
from openpi.runtime import agent as _agent


# TODO: Consider unifying policies and agents.
class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    def __init__(self, policy: _policy.Policy) -> None:
        self._policy = policy

    @override
    def get_action(self, observation: dict) -> dict:
        return self._policy.infer(observation)
