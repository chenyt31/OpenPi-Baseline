import pickle

import requests
from typing_extensions import override

from openpi.policies import policy as _policy


class HttpClientPolicy(_policy.Policy):
    """Implements the Policy interface by communicating with a server over HTTP.

    See HttpPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"http://{host}:{port}"

    @override
    def infer(self, obs: dict) -> dict:
        response = requests.post(f"{self._uri}/infer", data=pickle.dumps(obs))
        if response.status_code != 200:
            raise Exception(response.text)

        return pickle.loads(response.content)
