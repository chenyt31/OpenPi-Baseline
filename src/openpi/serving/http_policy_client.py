import pickle
import time

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
        t_start = time.time()
        print(f"Sending obs to {self._uri}")
        response = requests.post(f"{self._uri}/infer", data=pickle.dumps(obs))
        if response.status_code != 200:
            raise Exception(response.text)
        print(f"Time to get response: {time.time() - t_start}")

        t_start = time.time()
        content = pickle.loads(response.content)
        print(f"Time deserializing: {time.time() - t_start}")
        return content
