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
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        print(f"Waiting for server at {self._uri}...")
        while True:
            try:
                return requests.head(self._uri)
            except requests.exceptions.ConnectionError:
                print("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: dict) -> dict:
        response = requests.post(f"{self._uri}/infer", data=pickle.dumps(obs))
        if response.status_code != 200:
            raise Exception(response.text)

        return pickle.loads(response.content)
