"""
Client policy to run remote inference in a Modal GPU container.

Deploy the Modal app which will serve the inference. This only need to be run once.
```
uv run --with quic-portal[modal]==0.1.6 modal deploy examples/simple_modal_client/modal_policy.py
```
"""

import time

import numpy as np
from openpi_client import base_policy as _base_policy, msgpack_numpy
from quic_portal import Portal

import modal

app = modal.App("openpi-server")

openpi_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands("git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git /root/openpi")
    .pip_install("uv")
    .run_commands(
        "cd /root/openpi && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT='/usr/local/' uv sync"
    )
    .run_commands("cd /root && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 uv pip install --system -e openpi")
    .run_commands("cd /root && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 uv pip install --system modal")
    .run_commands("cd /root && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 uv pip install --system quic-portal==0.1.6")
)

volume = modal.Volume.from_name("openpi-cache", create_if_missing=True)


@app.function(
    image=openpi_image,
    region="us-west-1",
    timeout=3600,
    gpu="h100",
    volumes={"/root/.cache/openpi": volume},
)
def run_server(rendezvous: modal.Dict, policy_name: str, policy_checkpoint: str):
    import os

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi_client import msgpack_numpy

    print(f"[server] in {os.getenv('MODAL_REGION')}")

    class ModalPolicyServer:
        def __init__(self, policy, portal: Portal) -> None:
            self._policy = policy
            self._portal = portal
            self._packer = msgpack_numpy.Packer()

            # Server expects hello from client.
            assert self._portal.recv() == b"hello"
            self._portal.send(self._packer.pack(policy.metadata))
            print("[server] Server metadata sent.")

        def serve_forever(self):
            while True:
                obs = msgpack_numpy.unpackb(self._portal.recv())
                action = self._policy.infer(obs)
                self._portal.send(self._packer.pack(action))

    t0 = time.time()
    print(f"[server] Creating policy {policy_name} from {policy_checkpoint} ...")
    policy = _policy_config.create_trained_policy(
        _config.get_config(policy_name), policy_checkpoint, default_prompt=None
    )
    print(f"[server] Policy created in {time.time() - t0:.2f}s.")

    t0 = time.time()
    print("[server] Initalizing server ...")
    portal = Portal.create_server(rendezvous)
    server = ModalPolicyServer(policy, portal)
    print(f"[server] Server initialized in {time.time() - t0:.2f}s.")

    print("[server] Starting to serve policy ...")
    server.serve_forever()


class ModalPolicy(_base_policy.BasePolicy):
    def __init__(self, policy_name: str, policy_checkpoint: str):
        try:
            fn = modal.Function.from_name("openpi-server", "run_server").hydrate()
        except modal.exception.NotFoundError:
            raise Exception(
                "Please first deploy the app with "
                "`uv run --with quic-portal[modal]==0.1.6 modal deploy examples/simple_modal_client/modal_policy.py`"
            ) from None

        self._packer = msgpack_numpy.Packer()
        for attempt in range(5):
            try:
                with modal.Dict.ephemeral() as rendezvous:
                    fn.spawn(rendezvous, policy_name, policy_checkpoint)
                    portal = Portal.create_client(rendezvous)

                portal.send(b"hello")
                self._server_metadata = msgpack_numpy.unpackb(portal.recv())
                self._portal = portal
                return
            except Exception as e:
                print(f"[client] Error creating portal: {e}")
                if attempt == 4:
                    raise e

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, obs: np.ndarray) -> np.ndarray:
        self._portal.send(self._packer.pack(obs))
        return msgpack_numpy.unpackb(self._portal.recv())
