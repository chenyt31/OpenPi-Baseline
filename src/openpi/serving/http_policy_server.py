import http.server
import logging
import pickle
import threading
import traceback

from openpi.policies import policy as _policy


class HttpPolicyServer:
    """Hosts a generic policy over HTTP.

    See HttpClientPolicy for a corresponding client implementation.
    """

    def __init__(self, policy: _policy.Policy, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._server: http.server.HTTPServer | None = None
        self._started_event = threading.Event()

    def serve_forever(self) -> None:
        with http.server.HTTPServer((self._host, self._port), self) as http_server:
            self._server = http_server
            self._started_event.set()
            http_server.serve_forever()
        self._server = None

    def wait_for_start(self) -> None:
        self._started_event.wait(timeout=30)

    def shutdown(self) -> None:
        if not self._server:
            raise ValueError("Server is not running")
        self._server.shutdown()

    def __call__(self, *args, **kwargs):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(handler):  # type: ignore # noqa: N802,N805
                response = None
                try:
                    if handler.path == "/infer":
                        response = self._infer(handler)
                    else:
                        raise ValueError(f"Invalid path '{handler.path}'")
                    returncode = 200
                except Exception:
                    exc = traceback.format_exc()
                    logging.error(exc)
                    response = exc.encode("utf-8")
                    returncode = 500

                handler.send_response(returncode)
                handler.send_header("Content-type", "text/plain")
                handler.end_headers()
                handler.wfile.write(response)

        return Handler(*args, **kwargs)

    def _infer(self, handler: http.server.BaseHTTPRequestHandler) -> bytes:
        content_length = int(handler.headers["Content-Length"])
        data = handler.rfile.read(content_length)
        obs = pickle.loads(data)

        action = self._policy.infer(obs)

        return pickle.dumps(action)
