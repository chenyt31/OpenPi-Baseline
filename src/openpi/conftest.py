import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
