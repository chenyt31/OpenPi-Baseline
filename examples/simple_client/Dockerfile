# docker build . -t simple_client -f examples/simple_client/Dockerfile
# docker run --rm -it --network=host -v .:/app simple_client /bin/bash
FROM python:3.11-slim@sha256:370c586a6ffc8c619e6d652f81c094b34b14b8f2fb9251f092de23f16e299b78
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Copy the requirements files so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./examples/simple_client/requirements.txt /tmp/requirements.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt /tmp/openpi-client/pyproject.toml
ENV PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/simple_client/main.py"]
