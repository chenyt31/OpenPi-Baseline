# docker build . -t libero -f examples/libero/Dockerfile
# docker run --rm -it --network=host -v .:/app libero /bin/bash
FROM python:3.8-slim@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Copy the requirements files so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./examples/libero/requirements.txt /tmp/requirements.txt
COPY ./third_party/libero/requirements.txt /tmp/requirements-libero.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN uv venv --python 3.8 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
ENV PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/main.py"]
