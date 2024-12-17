# docker build . -t openpi_server -f scripts/serve_policy.Dockerfile
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
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
COPY ./pyproject.toml /tmp/pyproject.toml
# This is a bit of a hack because installing this the uv way would require copying
# the openpi-client code into the build, which is not desirable.
RUN sed -i '/openpi-client/d' /tmp/pyproject.toml
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml


# Install python dependencies.
RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install --system
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip compile /tmp/pyproject.toml -o /tmp/requirements.txt
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip sync /tmp/requirements.txt /tmp/openpi-client/pyproject.toml

ENV PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src

CMD /bin/bash -c "source /.venv/bin/activate && python scripts/serve_policy.py $SERVER_ARGS"
