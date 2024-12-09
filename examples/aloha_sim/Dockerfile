# docker build . -t aloha_sim -f examples/aloha_sim/Dockerfile
# docker run --rm -it --network=host -v .:/app aloha_sim /bin/bash
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev
ENV MUJOCO_GL=egl

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Copy the requirements file so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./examples/aloha_sim/requirements.txt /tmp/requirements.txt

# Install python dependencies.
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt
ENV PYTHONPATH=/app:/app/src

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/aloha_sim/main.py"]