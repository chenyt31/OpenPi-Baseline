# docker build . -t pi0_server -f docker/pi0/Dockerfile
# docker run --rm -it --network=host -v ./checkpoints:/app/checkpoints --gpus=all pi0_server /bin/bash
FROM python:3.11-slim-bookworm@sha256:e8381c802593deb0c4d25bd3f4e05e94382f6bf33090de22679fc7488cd68bbb
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen

CMD ["uv", "run", "src/openpi/serving/http_policy_server_pi0.py"]