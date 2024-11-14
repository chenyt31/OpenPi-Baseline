# docker build . -t pi0_server -f docker/pi0/Dockerfile
# docker run --rm -it --network=host -v .:/app --gpus=all pi0_server /bin/bash
FROM python:3.11-slim-bookworm@sha256:e8381c802593deb0c4d25bd3f4e05e94382f6bf33090de22679fc7488cd68bbb
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Copy the project config so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Sync the project into a new environment, using the frozen lockfile.
RUN uv sync --frozen --no-install-project --no-dev

CMD ["uv", "run", "src/openpi/serving/http_policy_server_pi0.py"]