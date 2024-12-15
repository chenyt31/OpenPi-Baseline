# openpi

See the design doc [here](https://docs.google.com/document/d/1ykjuV0GjuaYGzhppasmGlldJ5TOA-UCETBHKoxeoyqw/edit).

## Usage

When cloning this repo, make sure to update submodules:

```bash
git submodule update --init --recursive
```

```bash
uv run examples/hello.py
```

## Downloading checkpoints

Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). For linux, this means running the following commands:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Run the following from the cloned `openpi` directory:

```bash
AWS_ACCESS_KEY_ID=AKIA4MTWIIQIZBO44C62 AWS_SECRET_ACCESS_KEY=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 aws s3 sync s3://openpi-assets/checkpoints/pi0_base ./checkpoints/pi0_base
```

## Docker Setup

The recommended way to run many of our examples is using Docker. This will simplify software installation, produce a more stable environment, and also
allow you to avoid installing ROS and cluttering your machine, for examples which depend on ROS.

Docker installation instructions are [here](https://docs.docker.com/engine/install/). If using a GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). If your host machine is Ubuntu 22.04, you can use the convenience scripts `scripts/install_docker_ubuntu22.sh` and `scripts/install_nvidia_container_toolkit.sh`.

During the first run of any example, docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.

## Running Examples

We provide example integrations with several robotics platforms. See the README in each example for more details:

- [ALOHA Sim](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)

## Running the OpenPI Server

The OpenPI server hosts model inference for an OpenPI policy. The examples describe how to run it in conjunction with each environment, but you can also run it standalone:

### With Docker:

```bash
export SERVER_ARGS="--mode SIM --default_prompt 'my task'"
docker compose -f scripts/compose.yml up --build
```

### Without Docker:

```bash
uv run scripts/serve_policy.py
```
