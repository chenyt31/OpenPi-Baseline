# openpi

OpenPI holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, it is focused on the `pi0` model described in [this blog post](https://www.physicalintelligence.company/blog/pi0).

## Setup

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

### Using UV

We use [UV](https://docs.astral.sh/uv/) to manage Python dependencies. See the [UV installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up.

Once UV is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

### Docker Setup

All of the examples in this repo provide instructions for being run normally, and also using Docker. Although not required, the Docker option is recommended as this will simplify software installation, produce a more stable environment, and also allow you to avoid installing ROS and cluttering your machine, for examples which depend on ROS.

Docker installation instructions are [here](https://docs.docker.com/engine/install/). If using a GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). If your host machine is Ubuntu 22.04, you can use the convenience scripts `scripts/install_docker_ubuntu22.sh` and `scripts/install_nvidia_container_toolkit.sh`.

During the first run of any example, docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.

### Downloading checkpoints

Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). For linux, this means running the following commands:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Run the following from the cloned `openpi` directory:

```bash
# Set AWS credentials. These will no longer be needed after openpi graduates from beta.
export AWS_ACCESS_KEY_ID=AKIA4MTWIIQIZBO44C62
export AWS_SECRET_ACCESS_KEY=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

# Download the `pi0_sim` checkpoint.
export CHECKPOINT_NAME=pi0_sim
aws s3 sync s3://openpi-assets/checkpoints/$CHECKPOINT_NAME ./checkpoints/$CHECKPOINT_NAME
```

Available checkpoints:

- `pi0_sim`: TODO
- `pi0_droid`: TODO

## Running Examples

We provide example integrations with several robotics platforms. See the README in each example for more details:

- [ALOHA Sim](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [CALVIN](examples/calvin)
- [LIBERO](examples/libero)

## Running the OpenPI Server

The OpenPI server hosts model inference for an OpenPI policy. The examples describe how to run it in conjunction with each environment, but you can also run it standalone:

### With Docker:

```bash
export SERVER_ARGS="--mode SIM --default_prompt 'my task'"
docker compose -f scripts/compose.yml up --build
```

### Without Docker:

```bash
uv run scripts/serve_policy.py --mode SIM --default_prompt 'my task'
```
