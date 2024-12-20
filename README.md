# openpi

OpenPI holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, it is focused on the `pi0` model described in [this blog post](https://www.physicalintelligence.company/blog/pi0).

## Setup

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

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

By default checkpoint are cached in `~/.cache/openpi` when running fine-tuning or eval, you can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.

Available checkpoints:

- `pi0_sim`: TODO
- `pi0_droid`: TODO


## Running Training

Below example shows how to run training with a config defined in `openpi/training/config.py`. Note that JAX by default pre-allocates 75% GPU memory, in practice we have found allocating 90% of GPU memory with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is a good default for training pi0.

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config name (eg. pi0 / pi0_small / debug)> 
```

## Running Examples

We provide example integrations with several robotics platforms. See the README in each example for more details:

- [ALOHA Sim](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [CALVIN](examples/calvin)
- [LIBERO](examples/libero)

## Running the OpenPI Server

The OpenPI server hosts model inference for an OpenPI policy. 

The server can be configured using the folllowing commands line arguments:

- `--env`: The environment to serve the policy for.
- `--config-name`: If provided, loads the policy from a training config. Otherwise, loads the default pi0 policy.
- `--checkpoint-path`: Required if `config-name` is provided. Specifies the path to the checkpoint to load.
- `--default-prompt`: If provided, overrides the default prompt for the policy.

The examples describe how to run it in conjunction with each environment, but you can also run it standalone:

### With Docker:

```bash
export SERVER_ARGS="--env ALOHA_SIM --default_prompt 'my task'"
docker compose -f scripts/compose.yml up --build
```

### Without Docker:

```bash
uv run scripts/serve_policy.py --env ALOHA_SIM --default_prompt 'my task'
```
