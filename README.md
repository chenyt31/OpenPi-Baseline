# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, it is focused on the `pi0` model described in [this blog post](https://www.physicalintelligence.company/blog/pi0).

## Setup

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

### Using uv

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up.

Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

### Docker Setup

All of the examples in this repo provide instructions for being run normally, and also using Docker. Although not required, the Docker option is recommended as this will simplify software installation, produce a more stable environment, and also allow you to avoid installing ROS and cluttering your machine, for examples which depend on ROS.

Docker installation instructions are [here](https://docs.docker.com/engine/install/). If using a GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). If your host machine is Ubuntu 22.04, you can use the convenience scripts `scripts/install_docker_ubuntu22.sh` and `scripts/install_nvidia_container_toolkit.sh`.

During the first run of any example, Docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.

### Downloading checkpoints

By default checkpoints are downloaded from `s3://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.

## Running Training

Training configs are defined in [src/openpi/training/config.py](src/openpi/training/config.py) and the training script is in [scripts/train.py](scripts/train.py).

Each registered config is available as a command line argument to `scripts/train.py`. For example, to train with the `pi0` config, run:

```bash
uv run scripts/train.py pi0 --exp-name=my_experiment --overwrite
```

To find the available command line options for your config, run `uv run scripts/train.py <config-name> --help`, or look at the `TrainConfig` class in [src/openpi/training/config.py](src/openpi/training/config.py).

TIP: JAX pre-allocates 75% of GPU memory by default. However, in practice we have found that allocating 90% of GPU memory with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is a good default for training pi0 models.

## Running examples

We provide example integrations with several robotics platforms. See the README in each example for more details:

- [ALOHA Sim](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [CALVIN](examples/calvin)
- [LIBERO](examples/libero)

## Running the openpi server

The server can be configured to serve openpi policies in the following ways:

- Serve a default policy for the given environment.
- Serve a trained policy from a checkpoint.
- Serve an exported model.

### Serve the default policy for the LIBERO environment

```bash
uv run scripts/serve_policy.py --env LIBERO
```

### Serve a trained policy from an openpi checkpoint

```bash
uv run scripts/serve_policy.py --env ALOHA_SIM policy:checkpoint --policy.config=pi0_pretrained --policy.dir=checkpoints/pi0_pretrained/exp_name/10000
```

The training config us used to determine which data transformations should be applied to the runtime data before feeding into the model. The norm stats, which are used to normalize the transformed data, are loaded from the checkpoint directory.

### Serve an exported model

```bash
uv run scripts/serve_policy.py --env ALOHA policy:exported --policy.dir=s3://openpi-assets/exported/pi0_aloha/model --policy.processor=trossen_biarm_single_base_cam_24dim
```

In this case, the data transformations are taken from the default policy. However, the processor name will be used to determine which norms stats should be used to normalize the transformed data.


### Running with Docker:

```bash
export SERVER_ARGS="--env ALOHA_SIM --default_prompt 'my task'"
docker compose -f scripts/compose.yml up --build
```
