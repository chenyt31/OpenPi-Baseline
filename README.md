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

Each registered config is available as a command line argument to `scripts/train.py`. To find all available command line arguments for your config, run `uv run scripts/train.py <config-name> --help`, or look at the `TrainConfig` class in [src/openpi/training/config.py](src/openpi/training/config.py).


For example, to train with the `pi0_aloha_sim` config, run the following;

(one time only) Compute the norm stats for the training data:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim
```

Run training:

```bash
uv run scripts/train.py pi0_aloha_sim --exp-name=my_experiment --overwrite
```

The `pi0_aloha_sim` config is optimized for training on a single H100 GPU. By default, JAX pre-allocates 75% of available GPU memory. We set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use up to 90% of GPU memory, which enables training with larger batch sizes while maintaining stability.

The training script automatically utilizes all available GPUs on a single node. Currently, distributed training across multiple nodes is not supported.

An example for how to train on your own Aloha dataset is provided in the [ALOHA Real README](examples/aloha_real/README.md).
  
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
uv run scripts/serve_policy.py --env LIBERO --default_prompt "my task"
```

### Serve a trained policy from an openpi checkpoint

This option allows serving a model that was trained using the openpi training code.

```bash
uv run scripts/serve_policy.py --env ALOHA_SIM policy:checkpoint --policy.config=pi0_aloha_sim --policy.dir=checkpoints/pi0_aloha_sim/exp_name/10000
```

The training config is used to determine which data transformations should be applied to the runtime data before feeding into the model. The norm stats, which are used to normalize the transformed data, are loaded from the checkpoint directory.

### Serve an exported model

There are also a number of checkpoints that are available as exported JAX graphs, which we trained ourselves using our internal training code. These can be served using the following command:

```bash
uv run scripts/serve_policy.py --env ALOHA policy:exported --policy.dir=s3://openpi-assets/exported/pi0_aloha/model [--policy.processor=trossen_biarm_single_base_cam_24dim]
```

For these exported models, norm stats are loaded from processors that are exported along with the model, while data transformations are defined in the corresponding default policy (see `create_default_policy` in [scripts/serve_policy.py](scripts/serve_policy.py)). The processor name is optional, and if not provided, we will do the following:
- Try using the default environment processor name
- Load a processor if there is only one available
- Raise an error if there are multiple processors available and ask to provide a processor name

### Running with Docker:

```bash
export SERVER_ARGS="--env ALOHA_SIM --default_prompt 'my task'"
docker compose -f scripts/compose.yml up --build
```

## Fine-tuning to your own robot

This section provides a brief general guide to fine-tune $\pi_0$ to your own robot data. As a running example, we will describe adapting the policy to control a 6 DoF UR5e robot arm with a 1 DoF gripper. While $\pi_0$ may be adapted to a variety of different robot morphologies, we have tested single- and dual-arm setups with 6 DoF arms and single-arm setups with 7 DoF arms, typically with one base camera (over the shoulder) and a wrist camera on each wrist.

### Define the policy inputs and outputs

We will first define `UR5Inputs` and `UR5Outputs` classes derived from `DataTransformFn` to describe how the observations from the robot map onto model inputs and how the model outputs map onto robot actions, respectively. For a simple existing example such transforms, see [src/openpi/policies/droid_policy.py](src/openpi/policies/droid_policy.py).

We assume that the robot client produces a dict with the following fields, and that images have already been resized to the model's expected 224x224 resolution:
- `joints`: 6D vector of joint angles
- `gripper`: 1D gripper position in the range [0, 1]
- `base_rgb`: RGB image from the base camera at 224x224 resolution
- `wrist_rgb`: RGB image from the wrist camera at 224x224 resolution

Here is an example input transformation:

```
class UR5Inputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int, *, delta_action_mask: Sequence[bool] | None = None):
        self._action_dim = action_dim
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        # Pad to the expected input dimensionality of the model (same as action_dim).
        state = np.concatenate([data["joints"], data["gripper"]], axis=1)
        state = transforms.pad_to_dim(state, self._action_dim)

        inputs = {
            "state": state,
            # Dictionary containing image inputs, the keys must have these names.
            "image": {
                "base_0_rgb": data["base_rgb"],
                "left_wrist_0_rgb": data["wrist_rgb"]
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(data["base_rgb"]),
            },
            # 1D bool indicating if the image exists.
            "image_mask": {
                "base_0_rgb": np.ones(1, dtype=np.bool_),
                "left_wrist_0_rgb": np.ones(1, dtype=np.bool_),
                "right_wrist_0_rgb": np.zeros(1, dtype=np.bool_),
            },
        }

        # Pass the prompt along to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
```

Here is an example output transformation:

```
class UR5Outputs(transforms.DataTransformFn):
    def __init__(self, *, delta_action_mask: Sequence[bool] | None = None):
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        actions = np.asarray(data["actions"][..., :7])

        # _delta_action_mask indicates which dimensions of the action are relative to the
        # state. For these dimensions, add the state back in. Note we again use the first
        # 7 dimensions, because the robot expects 7D actions.
        if self._delta_action_mask is not None:
            state = np.asarray(data["state"][..., :7])
            mask = np.asarray(self._delta_action_mask[:7])
            actions = actions + np.expand_dims(np.where(mask, state, 0), axis=-2)

        return {"actions": actions}
```

You will also need to define a new entry in the `EnvMode` enum in [scripts/serve_policy.py](scripts/serve_policy.py). You do not need to do anything with this entry unless you want to use the `pi0_base` model in zero-shot, which is **almost certainly** the case if you are adding your own robot that $\pi_0$ was not trained on. But if you do want to use $\pi_0$ in zero-shot on a supported robot: You will need to add the right processor under `DEFAULT_EXPORTED` (if the model already supports your robot, this should exist, but you may need to file a Github issue to get the name), and add a case to `create_default_policy`, e.g.:

```
        case EnvMode.UR5:
            config = make_policy_config(
                input_layers=[
                    ur5_policy.UR5Inputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    ur5_policy.UR5Outputs(),
                ],
            )
```

### Define the training config

TODO: describe how to define the training config (presumably using the input/output transforms defined above)

### Convert your dataset and train

TODO: describe how to convert the dataset and train

### Serve your policy

TODO: explain how to do this
