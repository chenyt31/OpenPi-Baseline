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

### Pre-trained checkpoints

We currently have the following model weights available for fine-tuning:

name | path | description
--- | --- | ---
`pi0_base` | `s3://openpi-assets/checkpoints/pi0_base/params` | Standard pre-trained $\pi_0$ model for general fine-tuning

The path should be fed as an argument into `CheckpointWeightLoader` in your training config (see [config.py](src/openpi/training/config.py) for examples).

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


### Available exported models

We currently have the following exported models available for use. See [scripts/serve_policy.py](scripts/serve_policy.py) for details.

name | path | env | recommended language command | description
--- | --- | ---  | --- | ---
`pi0_base` | `s3://openpi-assets/exported/pi0_base/model/` | `ALOHA`, `DROID` | `"be a good robot"` | Standard pre-trained $\pi_0$, may not perform well in zero-shot
`pi0_aloha` | `s3://openpi-assets/exported/pi0_aloha/model` | `ALOHA` | `""` | $\pi_0$ model fine-tuned on public ALOHA data, supports pen cap/uncap task
`pi0_aloha_towel` | `s3://openpi-assets/exported/pi0_aloha_towel/model` | `ALOHA` | `"fold the towel"` | $\pi_0$ model fine-tuned to perform a towel folding task on ALOHA
`pi0_aloha_sim` | `s3://openpi-assets/exported/pi0_aloha_sim/model` | `ALOHA_SIM` | `"be a good robot"` | $\pi_0$ model fine-tuned on public simulated ALOHA cube transfer task
`pi0_droid` | `s3://openpi-assets/exported/pi0_droid/model` | `DROID` | any DROID command | $\pi_0$ model fine-tuned on public DROID dataset
`pi0_calvin` | `s3://openpi-assets/exported/pi0_calvin/model` | `CALVIN` | any CALVIN command | $\pi_0$ model fine-tuned on public CALVIN simulated dataset
`pi0_libero` | `s3://openpi-assets/exported/pi0_libero/model` | `LIBERO` | any LIBERO command | $\pi_0$ model fine-tuned on public LIBERO simulated dataset

### Running with Docker:

```bash
export SERVER_ARGS="--env ALOHA_SIM --default_prompt 'my task'"
docker compose -f scripts/compose.yml up --build
```

## Fine-tuning to your own robot

This section provides a brief general guide to fine-tune $\pi_0$ to your own robot data. As a running example, we will describe adapting the policy to control a 6 DoF UR5e robot arm with a 1 DoF gripper. While $\pi_0$ may be adapted to a variety of different robot morphologies, we have tested single- and dual-arm setups with 6 DoF arms and single-arm setups with 7 DoF arms, typically with one base camera (over the shoulder) and a wrist camera on each wrist.

### Define the policy inputs and outputs

We will first define `UR5Inputs` and `UR5Outputs` classes derived from `DataTransformFn` to describe how the observations from the robot map onto model inputs and how the model outputs map onto robot actions, respectively. For a simple existing example such transforms, see [src/openpi/policies/droid_policy.py](src/openpi/policies/droid_policy.py).

We assume that the robot client produces a dict with the following fields, and that images are `uint8` RGB in the range [0, 255]. The model expects 224x224 resolution images, but we can add a transformation to our processor stack that will resize them if necessary (more on this later):
- `joints`: 6D vector of joint angles
- `gripper`: 1D gripper position in the range [0, 1]
- `base_rgb`: RGB image from the base camera
- `wrist_rgb`: RGB image from the wrist camera

Here is an example input transformation:

```python
@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    # This is the action dimensionality of the model (not the robot!). The model has the same
    # state and action dimensionality, so this variable can be used to determine how to pad
    # state and action inputs loaded from the dataset.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        # Pad to the expected input dimensionality of the model (same as action_dim).
        state = np.concatenate([data["joints"], data["gripper"]])
        state = transforms.pad_to_dim(state, self.action_dim)

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
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.False_,
            },
        }

        # If this is called during training, actions will also be available.
        if "actions" in data:
            # The robot produces 7D actions (6 DoF + 1 gripper), and we pad these.
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        # Pass the prompt along to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
```

Here is an example output transformation:

```python
@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}
```

You will also need to define a new entry in the `EnvMode` enum in [scripts/serve_policy.py](scripts/serve_policy.py) (e.g., `UR5`). You do not need to do anything with this entry unless you want to use the `pi0_base` model in zero-shot, which is **almost certainly** the case if you are adding your own robot that $\pi_0$ was not trained on. But if you do want to use $\pi_0$ in zero-shot on a supported robot: You will need to add the right processor under `DEFAULT_EXPORTED` (if the model already supports your robot, this should exist, but you may need to file a Github issue to get the name), and add a case to `create_default_policy`, e.g.:

```python
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

To create a configuration for training on your robot, it will be necessary to add two things to [src/openpi/training/config.py](src/openpi/training/config.py): a `DataConfigFactory` subclass that defines the data source for your robot, and a `TrainConfig` object added to the `_CONFIGS` list, which lists the available training configs. Let's start with the `DataConfigFactory`: [config.py](src/openpi/training/config.py) already has one example (`LeRobotAlohaDataConfig`), which uses the LeRobot dataset format to import data for the ALOHA platform. This example is a bit more involved than we will need, since ALOHA has a few quirks that require special handling. Here is an example of `DataConfigFactory` subclass for our UR5 example above:

```python
@dataclasses.dataclass(frozen=True)
class UR5DataConfig(DataConfigFactory):
    # This is the repo id from LeRobot for this dataset repo.
    repo_id: str
    # If true, will convert joint dimensions to deltas with respect to the current state
    # before passing to the model. Most models expect this. Grippers do not use delta.
    use_delta_joint_actions: bool = False
    # By default, the prompt will be "be a good robot", but we can override it here.
    # We can also set the prompt in the command line, but it's good to specify a reasonable
    # default that matches a reasonable prompt in the training set.
    default_prompt: str | None = None
    # If true, will disable syncing the dataset from the huggingface hub.
    local_files_only: bool = False

    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        # This is standard boilerplate for loading norm stats for openpi checkpoints.
        norm_stats_path = metadata_dir / self.repo_id / "norm_stats.json"
        if norm_stats_path.exist():
            norm_stats = _normalize.deserialize_json(norm_stats_path.read_text())
        else:
            norm_stats = None

        # These transforms are the ones we wrote earlier.
        data_transforms=_transforms.Group(
            inputs=[ur5_policy.UR5Inputs(action_dim=model.action_dim)],
            outputs=[ur5_policy.UR5Outputs()],
        )

        if self.use_delta_joint_actions:
            # This mask defines which action/state dimensions correspond to grippers.
            # Positive arguments to make_bool_mask create that number of True values,
            # negative arguments create that number of False values, so this call will
            # create a bool array with 6xTrue followed by 1xFalse.
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        
        return DataConfig(
            repo_id=self.repo_id,
            norm_stats=norm_stats,
            data_transforms=data_transforms,
            # These transformations resize the images and tokenize the prompt.
            model_transforms=_transforms.Group(
                inputs=[
                    # This is only necessary if we expect the input images to require
                    # resizing. If your data is already 224x224 and your robot produces
                    # 224x224 images, then you can omit this, but it's also fine to leave
                    # this in as a precaution -- if the images are already 224x224, this
                    # adds no additional cost.
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model.max_token_len),
                        default_prompt=self.default_prompt,
                    ),
                ]
            ),
            local_files_only=self.local_files_only,
        )
```

Then, the `TrainConfig` that we add to the list `_CONFIGS` can look like this. You can further modify this if you want to tweak any other parameters, such as learning rate, batch size, and number of training steps (see the `TrainConfig` class for the full list of parameters):

```python
TrainConfig(
    name="ur5",
    data=UR5DataConfig(
        # This points to the repo ID for your dataset.
        repo_id="lerobot/my_ur5_dataset",
        # We set this to True because pi0_base expects delta actions.
        use_delta_joint_actions=True,
        # Set a reasonable default here.
        default_prompt="be a good robot",
        # Set this to True if you are using a dataset that is not on the HuggingFace hub.
        local_files_only=True,
    ),
    # This is the standard pi0 pretrained checkpoint.
    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
),
```

### Write your robot client

The openpi policy server serves the policy, and your robot can run a client to connect to this server, sending it observations and getting back actions. You can write your own robot client in whatever way you want, but if you would like to build on our client, we recommend installing the accompanying `openpi_client` package:

```bash
pip install -e packages/openpi-client
```

 Take a look at [examples/aloha_real/main.py](examples/aloha_real/main.py) for an example. The important bit of code is this:

```python
def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        # This specifies the environment (which you will implement to interface with your robot).
        environment=_env.AlohaRealEnvironment(),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                # This is the number of steps before getting a new action chunk.
                # Here it is taken from the CLI arguments, but you can also hard-code it.
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        # The expected real-time frequency of the robot.
        max_hz=50,
    )

    runtime.run()
```

If we look at the [Environment](packages/openpi-client/src/openpi_client/runtime/environment.py) class, we see that we need to implement just four methods (see [examples/aloha_real/env.py](examples/aloha_real/env.py)) for how these are implemented for the ALOHA robot:

- `def reset(self)`: This is called at the start of each episodes and might, for example, move the robot to a starting home position.
- `def done(self) -> bool`: This method returns `True` if the robot is done with the task, times out, etc. It is reasonable to always return `False` if you want the episode to run forever.
- `def get_observation(self) -> dict`: This method returns a dictionary of observations. If you followed this guide, your `UR5Inputs` class will receive exactly this dictionary as input. But it is possible to add other input transformations as well (which our ALOHA example does, see [aloha_policy.py](src/openpi/policies/aloha_policy.py)).
- `def apply_action(self, action: dict)`: This method sets the actions on the robot as specified by the dictionary `action`, which also matches the output of `UR5Outputs` unless other transformations are applied.

You should interface each of these functions with your own robot code as needed.

Congratulations, you are now done with all the additional openpi code!

### Convert your dataset and train

As described in the [Running Training](#running-training) section, we need to run two commands to train on this dataset. First, we need to compute normalization statistics for our training config. This needs to only be done once for the dataset (and redone if the dataset changes):

```bash
uv run scripts/compute_norm_stats.py --config-name ur5
```

Then we can run training:

```bash
uv run scripts/train.py ur5 --exp-name=my_experiment
```

You can use the `--overwrite` flag if you previously ran an experiment with the same name and would like to overwrite it, or `--resume` if you want to instead resume it from where you left off. Remember to set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use up to 90% of GPU memory. We optimized training to use H100 GPUs, though it will likely work on other high-memory GPUs. We use wandb to track training, and the script will prompt you to link to your wandb account and project the first time you start it. You can also disable it in the `TrainConfig` definition.


### Serve your policy

Once your model is trained, you can start the model server by running

```bash
uv run scripts/serve_policy.py --env UR5 policy:checkpoint --policy.config=ur5 --policy.dir=checkpoints/ur5/my_experiment/29999
```

where we use `29999` as the step number if we want the final checkpoint for a 30,000 step training run.

To start the robot client, if you used our client code, just run

```bash
python your_client_main.py
```

It is also possible to run the client with Docker, as described earlier in this document.
