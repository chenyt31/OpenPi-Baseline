from collections.abc import Iterator
import dataclasses
from functools import partial
import logging
from typing import Any

import etils.epath as epath
from flax.training import common_utils
import jax
import jax._src.tree_util as private_tree_util
import jax.experimental
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.common as _common
import openpi.models.model as _model
import openpi.policies.aloha_policy as aloha_policy
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool):
    exp_dir = epath.Path(config.checkpoint_dir) / config.exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist.")
    if resuming:
        run_id = (exp_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (exp_dir / "wandb_id.txt").write_text(wandb.run.id)

    # log all of the code in the repo
    wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(weight_loader: _weight_loaders.WeightLoader, params: at.Params) -> at.Params:
    new_params = weight_loader.load(jax.tree.map(lambda x: x, params))

    if errors := list(private_tree_util.equality_errors(params, new_params)):
        raise ValueError(
            "Weight loading changed the params structure:\n"
            + (
                "\n".join(
                    f"   - {jax.tree_util.keystr(path)} changed from {thing1} to {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    def check(kp, x, y):
        if (x := jax.ShapeDtypeStruct(x.shape, x.dtype)) != (y := jax.ShapeDtypeStruct(y.shape, y.dtype)):
            raise ValueError(
                f"Weight loading changed the params structure: expected {y}, got {x} at {jax.tree_util.keystr(kp)}"
            )

    jax.tree_util.tree_map_with_path(check, params, new_params)

    return new_params


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    model: _model.Model,
    init_rng: at.KeyArrayLike,
    batch: tuple[_common.Observation, _common.Actions],
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    weight_decay_mask = None
    freeze_mask = None
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask, freeze_mask)

    def init(rng: at.KeyArrayLike, data: tuple[_common.Observation, _common.Actions]) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        observation, actions = data
        params = model.init_params(model_rng, observation, actions)
        params = jax.experimental.io_callback(
            partial(_load_weights_and_validate, config.weight_loader), params, params, ordered=True
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            opt_state=tx.init(params),
            tx=tx,
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_parallel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("batch",)))

    train_state_shape = jax.eval_shape(init, init_rng, batch)
    # This is where we may want to shard the train state (e.g., FSDP).
    state_sharding = jax.tree.map(lambda _: replicated_sharding, train_state_shape)

    if resume:
        return train_state_shape, state_sharding

    train_state = jax.jit(
        init, in_shardings=(replicated_sharding, data_parallel_sharding), out_shardings=state_sharding
    )(init_rng, batch)
    return train_state, state_sharding


@at.typecheck
def train_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    model: _model.Model,
    batch: tuple[_common.Observation, _common.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    def loss_fn(params: at.Params, rng: at.KeyArrayLike, observation: _common.Observation, actions: _common.Actions):
        chunked_loss = model.compute_loss(rng, observation, actions, params=params, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    loss, grads = jax.value_and_grad(loss_fn)(state.params, train_rng, observation, actions)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = new_state.replace(
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            )
        )

    kernel_mask = training_utils.mask_from_regex(r".*\['kernel'\]", state.params)
    kernel_params = jax.tree.map(lambda p, m: p if m else None, state.params, kernel_mask)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),  # TODO: do not compute norm for frozen params
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


class LeRobotRepack(_transforms.DataTransformFn):
    def __call__(self, item) -> dict:
        img = item["observation.images.top"]
        return {
            "images": {"cam_high": img},
            "state": item["observation.state"],
            "actions": item["action"],
        }


def create_data_loader(
    config: _config.TrainConfig, model: _model.Model, sharding: jax.sharding.Sharding
) -> Iterator[tuple[_common.Observation, _common.Actions]]:
    repo_id = "lerobot/aloha_sim_transfer_cube_human"
    action_horizon = model.action_horizon

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id, delta_timestamps={"action": [t / dataset_meta.fps for t in range(action_horizon)]}
    )

    for batch in _data_loader.data_loader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        transforms=[
            LeRobotRepack(),
            aloha_policy.AlohaInputs(action_dim=model.action_dim),
            _transforms.ResizeImages(224, 224),
        ],
    ):
        # Perform this after the batch has been sharded.
        yield _common.Observation.from_dict(batch), batch["actions"]


def create_fake_data_loader(
    config: _config.TrainConfig, model: _model.Model, sharding: jax.sharding.Sharding
) -> Iterator[tuple[_common.Observation, _common.Actions]]:
    dataset = _data_loader.FakeDataset(model, config.num_train_steps)

    return _data_loader.data_loader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
    )


def main(config: _config.TrainConfig):
    init_logging()

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_threefry_partitionable", True)  # noqa: FBT003
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # data parallel only
    # TODO: replace with jax.make_mesh when available
    mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
    data_parallel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("batch",)))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint(
        epath.Path(config.checkpoint_dir) / config.exp_name,
        keep_interval=config.keep_interval,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming)

    model = _model.Model(module=config.module, action_dim=24, action_horizon=50, max_token_len=48)

    data_loader = create_data_loader(config, model, data_parallel_sharding)
    batch = next(data_loader)
    logging.info(f"Data loader initialized: {training_utils.to_tree_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, model, init_rng, batch, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.to_tree_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)

    ptrain_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, train_state_sharding, None, data_parallel_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        train_state, info = ptrain_step(train_rng, train_state, model, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_loader)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, step)

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
