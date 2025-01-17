import dataclasses
from functools import partial
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax._src.tree_util as private_tree_util
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(weight_loader: _weight_loaders.WeightLoader, params: at.Params) -> at.Params:
    """Runs the weight loader and validates that the params structure, shapes, and dtypes are unchanged."""
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
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    weight_decay_mask = None
    freeze_mask = None
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask, freeze_mask)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def init(rng: at.KeyArrayLike) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters)
        model = config.model.create(model_rng)
        # extract a pure PyTree of params
        graphdef, state = nnx.split(model)
        params = state.to_pure_dict()
        # run the weight loader on this pure PyTree
        params = jax.lax.with_sharding_constraint(params, replicated_sharding)
        params = jax.pure_callback(partial(_load_weights_and_validate, config.weight_loader), params, params)
        params = jax.lax.with_sharding_constraint(params, replicated_sharding)
        # update the model with loaded params
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)

        # initialize the optimizer
        optimizer = nnx.Optimizer(model, tx)
        return training_utils.TrainState(
            step=0,
            params=nnx.state(model),
            model_def=nnx.graphdef(model),
            opt_state=nnx.state(optimizer),
            opt_def=nnx.graphdef(optimizer),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else nnx.state(model),
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    train_state = jax.jit(init, in_shardings=replicated_sharding, out_shardings=state_sharding)(init_rng)
    return train_state, state_sharding


@at.typecheck
def train_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    optimizer = nnx.merge(state.opt_def, state.opt_state)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    loss, grads = nnx.value_and_grad(loss_fn)(model, train_rng, observation, actions)
    optimizer.update(grads)  # updates the model and optimizer params in-place

    new_state = state.replace(step=state.step + 1, params=nnx.state(model), opt_state=nnx.state(optimizer))
    if state.ema_decay is not None:
        new_state = new_state.replace(
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, nnx.state(model)
            )
        )

    kernel_params = nnx.state(model, lambda path, _: path[-1] == "kernel")
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),  # TODO: do not compute norm for frozen params
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_threefry_partitionable", True)  # noqa: FBT003
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    if jax.device_count() % config.fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {config.fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // config.fsdp_devices, config.fsdp_devices)
    # In FSDP, the data is sharded accross both the batch and model axes.
    data_axis = (sharding.BATCH_AXIS, sharding.FSDP_AXIS)
    mesh = jax.make_mesh(mesh_shape, data_axis)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(data_axis))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_interval=config.keep_interval,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
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
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
