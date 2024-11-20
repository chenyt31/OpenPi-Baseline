from functools import partial
import logging
import tqdm

from flax.training import common_utils
from flax.training import train_state
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import optax
import tyro

import openpi.base.array_typing as at
import openpi.models.common as _common
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.training.utils as training_utils
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config


def init_logging() -> None:
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
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def init_model(
    config: _config.TrainConfig,
    model: _model.Model,
    init_rng: at.KeyArrayLike,
    data: tuple[_common.Observation, _common.Actions],
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[train_state.TrainState, jax.sharding.NamedSharding]:
    def init_train_state(
        rng: at.KeyArrayLike, data: tuple[_common.Observation, _common.Actions], model: _model.Model
    ) -> train_state.TrainState:
        rng, model_rng = jax.random.split(rng)
        observation, actions = data
        model = model.init_params(model_rng, observation, actions)
        weight_decay_mask = None
        freeze_mask = None

        lr_schedule = (
            _optimizer.schedule(config.lr_schedule) if isinstance(config.lr_schedule, str) else config.lr_schedule()
        )
        tx = (
            _optimizer.optimizer(
                config.optimizer, lr_schedule, weight_decay_mask=weight_decay_mask, freeze_mask=freeze_mask
            )
            if isinstance(config.optimizer, str)
            else config.optimizer(lr_schedule, weight_decay_mask=weight_decay_mask, freeze_mask=freeze_mask)
        )

        def loss_fn(
            params: at.Params, rng: at.KeyArrayLike, observation: _common.Observation, actions: _common.Actions
        ):
            chunked_loss = model.compute_loss(rng, observation, actions, params={"params": params}, train=True)
            return jnp.mean(chunked_loss)

        return train_state.TrainState.create(
            apply_fn=loss_fn,
            params=model.params["params"],
            tx=tx,
        )

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_parallel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("batch",)))
    state_sharding = replicated_sharding
    state = jax.eval_shape(init_train_state, init_rng, data, model)
    if resume:
        return state, state_sharding
    # jax typechecking doesn't like sharding and jit
    with at.disable_typechecking():
        state = jax.jit(
            partial(init_train_state, model=model),
            in_shardings=(replicated_sharding, data_parallel_sharding),
            out_shardings=state_sharding,
        )(init_rng, data)
    # TODO: Improve this to not have two copies of the weight in memory.
    if config.load_pretrained_weights:
        state = _checkpoints.restore_weights(config.load_pretrained_weights, state, state_sharding)
    return state, state_sharding


def train_step(
    rng: at.KeyArrayLike, state: train_state.TrainState, batch: tuple[_common.Observation, _common.Actions]
) -> train_state.TrainState:
    old_params = state.params
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    loss, grads = jax.value_and_grad(state.apply_fn)(state.params, train_rng, observation, actions)
    state = state.apply_gradients(grads=grads)

    used_grads = jax.tree.map(lambda g, old, new: g if old is not new else None, grads, old_params, state.params)
    kernel_mask = training_utils.mask_from_regex(r".*\['kernel'\]", state.params)
    kernel_params = jax.tree.map(lambda p, m: p if m else None, state.params, kernel_mask)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(used_grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return state, info


def main(*, config: _config.TrainConfig, overwrite: bool = False, resume: bool = False):
    init_logging()
    jax.distributed.initialize()
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    jax.config.update("jax_threefry_partitionable", True)

    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)

    # data parallel only
    mesh_shape = (jax.device_count(),)
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh(mesh_shape), ("batch",))
    data_parallel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("batch",)))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint(
        config.checkpoint_dir, keep_interval=config.keep_interval, overwrite=overwrite, resume=resume
    )
    multihost_utils.sync_global_devices("init_checkpoint")

    model = _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)
    data_loader = _data_loader.fake_init_data_loader(
        model, local_batch_size=config.batch_size // jax.process_count(), dp_sharding=data_parallel_sharding
    )
    batch = next(data_loader)
    logging.info(f"Data loader initialized: {training_utils.to_tree_info(batch)}")
    multihost_utils.sync_global_devices("init_dataloader")

    train_state, train_state_sharding = init_model(config, model, init_rng, batch, mesh, resume=resuming)
    logging.info(f"Initialized train state: {training_utils.to_tree_info(train_state.params)}")
    multihost_utils.sync_global_devices("init_train_state")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)
        multihost_utils.sync_global_devices("resume_training")

    # jax typechecking doesn't like sharding and jit
    ptrain_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_parallel_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.trange(
        start_step,
        config.num_train_steps,
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=jax.process_index() != 0,
    )

    infos = []
    for step in pbar:
        with at.disable_typechecking():
            train_state, info = ptrain_step(rng, train_state, batch)
            infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree_util.tree_map(jnp.mean, stacked_infos))
            logging.info(f"Step {step}: {reduced_info}")
            infos = []
        with at.disable_typechecking():
            batch = next(data_loader)

        if step % config.save_interval == 0:
            _checkpoints.save_state(checkpoint_manager, train_state, step)


if __name__ == "__main__":
    tyro.cli(main)
