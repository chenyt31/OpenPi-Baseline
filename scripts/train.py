import logging

from flax.training import common_utils
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import optax
import tqdm

import openpi.models.common as _common
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils


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

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    model: _model.Model,
    init_rng: at.KeyArrayLike,
    batch: tuple[_common.Observation, _common.Actions],
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, jax.sharding.NamedSharding]:
    def init(rng: at.KeyArrayLike, data: tuple[_common.Observation, _common.Actions]) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        observation, actions = data
        params = model.init_params(model_rng, observation, actions)
        weight_decay_mask = None
        freeze_mask = None

        tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask, freeze_mask)
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
    state_sharding = replicated_sharding
    state = jax.eval_shape(init, init_rng, batch)
    if resume:
        return state, state_sharding
    with at.disable_typechecking():  # TODO: https://github.com/patrick-kidger/jaxtyping/issues/277
        state = jax.jit(
            init,
            in_shardings=(replicated_sharding, data_parallel_sharding),
            out_shardings=state_sharding,
        )(init_rng, batch)
    # TODO: Improve this to not have two copies of the weight in memory.
    if config.load_pretrained_weights:
        state = _checkpoints.restore_weights(config.load_pretrained_weights, state, state_sharding)
    return state, state_sharding


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
    return state, info


def main(config: _config.TrainConfig):
    init_logging()
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    jax.config.update("jax_threefry_partitionable", True)  # noqa: FBT003

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # data parallel only
    mesh_shape = (jax.device_count(),)
    # TODO: replace with jax.make_mesh when available
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh(mesh_shape), ("batch",))
    data_parallel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("batch",)))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint(
        config.checkpoint_dir, keep_interval=config.keep_interval, overwrite=config.overwrite, resume=config.resume
    )
    multihost_utils.sync_global_devices("init_checkpoint")

    model = _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)
    data_loader = _data_loader.fake_init_data_loader(
        model, local_batch_size=config.batch_size // jax.process_count(), dp_sharding=data_parallel_sharding
    )
    batch = next(data_loader)
    logging.info(f"Data loader initialized: {training_utils.to_tree_info(batch)}")
    multihost_utils.sync_global_devices("init_dataloader")

    train_state, train_state_sharding = init_train_state(config, model, init_rng, batch, mesh, resume=resuming)
    logging.info(f"Initialized train state:\n{training_utils.to_tree_info(train_state.params)}")
    multihost_utils.sync_global_devices("init_train_state")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)
        multihost_utils.sync_global_devices("resume_training")

    ptrain_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, train_state_sharding, None, data_parallel_sharding),
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
        with at.disable_typechecking():  # TODO: https://github.com/patrick-kidger/jaxtyping/issues/277
            train_state, info = ptrain_step(train_rng, train_state, model, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            logging.info(f"Step {step}: {reduced_info}")
            infos = []
        batch = next(data_loader)

        if step % config.save_interval == 0 and step > start_step:
            _checkpoints.save_state(checkpoint_manager, train_state, step)


if __name__ == "__main__":
    main(_config.cli())
