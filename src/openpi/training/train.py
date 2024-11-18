import dataclasses
from functools import partial
import logging
import time

import clu.metric_writers as metric_writers
import clu.periodic_actions as periodic_actions
from etils import epath
from flax.training import common_utils
from flax.training import train_state
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

import openpi.base.array_typing as at
import openpi.models.common as _common
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.training.utils as training_utils


@dataclasses.dataclass
class Config:
    keep_interval: int = 5000
    lr_schedule: training_utils.WarmupCosineDecay = dataclasses.field(
        default_factory=lambda: training_utils.WarmupCosineDecay(
            warmup_steps=1000, decay_steps=int(1e9), peak_lr=5e-5, decay_lr=5e-5
        )
    )
    optimizer: training_utils.AdamW = dataclasses.field(default_factory=training_utils.AdamW)
    weight_decay_keys_regex: str | None = None
    freeze_keys_regex: str | None = None
    load_pretrained_weights: str | None = None
    num_train_steps: int = 2000_000
    log_interval: int = 100
    save_interval: int = 1000


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


def init_data_loader(config: Config, model: _model.Model, mini_batch_size: int, dp_sharding: jax.sharding.Sharding):
    observation_spec, action_spec = _model.create_inputs_spec(model, batch_size=mini_batch_size)

    def make_from_spec(spec: jax.ShapeDtypeStruct, rng: at.KeyArrayLike):
        if spec.dtype == jnp.float32:
            return jax.random.uniform(rng, shape=spec.shape, minval=-1.0, maxval=1.0)
        if spec.dtype == jnp.int32:
            return jax.random.randint(rng, shape=spec.shape, minval=0, maxval=2048)
        return jnp.zeros(shape=spec.shape, dtype=spec.dtype)

    def _to_dp_array(stacked_arr: jax.Array):
        global_shape = (stacked_arr.shape[1] * len(dp_sharding.device_set), *stacked_arr.shape[2:])
        arrs = [
            jax.device_put(stacked_arr[device_idx, ...], device)
            for device_idx, device in zip(range(stacked_arr.shape[0]), dp_sharding.addressable_devices, strict=True)
        ]
        return jax.make_array_from_single_device_arrays(global_shape, dp_sharding, arrs)

    def data_loader():
        rng = jax.random.key(0)
        for _ in range(config.num_train_steps):
            rng, data_rng = jax.random.split(rng)
            observations = []
            actions = []
            for _ in range(len(dp_sharding.addressable_devices)):
                observations.append(jax.tree.map(partial(make_from_spec, rng=data_rng), observation_spec))
                actions.append(jax.tree.map(partial(make_from_spec, rng=data_rng), action_spec))
            stacked_observations = common_utils.stack_forest(observations)
            stacked_actions = common_utils.stack_forest(actions)
            # shard to mini batch, stack forest and then map & to dp array
            yield jax.tree.map(_to_dp_array, stacked_observations), jax.tree.map(_to_dp_array, stacked_actions)

    return data_loader()


def initialize_checkpoint(
    config: Config, checkpoint_dir: str, *, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir)
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            if jax.process_index() == 0:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(f"Checkpoint directory {checkpoint_dir} already exists")

    if jax.process_index() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Wait for the directory to be created everywhere. It may take some time
    # for the newly created directory to become visible across all training nodes.
    # This is needed for distributed filesystems that doesn't support strong consistency read after write.
    while not checkpoint_dir.exists():
        logging.info(f"Waiting for checkpoint dir to exist: {checkpoint_dir}")
        time.sleep(1.0)
    multihost_utils.sync_global_devices("checkpoint dir created")

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=config.keep_interval,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. in this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def init_model(
    config: Config,
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
        lr = config.lr_schedule.create()

        weight_decay_mask = None
        if config.weight_decay_keys_regex:
            weight_decay_mask = training_utils.mask_from_regex(config.weight_decay_keys_regex, model.params)
            logging.info(
                training_utils.to_readable_mask_info(weight_decay_mask, lambda x: "decay" if x else "no decay")
            )
        freeze_mask = None
        if config.freeze_keys_regex:
            freeze_mask = training_utils.mask_from_regex(config.freeze_keys_regex, model.params)
            logging.info(training_utils.to_readable_mask_info(freeze_mask, lambda x: "frozen" if x else "unfrozen"))
            # cast frozen weights to bfloat16
            model = model.replace(
                params=jax.tree.map(lambda x, f: x.astype(jnp.bfloat16) if f else x, model.params, freeze_mask)
            )
        tx = config.optimizer.create(lr, weight_decay_mask, freeze_mask)
        return train_state.TrainState.create(
            apply_fn=model.compute_loss,
            params=model.params["params"],
            tx=tx,
        )

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_parallel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("dp",)))
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
    if config.load_pretrained_weights:
        state = training_utils.restore_weights(config.load_pretrained_weights, state, state_sharding)
    return state, state_sharding


def train_step(
    rng: at.KeyArrayLike, state: train_state.TrainState, batch: tuple[_common.Observation, _common.Actions]
) -> train_state.TrainState:
    old_params = state.params
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    loss, grads = jax.value_and_grad(state.apply_fn, argnums=4)(
        train_rng, observation, actions, True, {"params": state.params}
    )
    state = state.apply_gradients(grads=grads["params"])

    used_grads = jax.tree.map(
        lambda g, old, new: g if old is not new else None, grads["params"], old_params, state.params
    )
    kernel_mask = training_utils.mask_from_regex(r".*\['kernel'\]", state.params)
    kernel_params = jax.tree.map(lambda p, m: p if m else None, state.params, kernel_mask)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(used_grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return state, info


def main(checkpoint_dir: str, seed: int, *, overwrite: bool = False, resume: bool = False):
    init_logging()
    config = Config()
    jax.distributed.initialize()

    rng = jax.random.key(seed)
    rng, init_rng = jax.random.split(rng)

    # data parallel only
    mesh_shape = (jax.device_count(),)
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh(mesh_shape), ("dp",))
    data_parallel_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("dp",)))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = initialize_checkpoint(config, checkpoint_dir, overwrite=overwrite, resume=resume)
    multihost_utils.sync_global_devices("init_checkpoint")

    model = _model.Model(module=pi0.Module())
    data_loader = init_data_loader(config, model, mini_batch_size=8, dp_sharding=data_parallel_sharding)
    batch = next(data_loader)
    logging.info(f"Data loader initialized: {training_utils.to_tree_info(batch)}")
    multihost_utils.sync_global_devices("init_dataloader")

    train_state, train_state_sharding = init_model(config, model, init_rng, batch, mesh, resume=resuming)
    logging.info(f"Initialized train state: {training_utils.to_tree_info(train_state.params)}")
    multihost_utils.sync_global_devices("init_train_state")

    if resuming:
        train_state = training_utils.restore_state(checkpoint_manager, train_state)
        multihost_utils.sync_global_devices("resume_training")

    # jax typechecking doesn't like sharding and jit
    ptrain_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_parallel_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    writer = metric_writers.create_default_writer(checkpoint_dir, just_logging=True)
    report_progress = periodic_actions.ReportProgress(writer=writer, num_train_steps=config.num_train_steps)

    infos = []
    for step in range(train_state.step, config.num_train_steps):
        with report_progress.timed("train_step"), at.disable_typechecking():
            train_state, info = ptrain_step(rng, train_state, batch)
            infos.append(info)
        if step % config.log_interval == 0:
            with report_progress.timed("write_scalars"):
                stacked_infos = common_utils.stack_forest(infos)
                writer.write_scalars(step, jax.tree_util.tree_map(jnp.mean, stacked_infos))
                infos = []
        with report_progress.timed("next_batch"):
            batch = next(data_loader)

        report_progress(step)
        if step % config.save_interval == 0:
            training_utils.save_state(checkpoint_manager, train_state, step)


if __name__ == "__main__":
    main(checkpoint_dir="/tmp/openpi/exp", seed=42, overwrite=True, resume=False)
