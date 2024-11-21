import logging
import orbax.checkpoint as ocp
from etils import epath
import jax
from flax.training import train_state
import openpi.models.model as _model
from jax.experimental import multihost_utils


def initialize_checkpoint(
    checkpoint_dir: str, keep_interval: int, *, overwrite: bool, resume: bool
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

    multihost_utils.sync_global_devices("checkpoint dir created")

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_interval,
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


def restore_weights(
    checkpoint_dir: str,
    state: train_state.TrainState,
    sharding: jax.sharding.NamedSharding,
) -> train_state.TrainState:
    """Restores pretrained weights from a given directory. Does not restore optimizer state or model configuration.

    Args:
        checkpoint_dir: Directory to load the checkpoint from.
        state: The target TrainState to restore into.
    """
    logging.info(f"Restoring weights from {checkpoint_dir}")
    path = epath.Path(checkpoint_dir).expanduser().resolve()
    return state.replace(model=_model.restore_params(state.model, path, sharding=sharding))


def save_state(checkpoint_manager: ocp.CheckpointManager, state: train_state.TrainState, step: int) -> None:
    checkpoint_manager.save(step, args=ocp.args.PyTreeSave(state))


def restore_state(
    checkpoint_manager: ocp.CheckpointManager, state: train_state.TrainState, step: int | None = None
) -> train_state.TrainState:
    if step is None:
        step = checkpoint_manager.latest_step()

    # providing a target `TrainState` to `FlaxRestore` causes the serialized metadata to be ignored, and the
    # provided object to be used to validate shapes/dtypes/structure instead.
    logging.info(f"Restoring checkpoint from {checkpoint_manager.directory}, step {step}")
    return checkpoint_manager.restore(step, args=ocp.args.PyTreeRestore(state))
