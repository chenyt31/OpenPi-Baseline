import logging

from etils import epath
import jax
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.training.utils as training_utils


def initialize_checkpoint(
    checkpoint_dir: str, keep_interval: int, *, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir)
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(f"Checkpoint directory {checkpoint_dir} already exists")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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


def save_state(checkpoint_manager: ocp.CheckpointManager, state: training_utils.TrainState, step: int):
    with at.disable_typechecking():
        checkpoint_manager.save(step, args=ocp.args.PyTreeSave(state))


def restore_state(
    checkpoint_manager: ocp.CheckpointManager, state: training_utils.TrainState, step: int | None = None
) -> training_utils.TrainState:
    with at.disable_typechecking():
        return checkpoint_manager.restore(step=step, args=ocp.args.PyTreeRestore(state))


def restore_params(ckpt_path: str, sharding: jax.sharding.Sharding | None = None) -> at.Params:
    """Restores params (but not optimizer state) from a given checkpoint saved with `save_state`."""
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(ckpt_path)
        # Use EMA params if they exist, otherwise regular params.
        params_name = "ema_params" if metadata.get("ema_params") is not None else "params"
        item = {params_name: metadata[params_name]}

        return ckptr.restore(
            ckpt_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), item),
                transforms={},
            ),
        )[params_name]
