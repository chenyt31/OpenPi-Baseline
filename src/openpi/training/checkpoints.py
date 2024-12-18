import logging

from etils import epath
import orbax.checkpoint as ocp

import openpi.training.utils as training_utils


def initialize_checkpoint(
    checkpoint_dir: epath.Path | str, keep_interval: int, *, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

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
    checkpoint_manager.save(step, args=ocp.args.PyTreeSave(state))


def restore_state(
    checkpoint_manager: ocp.CheckpointManager, state: training_utils.TrainState, step: int | None = None
) -> training_utils.TrainState:
    return checkpoint_manager.restore(step=step, args=ocp.args.PyTreeRestore(state))
