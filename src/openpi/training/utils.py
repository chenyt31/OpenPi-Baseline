from collections.abc import Callable
import dataclasses
import logging
import re

from etils import epath
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from openpi.base import array_typing as at
import openpi.models.model as _model


@at.typecheck
def mask_from_regex(regex: str, pytree: at.PyTree) -> at.PyTree[bool]:
    compiled = re.compile(regex)
    return jax.tree_util.tree_map_with_path(
        lambda path, _: compiled.fullmatch(jax.tree_util.keystr(path)) is not None, pytree
    )


@at.typecheck
def to_readable_mask_info(mask: at.PyTree[bool], interp_func: Callable[[bool], str]) -> str:
    tree, _ = jax.tree_util.tree_flatten_with_path(mask)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def to_tree_info(tree: at.PyTree) -> str:
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {value.shape}@{value.dtype}" for path, value in tree)


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


@dataclasses.dataclass
class WarmupCosineDecay:
    warmup_steps: int
    decay_steps: int
    peak_lr: float
    decay_lr: float

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass
class AdamW:
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    clip_gradient_norm: float | None = 100.0
    weight_decay: float = 1e-4

    def create(
        self, lr: optax.ScalarOrSchedule, weight_decay_mask: at.PyTree, freeze_mask: at.PyTree
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            mu_dtype=jnp.bfloat16,
            weight_decay=self.weight_decay,
            mask=weight_decay_mask,
        )
        if freeze_mask is not None:
            tx = optax.multi_transform(
                {"online": tx, "offline": optax.set_to_zero()},
                jax.tree_util.tree_map(lambda x: "offline" if x else "online", freeze_mask),
            )

        if self.clip_gradient_norm is not None:
            tx = optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)

        return tx
