from collections.abc import Callable
import re

from flax import struct
import jax
import optax

from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: at.Params
    opt_state: at.PyTree
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: at.Params | None


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
