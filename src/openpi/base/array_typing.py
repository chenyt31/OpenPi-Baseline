import contextlib
import functools as ft

import jax
import jax.core
from jaxtyping import Array as _Array
from jaxtyping import Bool  # noqa
from jaxtyping import DTypeLike  # noqa
from jaxtyping import Float  # noqa
from jaxtyping import Int  # noqa
from jaxtyping import Key  # noqa
from jaxtyping import Num  # noqa
from jaxtyping import PyTree
from jaxtyping import Real  # noqa
from jaxtyping import Shaped
from jaxtyping import UInt8  # noqa
from jaxtyping import config
from jaxtyping import jaxtyped
import typeguard

# TODO(ury): Review this and consider switching to jax.typing

# Support jax.ShapeDtypeStruct for compatibility with jax.eval_shape
# Support jax.core.ShapedArray for compatibility with various Flax transforms (e.g., nn.scan)
Array = _Array | jax.ShapeDtypeStruct | jax.core.ShapedArray
ArrayLike = jax.typing.ArrayLike
KeyArrayLike = jax.typing.ArrayLike

Params = PyTree  # don't actually check leaves, too slow
OptState = PyTree  # don't actually check leaves, too slow
Batch = PyTree[Shaped[ArrayLike, "b ..."]]

typecheck = ft.partial(jaxtyped, typechecker=typeguard.typechecked)


@contextlib.contextmanager
def disable_typechecking():
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)
    yield
    config.update("jaxtyping_disable", initial)
