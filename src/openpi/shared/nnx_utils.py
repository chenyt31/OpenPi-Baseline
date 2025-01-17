from collections.abc import Callable
import functools

import flax.nnx as nnx
import jax


def pure_jit(fn: Callable, *jit_args, **jit_kwargs):
    """A decorator that can be used to JIT-compile methods of `nnx.Module`.

    Why not `nnx.jit`? For some reason, naively applying `nnx.jit` to `nnx.Module` methods uses much more memory than
    necessary. I'm guessing it has something to do with the fact that it must keep track of module mutations.

    `pure_jit` is an alternative that works on `nnx.Module` methods that are *completely* pure. Unlike `nnx.jit`, the
    wrapped method must not return any `nnx.Module`s. Mutations to the `self` module are allowed, and will persist for
    the duration of the method, but they will not be propagated back to the caller.

    Similarly to `nnx.jit`, `pure_jit` incurs some performance overhead compared to a standard `jax.jit`. This is because
    the NNX graph traversal logic must run on every call (for `nnx.split`). On my workstation, I measured this overhead
    to be about 0.5ms on average for the Pi0 model, which seems acceptable. Hopefully this overhead will become
    negligible once flaxlib is complete; see https://github.com/google/flax/discussions/4224 for context.
    """

    @functools.partial(jax.jit, *jit_args, **jit_kwargs)
    def pure_fn(graphdef_and_state, *args, **kwargs):
        return fn(nnx.merge(*graphdef_and_state), *args, **kwargs)

    @functools.wraps(fn)
    def wrapper(module: nnx.Module, *args, **kwargs):
        if not isinstance(module, nnx.Module):
            raise ValueError("pure_jit must only be used on methods of nnx.Module.")

        return pure_fn(nnx.split(module), *args, **kwargs)

    return wrapper
