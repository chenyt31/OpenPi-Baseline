from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

Batch: TypeAlias = dict[str, Any]
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


class DataTransformFn(Protocol):
    def __call__(self, data: Batch) -> Batch: ...


class CompositeTransform(DataTransformFn):
    def __init__(self, transforms: Sequence[DataTransformFn]):
        self._transforms = list(transforms)

    def __call__(self, data: Batch) -> Batch:
        for transform in self._transforms:
            data = transform(data)
        return data


class Normalize(DataTransformFn):
    def __init__(self, norm_stats: at.PyTree[NormStats], *, strict: bool = False):
        self._norm_stats = norm_stats
        self._strict = strict

    def __call__(self, data: dict) -> dict:
        def normalize(x, stats: NormStats):
            return (x - stats.mean) / (stats.std + 1e-6)

        return apply_tree(data, self._norm_stats, normalize, strict=self._strict)


class Unnormalize(DataTransformFn):
    def __init__(self, norm_stats: at.PyTree[NormStats]):
        self._norm_stats = norm_stats

    def __call__(self, data: dict) -> dict:
        def unnormalize(x, stats: NormStats):
            return x * (stats.std + 1e-6) + stats.mean

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(data, self._norm_stats, unnormalize, strict=True)


class TokenizePrompt(DataTransformFn):
    # This is the default text prompt for the model.
    DEFAULT_PROMPT = "be a good robot"

    def __init__(self, tokenizer: _tokenizer.Tokenizer, *, default_prompt: str | None = None):
        self._tokenizer = tokenizer
        self._default_prompt = default_prompt or self.DEFAULT_PROMPT

    def __call__(self, data: dict) -> dict:
        if "prompt" not in data:
            batch_size = data["state"].shape[:-1]
            prompt = np.full(batch_size, self._default_prompt)
        else:
            prompt = np.asarray(data.pop("prompt"))

        tokens, token_masks = self._tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": jnp.asarray(tokens), "tokenized_prompt_mask": jnp.asarray(token_masks)}


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = traverse_util.flatten_dict(tree, sep="/")
    selector = traverse_util.flatten_dict(selector, sep="/")

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return traverse_util.unflatten_dict({k: transform(k, v) for k, v in tree.items()}, sep="/")


def pad_to_dim(x: jax.Array, target_dim: int, axis: int = -1) -> at.Array:  # type: ignore
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return jnp.pad(x, pad_width)
    return x
