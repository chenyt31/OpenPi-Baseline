from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

from openpi.base import array_typing as at
from openpi.base import normalize as _normalize

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
    def __init__(self, norm_stats: at.PyTree[NormStats], *, strict: bool = False):
        self._norm_stats = norm_stats
        self._strict = strict

    def __call__(self, data: dict) -> dict:
        def unnormalize(x, stats: NormStats):
            return x * (stats.std + 1e-6) + stats.mean

        return apply_tree(data, self._norm_stats, unnormalize, strict=self._strict)


class AlohaInputs(DataTransformFn):
    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        data = traverse_util.flatten_dict(data, sep="/")

        # Assume that base image always exists.
        base_image = data["observation/image/cam_low"]
        batch_size = base_image.shape[:-3]

        images = {
            "base_0_rgb": base_image,
            "base_0_rgb_mask": jnp.ones(batch_size, dtype=jnp.bool_),
        }

        # Add the extra images.
        extra_images = {
            "base_1_rgb": "observation/image/cam_high",
            "left_wrist_0_rgb": "observation/image/cam_left_wrist",
            "right_wrist_0_rgb": "observation/image/cam_right_wrist",
        }
        for dest, source in extra_images.items():
            if source in data:
                images[dest] = data[source]
                images[dest + "_mask"] = jnp.ones(batch_size, dtype=jnp.bool_)
            else:
                images[dest] = jnp.zeros_like(base_image)
                images[dest + "_mask"] = jnp.zeros(batch_size, dtype=jnp.bool_)

        inputs = {
            "image": images,
            "state": pad_to_dim(data["observation/qpos"], self._action_dim),
        }

        # Actions are only available during training.
        if "action/qpos" in data:
            # TODO(ury): We need to convert this to delta actions. Make sure that this is the
            # case when we do training.
            inputs["actions"] = pad_to_dim(data["action/qpos"], self._action_dim)

        return inputs


class AlohaOutputs(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Convert from delta to absolute actions.
        actions = jnp.expand_dims(data["state"], axis=-2) + data["actions"]
        # Only return the first 14 actions.
        return {"action/qpos": actions[..., :14]}


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
