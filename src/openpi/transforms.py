from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.base import array_typing as at
from openpi.base import normalize as _normalize
from openpi.models import tokenizer as _tokenizer

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
        data = self._aloha_to_pi_request(data["qpos"], data["image"])
        data = traverse_util.flatten_dict(data, sep="/")

        # Assume that base image always exists.
        base_image = data["observation/image/cam_high"]
        batch_size = base_image.shape[:-3]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        }

        # Add the extra images.
        extra_images = {
            "left_wrist_0_rgb": "observation/image/cam_left_wrist",
            "right_wrist_0_rgb": "observation/image/cam_right_wrist",
        }
        for dest, source in extra_images.items():
            if source in data:
                images[dest] = data[source]
                image_masks[dest] = jnp.ones(batch_size, dtype=jnp.bool_)
            else:
                images[dest] = jnp.zeros_like(base_image)
                image_masks[dest] = jnp.zeros(batch_size, dtype=jnp.bool_)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": pad_to_dim(data["observation/qpos"], self._action_dim),
        }

        # Actions are only available during training.
        if "action/qpos" in data:
            # TODO(ury): We need to convert this to delta actions. Make sure that this is the
            # case when we do training.
            inputs["actions"] = pad_to_dim(data["action/qpos"], self._action_dim)

        return inputs

    def _aloha_to_pi_request(self, qpos: np.ndarray, image: np.ndarray) -> dict:
        # qpos is (14,) of type float:
        # 0-5: left arm joint angles
        # 6: left arm gripper
        # 7-12: right arm joint angles
        # 13: right arm gripper
        #
        # image is [cam_idx, channel, height, width] with values in [0, 1] of type float
        # With real robot, cam_idx order is [cam_high, cam_low, cam_left_wrist, cam_right_wrist]
        # With sim, cam_idx order is [cam_high].

        # Convert to [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
        pi_qpos = qpos[jnp.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])]

        obs = {
            "observation": {
                "qpos": pi_qpos,
                "image": {},
            },
        }

        def add_image(cam_idx: int, key: str) -> None:
            if cam_idx >= image.shape[0]:
                return

            # Convert to [height, width, channel]
            converted_image = jnp.transpose(image[cam_idx, :, :, :] * 255, (1, 2, 0)).astype(jnp.uint8)

            obs["observation"]["image"][key] = converted_image
            obs["observation"]["image"][f"{key}_mask"] = jnp.array(True)

        add_image(0, "cam_high")
        add_image(1, "cam_low")
        add_image(2, "cam_left_wrist")
        add_image(3, "cam_right_wrist")

        return obs


class AlohaOutputs(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Convert from delta to absolute actions.
        actions = jnp.expand_dims(data["state"], axis=-2) + data["actions"]
        # Only return the first 14 actions.
        qpos = actions[..., :14]
        # Convert to [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
        qpos = qpos[jnp.array([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])]

        return {"qpos": qpos}


class TokenizePrompt(DataTransformFn):
    # This is the default text prompt for the model.
    DEFAULT_PROMPT = "be a good robot"

    def __init__(self, tokenizer: _tokenizer.Tokenizer, default_prompt: str = DEFAULT_PROMPT):
        self._tokenizer = tokenizer
        self._default_prompt = default_prompt

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
