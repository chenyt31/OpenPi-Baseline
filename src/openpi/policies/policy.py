import abc
from collections.abc import Sequence
import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.base import array_typing as at
from openpi.models import model as _model

BatchSpec: TypeAlias = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    # Names of all input fields and their shapes and dtypes.
    input_spec: BatchSpec
    # Names of all output fields and their shapes and dtypes.
    output_spec: BatchSpec
    # Expected image resolution.
    input_resolutions: dict[str, tuple[int, int]]
    # Whether to add masks.
    support_masking: bool
    # History dimension.
    seq_len_dim: int
    # State dimension. Not valid for all models.
    state_dim: int


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        """Infer actions from observations."""


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.Model,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._model = model
        self._input_transform = _transforms.CompositeTransform(transforms)
        self._output_transform = _transforms.CompositeTransform(output_transforms)
        self._tokenizer = None

    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        inputs = _make_batch(obs)
        inputs = self._input_transform(inputs)
        # TODO: Add check that ensures that all the necessary inputs are present.
        outputs = {"actions": self._model.sample_actions(inputs)}
        outputs = self._output_transform(outputs)
        return _unbatch(jax.device_get(outputs))


class ActionChunkBroker(BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: BasePolicy):
        self._policy = policy

        self._cur_inference: dict | None = None
        self._cur_idx: int = 0
        self._cur_len: int = 0

    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        if self._cur_inference is None:
            self._cur_inference = self._policy.infer(obs)
            self._cur_idx = 0
            self._cur_len = jax.tree_util.tree_reduce(
                lambda prev, x: max(prev, x.shape[0]),
                self._cur_inference,
                0,
            )

        result = jax.tree_util.tree_map(lambda x: x[self._cur_idx, ...], self._cur_inference)

        self._cur_idx += 1
        if self._cur_idx >= self._cur_len:
            self._cur_inference = None

        return result


def _make_batch(data: dict) -> dict:
    def _transform(x: np.ndarray | str) -> np.ndarray:
        # TODO: How to handle strings?
        if isinstance(x, str):
            return x
        return x[jnp.newaxis, ...]

    return jax.tree_util.tree_map(_transform, data)


def _unbatch(data: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: x[0, ...], data)
