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
        inputs = make_batch(obs)
        inputs = self._input_transform(inputs)
        # TODO: Add check that ensures that all the necessary inputs are present.
        outputs = {"actions": self._model.sample_actions(inputs)}
        outputs = self._output_transform(outputs)
        return unbatch(jax.device_get(outputs))


def make_batch(data: dict) -> dict:
    def _make_batch(x: np.ndarray | str) -> np.ndarray:
        # TODO: How to handle strings?
        if isinstance(x, str):
            return x
        return x[jnp.newaxis, ...]

    return jax.tree_util.tree_map(_make_batch, data)


def unbatch(data: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: x[0, ...], data)
