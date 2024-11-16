import abc
from collections.abc import Sequence
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.base import array_typing as at
from openpi.models import model as _model

BatchSpec: TypeAlias = dict[str, Any]


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        """Infer actions from observations."""


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.Model,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._model = model
        self._input_transform = _transforms.CompositeTransform(transforms)
        self._output_transform = _transforms.CompositeTransform(output_transforms)
        self._rng = rng or jax.random.key(0)

    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        # TODO: Add check that ensures that all the necessary inputs are present.
        inputs = _make_batch(obs)
        inputs = self._input_transform(inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {"state": inputs["state"], "actions": self._model.sample_actions(sample_rng, inputs)}
        outputs = self._output_transform(outputs)
        return _unbatch(jax.device_get(outputs))


class ActionChunkBroker(BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: BasePolicy, action_horizon: int):
        self._policy = policy

        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: np.ndarray | None = None

    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        results = jax.tree.map(lambda x: x[self._cur_step, ...], self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results


def _make_batch(data: dict) -> dict:
    def _transform(x: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(x)[jnp.newaxis, ...]

    return jax.tree_util.tree_map(_transform, data)


def _unbatch(data: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: np.asarray(x[0, ...]), data)
