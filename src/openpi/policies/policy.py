import abc
from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias
import csv

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.base import array_typing as at
from openpi.models import common
from openpi.models import model as _model
import datetime
import os
import PIL.Image
import json
import time

class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        """Infer actions from observations."""


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._model = model
        self._input_transform = _transforms.CompositeTransform(transforms)
        self._output_transform = _transforms.CompositeTransform(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._csv_saver_path = f'/data/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self._csv_saver_path, exist_ok=True)
        self._calls = 0

    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:

        # print(f'{self._calls=}')

        # Print the observation structure
        # print("Observation structure:")
        # for key, value in obs.items():
        #     if isinstance(value, np.ndarray):
        #         print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        #     else:
        #         print(f"  {key}: {type(value)}")

        # with open(self._csv_saver_path + '/obs.jsonl', 'a') as f:
        #    f.write(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items() if len(v.shape) < 3}) + '\n')

        # # # Save images from observation
        # for i, image in enumerate(obs['image']):
        #     # Convert from C,H,W to W,H,C format
        #     PIL.Image.fromarray(np.transpose(image, (1, 2, 0))).save(os.path.join(self._csv_saver_path, f'image_{self._calls}_{i}.png'))


        inputs = _make_batch(obs)
        inputs = self._input_transform(inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        obs_obj = common.Observation.from_dict(inputs)

        # with open(self._csv_saver_path + '/obs_obj.jsonl', 'a') as f:
        #     f.write(json.dumps({"state": obs_obj.state.tolist(), "image_masks": json.dumps({k: v.tolist() for k, v in obs_obj.image_masks.items()}), "tokenized_prompt": obs_obj.tokenized_prompt.tolist(), "tokenized_prompt_mask": obs_obj.tokenized_prompt_mask.tolist()}) + '\n')
            
        outputs = {
            "state": inputs["state"],
            "actions": self._model.sample_actions(sample_rng, obs_obj),
        }
        # with open(self._csv_saver_path + '/actions_raw.jsonl', 'a') as f:
        #     f.write(json.dumps({"actions": outputs["actions"].tolist()}) + '\n')

        outputs = self._output_transform(outputs)

        # with open(self._csv_saver_path + '/actions.jsonl', 'a') as f:
        #     f.write(json.dumps({k: v.tolist() for k, v in outputs.items()}) + '\n')

        self._calls += 1

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

        # import time
        # start_time = time.time()
        results = jax.tree.map(lambda x: x[self._cur_step, ...], self._last_results)
        # print(f"Time to get results: {time.time() - start_time}")
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results


class PolicyRecorder(BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    def infer(self, obs: dict) -> at.PyTree[np.ndarray]:
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results


def _make_batch(data: dict) -> dict:
    def _transform(x: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(x)[jnp.newaxis, ...]

    return jax.tree_util.tree_map(_transform, data)


def _unbatch(data: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: np.asarray(x[0, ...]), data)
