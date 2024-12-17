from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.common as _common
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class FakeDataset(Dataset):
    def __init__(self, model: _model.Model, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = _model.create_inputs_spec(model)

    def __getitem__(self, index: SupportsIndex) -> tuple[_common.Observation, _common.Actions]:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return observation, action

    def __len__(self) -> int:
        return self._num_samples


def data_loader(
    dataset: Dataset,
    local_batch_size: int,
    sharding: jax.sharding.Sharding,
    *,
    transforms: Sequence[_transforms.DataTransformFn] = (),
    shuffle: bool = False,
    max_batches: int | None = None,
) -> Iterator[at.PyTree[jax.Array]]:
    if jax.process_count() > 1:
        # TODO: Update the sampler to support multiple processes.
        raise NotImplementedError("Data loader with multiple processes is not supported.")

    num_samples = len(dataset)

    def sampler() -> Iterator[int]:
        rng = jax.random.key(0)
        while True:
            if not shuffle:
                yield from range(num_samples)
            else:
                rng, data_rng = jax.random.split(rng)
                yield from jax.random.permutation(data_rng, num_samples)

    def batch_sampler() -> Iterator[list[int]]:
        sampler_iter = iter(sampler())
        while True:
            yield [next(sampler_iter) for _ in range(local_batch_size)]

    def to_global_array(local_arr) -> jax.Array:
        global_shape = (local_arr.shape[0] * jax.process_count(), *local_arr.shape[1:])
        return jax.make_array_from_process_local_data(sharding, np.asarray(local_arr), global_shape)

    def transform(item):
        for transform in transforms:
            item = transform(item)
        return item

    def data_loader():
        batch_sampler_iter = iter(batch_sampler())
        num_batches = 0
        while True:
            if max_batches is not None and num_batches >= max_batches:
                break
            items = [transform(dataset[i]) for i in next(batch_sampler_iter)]
            batch = jax.tree.map(lambda *x: jnp.stack(jnp.asarray(x), axis=0), *items)
            batch = jax.tree.map(to_global_array, batch)
            yield batch
            num_batches += 1

    return data_loader()
