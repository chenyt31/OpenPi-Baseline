from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

import openpi.models.common as _common
import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class FakeDataset(Dataset):
    def __init__(self, model: _model.Model, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = _model.create_inputs_spec(model)

    def __getitem__(self, index: SupportsIndex) -> dict:
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

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model: _model.Model) -> Dataset:
    repo_id = data_config.repo_id
    if repo_id == "fake":
        return FakeDataset(model, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    return lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={"action": [t / dataset_meta.fps for t in range(model.action_horizon)]},
        root=data_config.dataset_root,
    )


def create_data_loader(
    config: _config.TrainConfig,
    model: _model.Model,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    max_batches: int | None = None,
) -> DataLoader[tuple[_common.Observation, _common.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.metadata_dir, model)
    dataset = create_dataset(data_config, model)

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. Make sure to run `compute_norm_stats.py` first.")
        norm_stats = data_config.norm_stats

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, transforms: Sequence[_transforms.DataTransformFn]):
            self._data_config = data_config
            self._transforms = transforms

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in simple_data_loader(
                dataset,
                local_batch_size=config.batch_size // jax.process_count(),
                sharding=sharding,
                transforms=self._transforms,
                max_batches=max_batches,
            ):
                # Perform this after the batch has been sharded.
                yield _common.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(
        data_config,
        transforms=[
            *data_config.input_transforms,
            _transforms.Normalize(norm_stats),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model.max_token_len),
                default_prompt=data_config.default_prompt,
            ),
        ],
    )


def simple_data_loader(
    dataset: Dataset,
    local_batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    transforms: Sequence[_transforms.DataTransformFn] = (),
    shuffle: bool = False,
    max_batches: int | None = None,
) -> Iterator[at.PyTree[jax.Array]]:
    """Simple single-threaded data loader. Will be replaced with a proper one in the future."""
    if jax.process_count() > 1:
        # TODO: Update the sampler to support multiple processes.
        raise NotImplementedError("Data loader with multiple processes is not supported.")

    num_samples = len(dataset)
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

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
