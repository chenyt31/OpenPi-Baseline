"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config metadata directory.
"""

import concurrent.futures
import dataclasses
import functools
import multiprocessing

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


@dataclasses.dataclass
class TaskInfo:
    config_name: str
    chunk_size: int
    keys: list[str]


def compute_stats_task(start: int, progress_counter, task_info: TaskInfo) -> dict[str, normalize.RunningStats]:
    """Compute the normalization statistics for a given config."""
    data_config, dataset = create_dataset(_config.get_config(task_info.config_name))

    num_frames = len(dataset)
    transform = _transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])

    stats = {key: normalize.RunningStats() for key in task_info.keys}
    for i in range(start, min(start + task_info.chunk_size, num_frames)):
        data = transform(dataset[i])
        for key in task_info.keys:
            values = np.asarray(data[key])
            stats[key].update(values.reshape(-1, values.shape[-1]))
        progress_counter.value += 1

    return stats


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    model = config.create_model()
    data_config = config.data.create(config.metadata_dir, model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    return data_config, _data_loader.create_dataset(data_config, model)


def main(config_name: str, chunk_size: int = 1000):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)
    num_frames = len(dataset)
    del dataset

    keys = ["state", "actions"]
    all_stats = {key: normalize.RunningStats() for key in keys}

    task_fn = functools.partial(
        compute_stats_task,
        task_info=TaskInfo(config_name=config_name, chunk_size=chunk_size, keys=keys),
    )

    with multiprocessing.Manager() as manager:
        progress_counter = manager.Value("i", 0)
        # Forking doesn't work with HF datasets.
        mp_context = multiprocessing.get_context("spawn")
        with (
            tqdm.tqdm(total=num_frames, desc="Computing stats") as pbar,
            concurrent.futures.ProcessPoolExecutor(mp_context=mp_context) as executor,
        ):
            futures = {executor.submit(task_fn, i, progress_counter) for i in range(0, num_frames, chunk_size)}
            completed = 0
            while futures:
                done, futures = concurrent.futures.wait(
                    futures, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    stats = future.result()
                    for key in keys:
                        all_stats[key].merge(stats[key])
                new_completed = progress_counter.value
                pbar.update(new_completed - completed)
                completed = new_completed

    norm_stats = {key: stats.get_statistics() for key, stats in all_stats.items()}

    output_path = config.metadata_dir / data_config.repo_id / "norm_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing stats to: {output_path}")
    output_path.write_text(normalize.serialize_json(norm_stats))


if __name__ == "__main__":
    tyro.cli(main)
