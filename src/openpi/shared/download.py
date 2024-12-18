from concurrent import futures
import pathlib
import time

import fsspec
import fsspec.generic
import tqdm_loggable.auto as tqdm


def download(url: str, **kwargs) -> pathlib.Path:
    """Download a file from a remote filesystem to the local cache, and return the local path."""
    cache_dir = pathlib.Path("~/.cache/openpi").expanduser().resolve()
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    if not fs.exists(path):
        raise FileNotFoundError(f"File not found at {url}")
    protocol_str = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    local_path = cache_dir / protocol_str / path

    total_size = fs.du(url)
    executor = futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fs.get, url, local_path, recursive=True)
    with tqdm.tqdm(total=total_size, desc=f"Downloading {url}", unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(0.1)
        pbar.update(total_size - pbar.n)
    return local_path
