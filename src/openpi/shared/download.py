import concurrent.futures
import logging
import os
import pathlib
import stat
import time
from urllib.parse import urlparse

import boto3
import boto3.s3.transfer as s3_transfer
import botocore
import filelock
import fsspec
import fsspec.generic
import s3transfer.futures as s3_transfer_futures
import tqdm_loggable.auto as tqdm
from types_boto3_s3.service_resource import ObjectSummary

# Environment variable to control cache directory path, ~/.cache/openpi will be used by default.
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"


def is_openpi_url(url: str) -> bool:
    """Check if the url is an OpenPI S3 bucket url."""
    return url.startswith("s3://openpi-assets")


def download(url: str, **kwargs) -> pathlib.Path:
    """Download a file from a remote filesystem to the local cache, and return the local path."""
    cache_dir = _cache_dir()
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    if not fs.exists(path):
        raise FileNotFoundError(f"File not found at {url}")
    protocol_str = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    local_path = cache_dir / protocol_str / path
    if local_path.exists():
        return local_path

    total_size = fs.du(url)
    with (
        tqdm.tqdm(
            total=total_size, desc=f"Downloading {url} to {local_path}", unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar,
        filelock.FileLock(local_path / ".lock"),
    ):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=True)
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        pbar.update(total_size - pbar.n)
    _ensure_permissions(local_path)
    return local_path


def download_openpi(url: str, *, boto_session: boto3.Session = None, workers: int = 20) -> pathlib.Path:
    """Download a file from the OpenPI S3 bucket using boto3. This is a more performant version of download but can
    only handle s3 urls. In openpi repo, this is mainly used to access assets in S3 with higher throughput.

    Input:
        url: URL to openpi checkpoint path.
        boto_session: Optional boto3 session, will create by default if not provided.
        workers: number of workers for downloading.
    Output:
        pathlib.Path: local path to the downloaded file.
    """

    def validate_and_parse_url(maybe_s3_url: str) -> tuple[str, str]:
        parsed = urlparse(maybe_s3_url)
        if parsed.scheme != "s3":
            raise ValueError(f"URL must be an S3 URL (s3://), got: {maybe_s3_url}")
        bucket_name = parsed.netloc
        prefix = parsed.path.strip("/")
        return bucket_name, prefix

    cache_dir = _cache_dir()
    bucket_name, prefix = validate_and_parse_url(url)
    s3api = boto3.resource("s3")
    bucket = s3api.Bucket(bucket_name)
    return_dir = cache_dir / prefix
    return_dir.mkdir(parents=True, exist_ok=True)

    objects = list(bucket.objects.filter(Prefix=prefix))
    total_size = sum(obj.size for obj in objects)

    session = boto_session or boto3.Session()
    s3t = _get_s3_transfer_manager(session, workers)

    def transfer(
        s3obj: ObjectSummary, local_path: pathlib.Path, progress_func
    ) -> s3_transfer_futures.TransferFuture | None:
        if local_path.exists():
            local_stat = local_path.stat()
            if s3obj.size == local_stat.st_size:
                progress_func(s3obj.size)
                return None
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return s3t.download(
            bucket_name,
            s3obj.key,
            str(local_path),
            subscribers=[
                s3_transfer.ProgressCallbackInvoker(progress_func),
            ],
        )

    try:
        with (
            tqdm.tqdm(
                total=total_size,
                desc=f"Downloading {url} to {return_dir}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
            filelock.FileLock(return_dir / ".lock"),
        ):
            futures = []
            for obj in objects:
                relative_path = pathlib.Path(obj.key).relative_to(prefix)
                local_path = return_dir / relative_path
                if future := transfer(obj, local_path, pbar.update):
                    futures.append(future)
                # S3 TransferFuture is not a concurrent.futures.Future, so we need to manually call result()
                # to wait for futures.
                for future in futures:
                    future.result()
    finally:
        s3t.shutdown()
    _ensure_permissions(return_dir)
    return return_dir


def _get_s3_transfer_manager(session: boto3.Session, workers: int) -> s3_transfer.TransferManager:
    botocore_config = botocore.config.Config(max_pool_connections=workers)
    s3client = session.client("s3", config=botocore_config)
    transfer_config = s3_transfer.TransferConfig(
        use_threads=True,
        max_concurrency=workers,
    )
    return s3_transfer.create_transfer_manager(s3client, transfer_config)


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod requires executable permission to be set, so we skip if the permission is already match with the target."""
    if path.stat().st_mode & target_permission == target_permission:
        logging.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logging.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """Set folder permission to be read, write and searchable."""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, "~/.cache/openpi")).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def _ensure_permissions(path: pathlib.Path) -> None:
    """Since we are sharing cache directory with containerized runtime as well as training script, we need to
    ensure that the cache directory has the correct permissions.
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = _cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """Set all files to be read & writable, if it is a script, keep it as a script."""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)
