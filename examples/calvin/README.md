# CALVIN Benchmark

This example runs the CALVIN benchmark: https://github.com/mees/calvin

## With Docker

```bash
export SERVER_ARGS="--mode CALVIN"
docker compose -f examples/calvin/compose.yml up --build
```

## Without Docker

The CALVIN installation procedure is finicky, and is quite dependent on specific versions of host software including CMake and setuptools. Because of this, we recommend using Docker for this example.

Otherwise, refer to the [CALVIN github repo](https://github.com/mees/calvin) and the Dockerfile in this directory.