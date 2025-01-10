# Run DROID

This example shows how to run the [DROID platform](https://github.com/droid-dataset/droid) with OpenPI.

# Usage

1. With the DROID conda environment activated, run `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .` to install the OpenPI client.
1. Copy the `main.py` file from this directory to the `$DROID_ROOT/scripts` directory.
1. Start the OpenPI server via the following command:

```bash
uv run scripts/serve_policy.py --env DROID
```
