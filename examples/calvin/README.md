# CALVIN Benchmark

This example runs the CALVIN benchmark: https://github.com/mees/calvin

## Without Docker

Terminal window 1:

```bash
cd $OPENPI_ROOT
ENV PYTHONPATH=$PYTHONPATH:$OPENPI_ROOT/packages/openpi-client/src
conda create -n calvin python=3.8
conda activate calvin
pip install setuptools==57.5.0

git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin && ./install.sh
pip install imageio[ffmpeg] moviepy numpy==1.23.0 tqdm tyro websockets msgpack

# Download CALVIN dataset, see https://github.com/mees/calvin/blob/main/dataset/download_data.sh
export CALVIN_DATASETS_DIR=~/datasets
export CALVIN_DATASET=calvin_debug_dataset
mkdir -p $CALVIN_DATASETS_DIR && cd $CALVIN_DATASETS_DIR
wget http://calvin.cs.uni-freiburg.de/dataset/$CALVIN_DATASET.zip
unzip $CALVIN_DATASET.zip
rm $CALVIN_DATASET.zip

# Run the simulation
cd $OPENPI_ROOT
python examples/calvin/main.py --args.calvin_data_path=$CALVIN_DATASETS_DIR
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --mode CALVIN
```

## With Docker

**TODO:** This is not working yet.

```bash
export SERVER_ARGS="--mode CALVIN"
docker compose -f examples/calvin/compose.yml up --build
```
