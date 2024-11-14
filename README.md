# openpi

See the design doc [here](https://docs.google.com/document/d/1ykjuV0GjuaYGzhppasmGlldJ5TOA-UCETBHKoxeoyqw/edit).

## Usage

```bash
uv run examples/hello.py
```

## Downloading checkpoints

```bash
aws s3 sync s3://openpi-assets/checkpoints/pi0_base ./checkpoints/pi0_base
```

## Running Examples

The recommended way to run the examples is using docker compose. This will avoid software installation, and also
allow you to avoid installing ROS even when running ROS code.

Docker installation instructions are [here](https://docs.docker.com/engine/install/). If using a GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). If your host machine is Ubuntu 22.04, you can use the convenience scripts `scripts/install_docker_ubuntu22.sh` and `scripts/install_nvidia_container_toolkit.sh`.

The first time you run the examples, docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.

### ALOHA+ACT Sim

```bash
docker compose -f docker/aloha_act/compose_sim.yml up --build
```
