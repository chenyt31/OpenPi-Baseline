# openpi

See the design doc [here](https://docs.google.com/document/d/1ykjuV0GjuaYGzhppasmGlldJ5TOA-UCETBHKoxeoyqw/edit).

## Usage

When cloning this repo, make sure to update submodules:

```bash
git submodule update --init --recursive
```

```bash
uv run examples/hello.py
```

## Downloading checkpoints

Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). For linux, this means running the following commands:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Run the following from the cloned `openpi` directory:

```bash
AWS_ACCESS_KEY_ID=AKIA4MTWIIQIZBO44C62 AWS_SECRET_ACCESS_KEY=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 aws s3 sync s3://openpi-assets/checkpoints/pi0_base ./checkpoints/pi0_base
```

## Docker Setup

The recommended way to run many of our examples is using Docker. This will simplify software installation, produce a more stable environment, and also
allow you to avoid installing ROS and cluttering your machine, even when depending on ROS.

Docker installation instructions are [here](https://docs.docker.com/engine/install/). If using a GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). If your host machine is Ubuntu 22.04, you can use the convenience scripts `scripts/install_docker_ubuntu22.sh` and `scripts/install_nvidia_container_toolkit.sh`.

Per [this answer](https://askubuntu.com/a/1470341) you'll probably need to run `xhost +Local:*` in a terminal for certain visualizations to render.

During the first run of any example, docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.

## ALOHA+ACT

### Running the Sim, with Docker

- Follow the instructions [above](#docker-setup) to install Docker.

```bash
docker compose -f docker/aloha_act/compose_sim.yml up --build
```

### Running the Sim, without Docker

- Perform the ALOHA software installation ([part 1](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#software-installation---ros) and [part 2](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#software-installation---conda))
- Clone [ACT](https://github.com/tonyzhaozh/act) and follow its installation instructions [here](https://github.com/tonyzhaozh/act?tab=readme-ov-file#installation).
- **Temporary**: Modify your ACT code to incorporate the changes in [this fork](https://github.com/jimmyt857/act). [Diff viewer](https://github.com/tonyzhaozh/act/compare/main...jimmyt857:main).
- Install [UV](https://docs.astral.sh/uv/getting-started/installation/).

Run ALOHA and ACT in separate terminals. Then, in another terminal, run the following from the cloned `openpi` directory:

```bash
uv run src/openpi/serving/http_policy_server_pi0.py
```

### Running the Real Robot, with Docker

- Follow the instructions [above](#docker-setup) to install Docker.
- Perform the [ALOHA hardware installation](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation).

```bash
docker compose -f docker/aloha_act/compose_real.yml up --build
```