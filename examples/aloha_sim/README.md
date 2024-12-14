# Run Aloha Sim

## With Docker

```
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Terminal window 2:

```
# Run the server
uv run scripts/serve_policy.py --mode SIM
```
