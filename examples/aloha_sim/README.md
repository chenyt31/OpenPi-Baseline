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
uv pip install -e .

# Run the simulation
python examples/aloha_sim/main.py
```

Terminal window 2:

```
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate
uv pip sync requirements.txt
uv pip install -e .

# Run the server
python src/openpi/serving/http_policy_server_pi0.py --mode SIM
```
