# Run Aloha (Real Robot)

This example demonstrates how to run with a real robot using an [ALOHA setup](https://github.com/tonyzhaozh/aloha).

## Prerequisites

This repo uses a fork of the ALOHA repo, with very minor modifications to use Realsense cameras.

1. Follow the [hardware installation instructions](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation) in the ALOHA repo.
1. Modify the `third_party/aloha/aloha_scripts/realsense_publisher.py` file to use serial numbers for your cameras.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python examples/aloha_real/main.py
```

Terminal window 2:

```bash
roslaunch --wait aloha ros_nodes.launch
```

Terminal window 3:

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```
## **Model Guide**

The $\pi_0$ base model is an out-of-the-box model for general tasks. While we strongly recommend fine-tuning the model to your own data to adapt it to particular tasks, it may be possible to prompt the model to attempt some tasks that were in the pre-training data.  For example, the model has been trained to take toast out of a toaster when prompted to "take the toast out of the toaster".

We additionally provide a few example policies fine-tuned to:
- Fold towels (config: `pi0_aloha_towel`)
- Take food out of tupperware and place it onto a plate (config: `pi0_aloha_tupperware`)

While we have found these policies to work in unseen conditions across multiple ALOHA stations, we provide some pointers below on how to best set up scenes to maximize the chance of policy success.  

We cover:
- The prompts to use for each policy
- Objects we’ve seen it work well on  
- Well-represented initial state distributions  

---

### **Toast Task**  

This task involves the robot taking **two pieces of toast out of a toaster** and placing them on a plate.  

- **Prompt**: "take the toast out of the toaster"
- **Objects needed**: Two pieces of toast, a plate, and a standard toaster.  
- **Object Distribution**:  
  - Works on both real toast and rubber fake toast  
  - Compatible with standard 2-slice toasters  
  - Works with plates of varying colors  

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- The toaster should be positioned in the top-left quadrant of the workspace.  
- Both pieces of toast should start inside the toaster, with at least 1 cm of bread sticking out from the top.  
- The plate should be placed roughly in the lower-center of the workspace.  
- Works with both natural and synthetic lighting, but avoid making the scene too dark (e.g., don't place the setup inside an enclosed space or under a curtain).  


### **Towel Task**  

This task involves folding a small towel (e.g., roughly the size of a hand towel) into eighths.

- **Prompt**: "fold the towel"  
- **Object Distribution**:  
  - Works on towels of varying solid colors 
  - Performance is worse on heavily textured or striped towels 

### **Scene Setup Guidelines**  
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- The towel should be flattened and roughly centered on the table.  
- Choose a towel that does not blend in with the table surface.  


### **Tupperware Task**  

This task involves opening a tupperware filled with food and pouring the contents onto a plate.  

- **Prompt**: "open the tupperware and put the food on the plate"
- **Objects needed**: Tupperware, food (or food-like items), and a plate.  
- **Object Distribution**:  
  - Works on various types of fake food (e.g., fake chicken nuggets, fries, and fried chicken).  
  - Compatible with tupperware of different lid colors and shapes, with best performance on square tupperware with a corner flap (see images below).  
  - The policy has seen plates of varying solid colors.  

### **Scene Setup Guidelines** 
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- Best performance observed when both the tupperware and plate are roughly centered in the workspace.  
- Positioning:  
  - Tupperware should be on the left.  
  - Plate should be on the right or bottom.  
  - The tupperware flap should point toward the plate.  

## Training on your own Aloha dataset

openpi suppports training on data collected in the default aloha hdf5 format using the `examples/aloha_real/aloha_hd5.py` conversion script. Once the dataset is converted, add a new `TrainConfig` to `src/openpi/training/configs.py` (see the `pi0_aloha_static_cups_open` example config) and replace repo id with the id assigned to your dataset during conversion. Before training on a new dataset, you must first compute the norm stats using `scripts/compute_norm_stats.py`.
