"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import glob
import os
import pickle
import random
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro
from abc import abstractmethod

class ProprioModeBase:
    @abstractmethod
    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        pass

    @abstractmethod
    def get_description(self):
        """Returns a brief description of the action."""
        pass

    @abstractmethod
    def get_shape(self):
        """Returns the shape the action."""
        pass

class JointPositionsProprioMode(ProprioModeBase):
    def __init__(self, joint_count):
        self.joint_count = joint_count

    def convert_low_dim_obs_to_proprio(self, step: int, low_dim_obs: list[dict]):
        return np.concatenate(
            [low_dim_obs[step].joint_positions, [low_dim_obs[step].gripper_open]],
            axis=-1,
            dtype=np.float32,
        )

    def get_description(self):
        return f"Joint positions angles ({self.joint_count}) and the discrete gripper state (1)."

    def get_shape(self):
        return (self.joint_count + 1,)


def main(data_dir: str, repo_name: str, *, push_to_hub: bool = False):
    REPO_NAME = repo_name
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "front_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            # "left_shoulder_image": {
            #     "dtype": "image",
            #     "shape": (256, 256, 3),
            #     "names": ["height", "width", "channel"],
            # },
            # "right_shoulder_image": {
            #     "dtype": "image",
            #     "shape": (256, 256, 3),
            #     "names": ["height", "width", "channel"],
            # },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Initialize proprio mode
    PROPRIO_MODE = JointPositionsProprioMode(joint_count=7)  # 7 joints for single arm
    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    
    task_folders = glob.glob(data_dir + "/*")
    for task_folder in task_folders:
        if not os.path.isdir(task_folder):
            continue
        
        # Extract the task name from the last part of the task_folder path
        task_name = os.path.basename(task_folder)
        
        episodes_paths = glob.glob(task_folder + "/all_variations/episodes/*")

        for episode_path in episodes_paths:

            # language
            descriptions_path = episode_path + "/variation_descriptions.pkl"
            with open(descriptions_path, "rb") as file:
                description = pickle.load(file)['vanilla'][0]

            # action
            low_dim_obs_path = episode_path + "/low_dim_obs.pkl"
            with open(low_dim_obs_path, "rb") as file:
                low_dim_obs = pickle.load(file)

            for step_idx, step in enumerate(low_dim_obs):
                if step_idx == len(low_dim_obs) - 1:
                    break
                front_image_path = episode_path + f"/front_rgb/{step_idx}.png"
                # left_shoulder_image_path = episode_path + f"/left_shoulder_rgb/{step_idx}.png"
                # right_shoulder_image_path = episode_path + f"/right_shoulder_rgb/{step_idx}.png"
                wrist_image_path = episode_path + f"/wrist_rgb/{step_idx}.png"
                front_image = np.array(Image.open(front_image_path))
                # left_shoulder_image = np.array(Image.open(left_shoulder_image_path))
                # right_shoulder_image = np.array(Image.open(right_shoulder_image_path))
                wrist_image = np.array(Image.open(wrist_image_path))
                proprio = PROPRIO_MODE.convert_low_dim_obs_to_proprio(step_idx, low_dim_obs)
                proprio_next = PROPRIO_MODE.convert_low_dim_obs_to_proprio(step_idx+1, low_dim_obs)
                dataset.add_frame(
                    {
                        "front_image": front_image,
                        # "left_shoulder_image": left_shoulder_image,
                        # "right_shoulder_image": right_shoulder_image,
                        "wrist_image": wrist_image,
                        "state": proprio,
                        "actions": proprio_next,
                        "task": description
                    }
                )
            dataset.save_episode()

    # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
