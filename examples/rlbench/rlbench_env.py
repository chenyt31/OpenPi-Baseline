import glob
import logging
import os
import numpy as np
from examples.rlbench.rlbench_utils import (
    create_obs_config, 
    task_file_to_task_class
)

from pyrep.const import RenderMode
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.environment import Environment
from rlbench.backend.exceptions import InvalidActionError
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.errors import IKError, ConfigurationPathError
import torch
from typing import List
from examples.rlbench.rlbench_utils import Mover, Actioner

from typing import Optional, Type, List, Dict, Any
import os
import json
from omegaconf import DictConfig, OmegaConf
from colosseum import (
    ASSETS_CONFIGS_FOLDER,
    ASSETS_JSON_FOLDER,
    TASKS_TTM_FOLDER,
    ASSETS_ATOMIC_CONFIGS_FOLDER,
    ASSETS_ATOMIC_JSON_FOLDER,
    ATOMIC_TASKS_TTM_FOLDER,
    ASSETS_COMPOSITIONAL_CONFIGS_FOLDER,
    ASSETS_COMPOSITIONAL_JSON_FOLDER,
    COMPOSITIONAL_TASKS_TTM_FOLDER
)
from colosseum import TASKS_PY_FOLDER, ATOMIC_TASKS_PY_FOLDER, COMPOSITIONAL_TASKS_PY_FOLDER

from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import (
    ObservationConfigExt,
    check_and_make,
    name_to_class,
    save_demo,
)
from colosseum.variations.utils import safeGetValue
from functools import reduce

class MoveJointArmThenGripper(ActionMode):
    """The arm action is first applied, followed by the gripper action. """

    def action(self, scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

def change_case(str):
    return reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, str).lower()


def get_spreadsheet_config(
    base_cfg: DictConfig, collection_cfg: Dict[str, Any], spreadsheet_idx: int
) -> DictConfig:
    """
    Creates a new config object based on a base configuration, updated with
    entries to match the options from the data collection strategy in JSON
    format for the given spreadsheet index.

    Parameters
    ----------
        base_cfg : DictConfig
            The base configuration for the current task
        collection_cfg : Dict[str, Any]
            The data collection strategy parsed from the JSON strategy file
        spreadsheet_idx : int
            The index in the spreadsheet to use for the current task variation

    Returns
    -------
        DictConfig
            The new configuration object with the updated options for this
            variation
    """
    spreadsheet_cfg = base_cfg.copy()

    collections_variation_cfg = collection_cfg["strategy"][spreadsheet_idx][
        "variations"
    ]
    for collection_var_cfg in collections_variation_cfg:
        var_type = collection_var_cfg["type"]
        var_name = collection_var_cfg["name"]
        var_enabled = collection_var_cfg["enabled"]
        for variation_cfg in spreadsheet_cfg.env.scene.factors:
            if variation_cfg.variation != var_type:
                continue
            else:
                if var_name == "any" or (
                    "name" in variation_cfg and variation_cfg.name == var_name
                ):
                    variation_cfg.enabled = var_enabled

    return spreadsheet_cfg

class MultiTaskRLBenchEnv():

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 headless=True,
                 swap_task_every: int = 1,
                 base_cfg_name=None,
                 task_class_variation_idx=None):
        # super(MultiTaskRLBenchEnv, self).__init__()

        self._task_classes = task_classes
        self._observation_config = observation_config
        # self._rlbench_env = Environment(
        #     action_mode=action_mode, obs_config=observation_config,
        #     dataset_root=dataset_root, headless=headless)
        self._task = None
        self._task_name = ''
        self._lang_goal = 'unknown goal'
        self._swap_task_every = swap_task_every
        
        self._episodes_this_task = 0
        self._active_task_id = -1

        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        self._base_cfg_name = base_cfg_name
        self._task_class_variation_idx = task_class_variation_idx
        self._action_mode = action_mode
        self._observation_config = observation_config
        self._dataset_root = dataset_root
        self._headless = headless

        self._record_cam = None
        self._recorded_images = []

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self.descriptions = descriptions
        try:
            self._lang_goal = descriptions['vanilla'][0]
        except:
            self._lang_goal = descriptions[0]
        # self._lang_goal = descriptions[0] # first description variant

    def launch(self, task_type=None):
        if task_type == "atomic":
            ASSETS_CONFIGS_FOLDER = ASSETS_ATOMIC_CONFIGS_FOLDER
            ASSETS_JSON_FOLDER = ASSETS_ATOMIC_JSON_FOLDER
            TASKS_TTM_FOLDER = ATOMIC_TASKS_TTM_FOLDER
        elif task_type == "compositional":
            ASSETS_CONFIGS_FOLDER = ASSETS_COMPOSITIONAL_CONFIGS_FOLDER
            ASSETS_JSON_FOLDER = ASSETS_COMPOSITIONAL_JSON_FOLDER
            TASKS_TTM_FOLDER = COMPOSITIONAL_TASKS_TTM_FOLDER
            
        base_cfg_path = os.path.join(ASSETS_CONFIGS_FOLDER, f"{self._base_cfg_name[self._active_task_id]}.yaml")
        if os.path.exists(base_cfg_path):
            with open(base_cfg_path, 'r') as f:
                base_cfg = OmegaConf.load(f)

        collection_cfg_path: str = (
        os.path.join(ASSETS_JSON_FOLDER, base_cfg.env.task_name) + ".json"
        )
        collection_cfg: Optional[Any] = None
        with open(collection_cfg_path, "r") as fh:
            collection_cfg = json.load(fh)

        if collection_cfg is None:
            return 1

        if "strategy" not in collection_cfg:
            return 1

        num_spreadsheet_idx = len(collection_cfg["strategy"])
        
        if self._task_class_variation_idx != None:
            full_config = get_spreadsheet_config(
                        base_cfg,
                        collection_cfg,
                        self._task_class_variation_idx[self._active_task_id],
                    )
            _, env_cfg = full_config.data, full_config.env  
        else:
            env_cfg = None

        self._rlbench_env = EnvironmentExt(
            action_mode=self._action_mode, obs_config=self._observation_config, 
            path_task_ttms=TASKS_TTM_FOLDER,
            dataset_root=self._dataset_root, headless=self._headless, env_config=env_cfg,)
        self._rlbench_env

        self._rlbench_env.launch()
        self._set_new_task()

        # record
        self._task._scene.register_step_callback(self._my_callback)
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_base = Dummy('cam_cinematic_base')
        cam_base.rotate([0, 0, np.pi * 0.75])
        self._record_cam = VisionSensor.create([320, 180])
        self._record_cam.set_explicit_handling(True)
        self._record_cam.set_pose(cam_placeholder.get_pose())
        self._record_cam.set_render_mode(RenderMode.OPENGL)

    def _my_callback(self):
        self._record_cam.handle_explicitly()
        cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(cap)

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        descriptions, obs = self._task.reset()
        self.descriptions = descriptions
        try:
            self._lang_goal = descriptions['vanilla'][0]
        except:
            self._lang_goal = descriptions[0]
        # self._lang_goal = descriptions[0] # first description variant

        return descriptions, obs
    
    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

        vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
        return vid

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0

        if self._task_class_variation_idx != None:
            self._task.set_variation(-1)
            self._task._task.task_path = self._task._task.name + f"_{str(self._task_class_variation_idx[self._active_task_id])}"
        else:
            self._task.set_variation(-1)

        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        self._recorded_images.clear()

        descriptions, obs = self._task.reset_to_demo(d)
        self.descriptions = descriptions
        try:
            self._lang_goal = descriptions['vanilla'][0]
        except:
            self._lang_goal = descriptions[0]
        return descriptions, obs

class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_mask=False,
        headless=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        tasks_type='atomic',
        tasks=None
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_mask = apply_mask
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = create_obs_config(
            image_size=image_size,
            apply_rgb=self.apply_rgb,
            apply_depth=self.apply_depth,
            apply_pc=self.apply_pc,
            apply_cameras=self.apply_cameras
        )

        self.action_mode = MoveJointArmThenGripper(
            arm_action_mode=JointPosition(),
            gripper_action_mode=Discrete()
        )
        self.tasks_type = tasks_type
        task_classes = []
        task_class_variation_idx = []
        task_class_base = []
        for task in tasks:
            task_class_base.append('_'.join(task.split('_')[:-1]))
            # if task_class_base[-1] not in task_files:
            #     raise ValueError('Task %s not recognised!.' % task)
            if tasks_type == 'atomic':
                task_class = name_to_class(task_class_base[-1], ATOMIC_TASKS_PY_FOLDER)
            elif tasks_type == 'compositional':
                task_class = name_to_class(task_class_base[-1], COMPOSITIONAL_TASKS_PY_FOLDER)
            else:
                task_class = name_to_class(task_class_base[-1], TASKS_PY_FOLDER) # task_file_to_task_class(task_class_base)
            task_class_variation_idx.append(int(task.split('_')[-1]))
            task_classes.append(task_class)
        
        self.env = MultiTaskRLBenchEnv(
            task_classes=task_classes,
            observation_config=self.obs_config,
            action_mode=self.action_mode,
            dataset_root=str(data_path),
            headless=headless,
            swap_task_every=25,
            base_cfg_name=task_class_base,
            task_class_variation_idx=task_class_variation_idx,
        )
        self.image_size = image_size

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        eval_demo_seed: int,
        max_steps: int,
        actioner: Actioner,        
        verbose: bool = False,
        eval_mode: str = "half" # vanilla | half | vlm
    ):
        # Reset task to demo state
        instruction, obs = self.env.reset_to_demo(eval_demo_seed)
        # instr
        # ============================================================================
        instruction_vanilla = instruction['vanilla'][0]
        instruction_oracle_half = instruction['oracle_half'][0].split('\n')
        instruction = ""
        instr_index = 0
        grasped_objects = self.env._rlbench_env._scene.robot.gripper.get_grasped_objects()
        prev_grasped_objects_len = len(grasped_objects)
        # ============================================================================

        reward = 0.0
        
        # Add task information to kwargs
        kwargs = {}
        kwargs['task_str'] = task_str

        actioner.reset()

        for step_id in range(max_steps):
            # ============================================================================
            if eval_mode == "vanilla":
                instruction = instruction_vanilla
            elif eval_mode == "half":
                instruction = instruction_oracle_half[instr_index]
            elif eval_mode == "vlm":
                pass
            else:
                raise ValueError(f"unknown eval mode: {eval_mode}")
            if verbose:
                print(f'instruction: {instruction}')
            # ============================================================================
            
            # Get front RGB image
            front_rgb = obs.front_rgb
            wrist_rgb = obs.wrist_rgb
            proprio = np.concatenate(
                [obs.joint_positions, [obs.gripper_open]],
                axis=-1,
                dtype=np.float32,
            )
            
            # Get action prediction
            trajectory = actioner.predict(
                proprio,
                front_rgb,
                wrist_rgb,
                instruction,
                **kwargs
            )

            try:
                # Execute actions
                for action in trajectory:  
                    obs, reward, terminate = self.env._task.step(action)
            
                if reward == 1:
                    break

                if terminate:
                    print("Episode terminated!")
                    break
            
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(task_str, eval_demo_seed, step_id, e)
                reward = 0
                #break
            
            # plan forward according to current task
            # ============================================================================
            grasped_objects = self.env._rlbench_env._scene.robot.gripper.get_grasped_objects()
            cur_grasped_objects_len = len(grasped_objects)
            if cur_grasped_objects_len != prev_grasped_objects_len:
                instr_index = (instr_index + 1) % len(instruction_oracle_half)
            prev_grasped_objects_len = cur_grasped_objects_len
            # ============================================================================

        success = True if reward == 1 else False
        vid = self.env._append_final_frame(success)
        return reward, vid