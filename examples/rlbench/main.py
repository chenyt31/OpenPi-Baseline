import logging
import numpy as np
import argparse
import tyro
import yaml
import torch
from examples.rlbench.rlbench_env import RLBenchEnv
from examples.rlbench.rlbench_utils import Actioner
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import os
from pathlib import Path
import datetime

@dataclass
class Args:
    # Model parameters
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = -1

    # Environment parameters
    data_dir: Optional[str] = ""  # Path to the dataset
    random_seed: int = 42  # Random seed for the environment
    tasks: Tuple[str, ...] = ("close_jar",)  # Tasks to evaluate, comma separated
    apply_cameras: str = "left_shoulder,right_shoulder,wrist,front"  # Cameras to use, comma separated
    start_episode: int = 0
    num_episodes: int = 1  # Number of evaluation episodes per task
    max_steps: int = 300  # Maximum steps per episode
    headless: bool = True  # Run headless
    image_size: str = "256,256"  # Image size (width,height)
    tasks_type: str = "atomic"
    
    # Utility settings
    verbose: bool = True  # Verbose output
    output_file: Path = Path(__file__).parent / "eval.json" # log file
    eval_mode: str = "vanilla"

def save_video(vid, save_path, fps=10):
    import imageio
    vid = vid.transpose(0, 2, 3, 1)  # (T, H, W, C)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, vid, fps=fps)

def eval_rlbench(args: Args) -> None:
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    log_dirpath = os.path.dirname(args.output_file)
    txt_log_path = os.path.join(log_dirpath, "episode_log.txt")

    # set random seeds
    seed = args.random_seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.apply_cameras.split(","),
        tasks_type=args.tasks_type,
        tasks=args.tasks
    )

    # Create Actioner
    actioner = Actioner(
        host=args.host,
        port=args.port,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
    )

    env.env.launch(args.tasks_type)
    num_tasks = len(args.tasks)
    for task_id in range(num_tasks):
        task_name = args.tasks[task_id]
        success_list = []
        for ep in range(args.start_episode, args.start_episode + args.num_episodes):
            sr, vid = env._evaluate_task_on_one_variation(
                task_str=task_name,
                eval_demo_seed=ep,
                max_steps=args.max_steps,
                actioner=actioner,
                verbose=args.verbose,
                eval_mode=args.eval_mode
            )
            # log eps info
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp} | [Task {task_name}] Episode {ep} -> Success={sr}\n"
            print(log_line.strip())
            with open(txt_log_path, "a") as f:
                f.write(log_line)

            # log video
            video_dir = os.path.join(log_dirpath, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{task_name}_ep{ep}_{sr}.mp4")
            save_video(vid, video_path, fps=15)
            print(f'save video in {video_path}')

            success_list.append(sr)
    
        # log avg. sr to output file
        avg_sr = float(np.mean(success_list))
        avg_line = f"[{task_name}]: {avg_sr:.3f}\n"
        print(avg_line.strip())
        with open(args.output_file, "a") as f:
            f.write(avg_line)

    env.env.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_rlbench)