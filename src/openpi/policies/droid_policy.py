import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]])
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = data["observation/exterior_image_1_left"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": data["observation/exterior_image_1_left"],
                "left_wrist_0_rgb": data["observation/wrist_image_left"],
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidInputsFAST(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]])
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = data["observation/exterior_image_1_left"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": data["observation/exterior_image_1_left"],
                "base_1_rgb": np.zeros_like(base_image),
                "wrist_0_rgb": data["observation/wrist_image_left"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.True_,  # We don't mask out padding images for FAST models
                "wrist_0_rgb": np.True_,
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}
