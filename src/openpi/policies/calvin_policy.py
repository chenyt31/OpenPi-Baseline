import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class CalvinInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "rgb_static": data["observation/rgb_static"],
                "rgb_gripper": data["observation/rgb_gripper"],
            },
            "image_mask": {
                "rgb_static": np.ones(1, dtype=np.bool_),
                "rgb_gripper": np.ones(1, dtype=np.bool_),
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 15 dims.
        return {"actions": np.asarray(data["actions"][..., :15])}
