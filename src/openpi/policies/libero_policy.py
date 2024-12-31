import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "image": data["observation/image"],
                "wrist_image": data["observation/wrist_image"],
            },
            "image_mask": {
                "image": np.True_,
                "wrist_image": np.True_,
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}
