import jax.numpy as jnp

from openpi import transforms


class CalvinInputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation/state"], self._action_dim)

        base_image = data["observation/rgb_static"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": data["observation/rgb_static"],
                "left_wrist_0_rgb": data["observation/rgb_gripper"],
                "right_wrist_0_rgb": jnp.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": jnp.ones(1, dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones(1, dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.zeros(1, dtype=jnp.bool_),
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class CalvinOutputs(transforms.DataTransformFn):
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = jnp.asarray(data["actions"][..., :8])
        return {"actions": actions}
