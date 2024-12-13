from collections.abc import Sequence

import jax.numpy as jnp

from openpi import transforms


class DroidInputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int, *, delta_action_mask: Sequence[bool] | None = None):
        self._action_dim = action_dim
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        # Note to self (Ury): The action dim in Karl's model is 32.
        # Pad from 8 to the model action dim.
        data["state"] = transforms.pad_to_dim(data["state"], self._action_dim)

        # TODO(karl): Rename these to the droid keys.
        data["image_mask"] = {
            "base_0_rgb": jnp.ones(1, dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones(1, dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones(1, dtype=jnp.bool_),
        }

        return data


class DroidOutputs(transforms.DataTransformFn):
    def __init__(self, *, delta_action_mask: Sequence[bool] | None = None):
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = jnp.asarray(data["actions"][..., :8])

        # Apply the delta action mask.
        if self._delta_action_mask is not None:
            state = jnp.asarray(data["state"][..., :8])
            mask = jnp.asarray(self._delta_action_mask[:8])
            actions = actions + jnp.expand_dims(jnp.where(mask, state, 0), axis=-2)

        return {"qpos": actions}
