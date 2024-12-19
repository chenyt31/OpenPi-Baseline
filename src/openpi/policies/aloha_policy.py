from collections.abc import Sequence
import dataclasses

import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import tokenizer
from openpi.policies import policy as _policy


def load_pi0_model() -> _model.Model:
    model = _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)
    return model.set_params(_model.restore_params("checkpoints/pi0_base/model"))


@dataclasses.dataclass
class PolicyConfig:
    norm_stats: dict[str, transforms.NormStats]

    default_prompt: str | None = None
    # Boolean mask for the action dimensions. If True, the action is a delta action.
    delta_action_mask: Sequence[bool] | None = None
    # If true, will adapt the joint and gripper values to match the pi runtime.
    adapt_to_pi: bool = True


def create_aloha_policy(model: _model.BaseModel, config: PolicyConfig) -> _policy.Policy:
    return _policy.Policy(
        model,
        transforms=[
            ActInputsRepack(),
            AlohaInputs(
                action_dim=model.action_dim,
                delta_action_mask=config.delta_action_mask,
                adapt_to_pi=config.adapt_to_pi,
            ),
            transforms.Normalize(config.norm_stats),
            transforms.TokenizePrompt(
                tokenizer.PaligemmaTokenizer(model.max_token_len),
                default_prompt=config.default_prompt,
            ),
        ],
        output_transforms=[
            transforms.Unnormalize(config.norm_stats),
            AlohaOutputs(
                delta_action_mask=config.delta_action_mask,
                adapt_to_pi=config.adapt_to_pi,
            ),
            ActOutputsRepack(),
        ],
    )


def make_aloha_example() -> dict:
    return {
        "qpos": np.ones((14,)),
        "image": np.random.rand(4, 3, 480, 640).astype(np.float32),
    }


def make_aloha_norm_stats():
    return {
        "actions": transforms.NormStats(
            mean=np.array(
                [
                    -1.3422864e-04,
                    1.4327176e-02,
                    2.1454914e-02,
                    9.6659490e-04,
                    -7.1675335e-03,
                    3.6924356e-04,
                    4.4476333e-01,
                    -5.5626035e-04,
                    1.8072689e-02,
                    2.0288860e-02,
                    9.7438082e-04,
                    -6.0532284e-03,
                    7.7235349e-04,
                    4.3148258e-01,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                ]
            ),
            std=np.array(
                [
                    0.10911781,
                    0.17074126,
                    0.15858743,
                    0.11406235,
                    0.17401601,
                    0.15218027,
                    0.40970784,
                    0.11649027,
                    0.18967018,
                    0.17903736,
                    0.13740747,
                    0.18568376,
                    0.18511638,
                    0.3874426,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
        "state": transforms.NormStats(
            mean=np.array(
                [
                    0.11008687,
                    0.45310053,
                    -0.60365814,
                    0.13312024,
                    0.4982536,
                    -0.20298564,
                    0.44766998,
                    -0.06511051,
                    0.305372,
                    -0.47600493,
                    -0.125554,
                    0.55477023,
                    0.21142039,
                    0.4418945,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            std=np.array(
                [
                    0.30499065,
                    0.539693,
                    0.54825234,
                    0.27636755,
                    0.47298893,
                    0.4384909,
                    0.38080454,
                    0.31720614,
                    0.56667984,
                    0.56000483,
                    0.2969444,
                    0.49062347,
                    0.54115033,
                    0.35937077,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    }


class ActInputsRepack(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # images is [..., num_cams, channel, height, width] of type uint8.
        # number of cameras (num_cams) depends on the environment.
        images = jnp.asarray(data["image"])

        num_cams = images.shape[-4]
        if num_cams == 4:
            cam_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        elif num_cams == 1:
            cam_names = ["cam_high"]
        else:
            raise ValueError(f"Expected 1 or 4 cameras, got {num_cams}")

        # `images` have shape [..., cam_idx, channel, height, width].
        image_splits = [jnp.squeeze(x, axis=-4) for x in jnp.split(images, num_cams, axis=-4)]
        images_dict = dict(zip(cam_names, image_splits, strict=True))

        return {
            "images": images_dict,
            "state": data["qpos"],
        }


class ActOutputsRepack(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"qpos": data["actions"]}


class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [..., channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [..., 14]
    - actions: [..., action_horizon, action_dim]

    Args:
        action_dim: The dimension of the action space.
        delta_action_mask: A boolean mask for the action dimensions. If None, absolute actions are used.
        adapt_to_pi: If true, will adapt the joint and gripper values to match the pi runtime.
    """

    EXPECTED_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __init__(self, action_dim: int, *, delta_action_mask: Sequence[bool] | None = None, adapt_to_pi: bool = False):
        self._action_dim = action_dim
        self._delta_action_mask = delta_action_mask
        self._adapt_to_pi = adapt_to_pi

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self._adapt_to_pi)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self._action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]
        batch_size = base_image.shape[:-3]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = jnp.ones(batch_size, dtype=jnp.bool_)
            else:
                images[dest] = jnp.zeros_like(base_image)
                image_masks[dest] = jnp.zeros(batch_size, dtype=jnp.bool_)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = jnp.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self._adapt_to_pi)

            if self._delta_action_mask is not None:
                mask = jnp.asarray(self._delta_action_mask[:14])
                actions = actions - jnp.expand_dims(jnp.where(mask, state[..., :14], 0), axis=-2)

            inputs["actions"] = transforms.pad_to_dim(actions, self._action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy.

    Args:
        delta_action_mask: A boolean mask for the action dimensions. If None, absolute actions are used.
        adapt_to_pi: If true, will adapt the joint and gripper values to match the pi runtime.
    """

    def __init__(self, *, delta_action_mask: Sequence[bool] | None = None, adapt_to_pi: bool = False):
        self._delta_action_mask = delta_action_mask
        self._adapt_to_pi = adapt_to_pi

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = jnp.asarray(data["actions"][..., :14])

        # Apply the delta action mask.
        if self._delta_action_mask is not None:
            state = jnp.asarray(data["state"][..., :14])
            mask = jnp.asarray(self._delta_action_mask[:14])
            actions = actions + jnp.expand_dims(jnp.where(mask, state, 0), axis=-2)

        return {"actions": _encode_actions(actions, adapt_to_pi=self._adapt_to_pi)}


def joint_flip_mask() -> jax.Array:
    """Used to convert between aloha and pi joint angles."""
    return jnp.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return jnp.arcsin(jnp.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = jnp.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = jnp.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, jnp.floating):
            img = (255 * img).astype(jnp.uint8)
        # Convert from [..., channel, height, width] to [..., height, width, channel].
        return einops.rearrange(img, "... c h w -> ... h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: jax.Array, *, adapt_to_pi: bool = False) -> jax.Array:
    if adapt_to_pi:
        # Flip the joints.
        state = joint_flip_mask() * state

        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state = state.at[..., 6].set(gripper_to_angular(state[..., 6]))
        state = state.at[..., 13].set(gripper_to_angular(state[..., 13]))

    return state


def _encode_actions(actions: jax.Array, *, adapt_to_pi: bool = False) -> jax.Array:
    if adapt_to_pi:
        # Flip the joints.
        actions = joint_flip_mask() * actions

        actions = actions.at[..., 6].set(gripper_from_angular(actions[..., 6]))
        actions = actions.at[..., 13].set(gripper_from_angular(actions[..., 13]))

    return actions


def _encode_actions_inv(actions: jax.Array, *, adapt_to_pi: bool = False) -> jax.Array:
    if adapt_to_pi:
        actions = joint_flip_mask() * actions

        actions = actions.at[..., 6].set(gripper_from_angular_inv(actions[..., 6]))
        actions = actions.at[..., 13].set(gripper_from_angular_inv(actions[..., 13]))

    return actions
