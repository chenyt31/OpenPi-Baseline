import pathlib

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
    return _model.restore_params(model, pathlib.Path("checkpoints/pi0_base/model").absolute())


def create_aloha_policy(
    model: _model.BaseModel, norm_stats: dict[str, transforms.NormStats], *, default_prompt: str | None = None
) -> _policy.Policy:
    reorder_dims = False
    delta_actions = True

    return _policy.Policy(
        model,
        transforms=[
            AlohaInputs(action_dim=model.action_dim, reorder_dims=reorder_dims, delta_actions=delta_actions),
            transforms.Normalize(norm_stats),
            transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(model.max_token_len), default_prompt=default_prompt),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats),
            AlohaOutputs(reorder_dims=reorder_dims, delta_actions=delta_actions),
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


class AlohaInputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int, *, reorder_dims: bool = False, delta_actions: bool = False):
        self._action_dim = action_dim
        self._reorder_dims = reorder_dims
        self._delta_actions = delta_actions
        # Range closed, open
        self.left_gripper_range = (0.48, 1.22)
        self.right_gripper_range = (0.23, 1.36)

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, reorder_dims=self._reorder_dims)

        # Assume that base image always exists.
        base_image = data["cam_high"]
        batch_size = base_image.shape[:-3]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        }

        # Add the extra images.
        extra_images = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_images.items():
            if source in data:
                images[dest] = data[source]
                image_masks[dest] = jnp.ones(batch_size, dtype=jnp.bool_)
            else:
                images[dest] = jnp.zeros_like(base_image)
                image_masks[dest] = jnp.zeros(batch_size, dtype=jnp.bool_)

        # Update the signs to match monopi
        data["state"] = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * data["state"]

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": transforms.pad_to_dim(data["state"], self._action_dim),
        }

        # Actions are only available during training.
        if "action/qpos" in data:
            # TODO(ury): We need to convert this to delta actions. Make sure that this is the
            # case when we do training.
            inputs["actions"] = transforms.pad_to_dim(data["action/qpos"], self._action_dim)

        return inputs


class AlohaOutputs(transforms.DataTransformFn):
    def __init__(self, *, reorder_dims: bool = False, delta_actions: bool = False):
        self._reorder_dims = reorder_dims
        self._delta_actions = delta_actions
        self.left_gripper_range = (0.48, 1.22)
        self.right_gripper_range = (0.23, 1.36)

    def __call__(self, data: dict) -> dict:
        if self._delta_actions:
            # Convert from delta to absolute actions.
            actions = (
                data["actions"]
                .at[..., :6]
                .set(jnp.expand_dims(data["state"], axis=-2)[..., :6] + data["actions"][..., :6])
            )
            actions = actions.at[..., 7:13].set(
                jnp.expand_dims(data["state"], axis=-2)[..., 7:13] + data["actions"][..., 7:13]
            )
        else:
            actions = data["actions"]

        # Update the signs to match monopi
        actions = actions.at[..., :14].set(
            jnp.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]) * actions[..., :14]
        )

        # Only return the first 14 dims.
        return {"qpos": _encode_aloha(actions[..., :14], reorder_dims=self._reorder_dims)}


# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
    PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
)


HORN_RADIUS = 0.022
ARM_LENGTH = 0.036

# def convert_linear_position_to_radian(linear_position, arm_length, horn_radius):
#     half_dist = linear_position / 2.0
#     result = jnp.pi / 2.0 - jnp.arccos(
#         (horn_radius ** 2 + half_dist ** 2 - arm_length ** 2) / (2 * horn_radius * half_dist)
#     )
#     return result


def convert_linear_position_to_radian(linear_position, arm_length, horn_radius):
    return jnp.arcsin((-(arm_length**2) + horn_radius**2 + linear_position**2) / (2 * horn_radius * linear_position))


def convert_angular_position_to_linear(angular_position, arm_length, horn_radius):
    a1 = horn_radius * jnp.sin(angular_position)
    c = jnp.sqrt(horn_radius**2 - a1**2)
    a2 = jnp.sqrt(arm_length**2 - c**2)
    return a1 + a2


MAX_GRIPPER_JOINT_POS = convert_linear_position_to_radian(PUPPET_GRIPPER_POSITION_OPEN, ARM_LENGTH, HORN_RADIUS)
MIN_GRIPPER_JOINT_POS = convert_linear_position_to_radian(PUPPET_GRIPPER_POSITION_CLOSE, ARM_LENGTH, HORN_RADIUS)


def _decode_aloha(data: dict, *, reorder_dims: bool = False) -> dict:
    # qpos is [..., 14] of type float:
    # 0-5: left arm joint angles
    # 6: left arm gripper
    # 7-12: right arm joint angles
    # 13: right arm gripper
    qpos = jnp.asarray(data["qpos"])

    # unnormalize the gripper position applied by aloha
    # qpos = qpos.at[..., 6].set(PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(qpos[..., 6]))
    # qpos = qpos.at[..., 13].set(PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(qpos[..., 13]))

    # Convert gripper from linear to angular
    # qpos = qpos.at[..., 6].set(convert_linear_position_to_radian(qpos[..., 6], ARM_LENGTH, HORN_RADIUS))
    # qpos = qpos.at[..., 13].set(convert_linear_position_to_radian(qpos[..., 13], ARM_LENGTH, HORN_RADIUS))

    # print("qpos after inversion", qpos[..., 6], qpos[..., 13])

    # Get the observer's gripper position in the range [-1, 1]
    max_gripper_qpos = 1.5
    min_gripper_qpos = 0.4
    # print("MAX/MIN GRIPPER JOINT POS", MAX_GRIPPER_JOINT_POS, MIN_GRIPPER_JOINT_POS)
    gripper_qpos_normalize_fn = lambda x: (x - min_gripper_qpos) / (max_gripper_qpos - min_gripper_qpos)
    qpos = qpos.at[..., 6].set(gripper_qpos_normalize_fn(qpos[..., 6]))
    qpos = qpos.at[..., 13].set(gripper_qpos_normalize_fn(qpos[..., 13]))

    # Convert to [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    state = qpos[..., jnp.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])] if reorder_dims else qpos

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
    # Convert from [..., channel, height, width] to [..., height, width, channel].
    images = jnp.rollaxis(images, -3, len(images.shape))
    # Convert to uint8 if using float images.
    if np.issubdtype(images.dtype, jnp.floating):
        images = (255 * images).astype(jnp.uint8)
    # Split into a dict with keys as camera names.
    image_splits = [jnp.squeeze(x, axis=-4) for x in jnp.split(images, num_cams, axis=-4)]
    images_dict = dict(zip(cam_names, image_splits, strict=True))

    return {**images_dict, "state": state}


def _encode_aloha(actions: jax.Array, *, reorder_dims: bool = False) -> jax.Array:
    # Convert to [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # TODO (michael): The following code assumes that the gripper position is in the range [0, 1]
    # out model produces outputs that look more like 0-1.05 so we should normalize to account for this.
    # We should verify if this is actually necessary.
    model_output_max = 1.05
    model_output_min = 0.0
    gripper_qpos_normalize_fn = lambda x: (x - model_output_min) / (model_output_max - model_output_min)
    actions = actions.at[..., 6].set(gripper_qpos_normalize_fn(actions[..., 6]))
    actions = actions.at[..., 13].set(gripper_qpos_normalize_fn(actions[..., 13]))

    # Denormalize the gripper position applied by the observer
    # TODO (michael): THESE ARE MIN/MAX QPOS of the station. There may be some different station to station so we need to make these more easily configurable
    # max_gripper_pos = 1.5
    # min_gripper_pos = 0.4
    # gripper_qpos_unnormalize_fn = lambda x: x * (max_gripper_pos - min_gripper_pos) + min_gripper_pos
    # actions = actions.at[..., 6].set(gripper_qpos_unnormalize_fn(actions[..., 6]))
    # actions = actions.at[..., 13].set(gripper_qpos_unnormalize_fn(actions[..., 13]))

    # normalize the gripper joint applied by aloha
    # actions = actions.at[..., 6].set(PUPPET_GRIPPER_JOINT_NORMALIZE_FN(actions[..., 6]))
    # actions = actions.at[..., 13].set(PUPPET_GRIPPER_JOINT_NORMALIZE_FN(actions[..., 13]))

    return actions[..., jnp.array([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])] if reorder_dims else actions
