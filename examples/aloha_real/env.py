import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import real_env as _real_env


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(self, render_height: int = 480, render_width: int = 640) -> None:
        self._env = _real_env.make_real_env(init_node=True)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def done(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        return {
            "state": _decode_state(obs["qpos"]),
            "images": {name: image_tools.convert_to_uint8(img) for name, img in obs["images"].items()},
        }

    @override
    def apply_action(self, action: dict) -> None:
        actions = _encode_actions(action["actions"])
        self._ts = self._env.step(actions)


def _decode_state(state: np.ndarray) -> np.ndarray:
    # Flip the joints.
    state = _joint_flip_mask() * state

    # Reverse the gripper transformation that is being applied by the Aloha runtime.
    state[..., 6] = _gripper_to_angular(state[..., 6])
    state[..., 13] = _gripper_to_angular(state[..., 13])

    return state


def _encode_actions(actions: np.ndarray) -> np.ndarray:
    # Flip the joints.
    actions = _joint_flip_mask() * actions

    actions[..., 6] = _gripper_from_angular(actions[..., 6])
    actions[..., 13] = _gripper_from_angular(actions[..., 13])

    return actions


def _encode_actions_inv(actions: np.ndarray) -> np.ndarray:
    """Use this for encoding actions, when training."""
    actions = _joint_flip_mask() * actions

    actions[..., 6] = _gripper_from_angular_inv(actions[..., 6])
    actions[..., 13] = _gripper_from_angular_inv(actions[..., 13])

    return actions


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)
