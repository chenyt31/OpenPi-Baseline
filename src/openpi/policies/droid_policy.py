from collections.abc import Sequence
import logging
import pathlib

import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import tokenizer
from openpi.policies import policy as _policy


def load_pi0_model() -> _model.Model:
    # TODO(karl): Change this to the droid model.
    model = _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)
    return _model.restore_params(model, pathlib.Path("checkpoints/pi0_base/model").absolute())


def create_droid_policy(default_prompt: str) -> _policy.Policy:
    logging.info("Loading model...")
    model = load_pi0_model()

    norm_stats = make_droid_norm_stats()
    delta_action_mask = None

    logging.info("Creating policy...")
    return _policy.Policy(
        model,
        transforms=[
            DroidInputs(
                action_dim=model.action_dim,
                delta_action_mask=delta_action_mask,
            ),
            transforms.Normalize(norm_stats),
            transforms.TokenizePrompt(
                tokenizer.PaligemmaTokenizer(model.max_token_len),
                default_prompt=default_prompt,
            ),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats),
            DroidOutputs(
                delta_action_mask=delta_action_mask,
            ),
        ],
    )


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


# def make_droid_example() -> dict:
#     return {
#         "qpos": np.ones((14,)),
#         "image": np.random.rand(4, 3, 480, 640).astype(np.float32),
#     }


def make_droid_norm_stats():
    # TODO(karl): Change these to the droid stats.
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
