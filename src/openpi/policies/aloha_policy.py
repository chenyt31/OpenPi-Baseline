import pathlib

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import tokenizer
from openpi.policies import policy as _policy


def load_pi0_model() -> _model.Model:
    model = _model.Model(pi0.Module())
    return _model.restore_params(model, pathlib.Path("checkpoints/pi0_base/model").absolute())


def create_aloha_policy(model: _model.Model) -> _policy.Policy:
    norm_stats = make_aloha_norm_stats()
    return _policy.Policy(
        model,
        transforms=[
            transforms.AlohaInputs(action_dim=model.action_dim),
            transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(model.max_token_len)),
            transforms.Normalize(norm_stats),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats),
            transforms.AlohaOutputs(),
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
                    -0.0,
                    0.01,
                    0.02,
                    0.0,
                    -0.01,
                    0.0,
                    0.44,
                    -0.0,
                    0.02,
                    0.02,
                    0.0,
                    -0.01,
                    0.0,
                    0.43,
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
                    0.11,
                    0.17,
                    0.16,
                    0.11,
                    0.17,
                    0.15,
                    0.41,
                    0.12,
                    0.19,
                    0.18,
                    0.14,
                    0.19,
                    0.19,
                    0.39,
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
                    0.11,
                    0.45,
                    -0.6,
                    0.13,
                    0.5,
                    -0.2,
                    0.45,
                    -0.07,
                    0.31,
                    -0.48,
                    -0.13,
                    0.55,
                    0.21,
                    0.44,
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
                    0.3,
                    0.54,
                    0.55,
                    0.28,
                    0.47,
                    0.44,
                    0.38,
                    0.32,
                    0.57,
                    0.56,
                    0.3,
                    0.49,
                    0.54,
                    0.36,
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
