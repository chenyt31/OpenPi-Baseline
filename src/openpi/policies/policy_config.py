from collections.abc import Sequence
import dataclasses
import logging
from typing import Any

from openpi.models import tokenizer
import openpi.models.model as _model
import openpi.policies.policy as _policy
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    model: _model.Model

    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def create_policy(config: PolicyConfig) -> _policy.Policy:
    """Creates a default pi0 policy."""
    return _policy.Policy(
        config.model,
        transforms=[
            *config.input_layers,
            transforms.Normalize(config.norm_stats),
            transforms.TokenizePrompt(
                tokenizer.PaligemmaTokenizer(config.model.max_token_len), default_prompt=config.default_prompt
            ),
        ],
        output_transforms=[
            transforms.Unnormalize(config.norm_stats),
            *config.output_layers,
        ],
        sample_kwargs=config.sample_kwargs,
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_path: str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained model."""
    repack_transforms = repack_transforms or transforms.Group()

    logging.info("Loading model...")
    model = train_config.create_model()
    model = model.set_params(_model.restore_params(checkpoint_path))

    data_config = train_config.data.create(train_config.metadata_dir, model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_path)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
    )
