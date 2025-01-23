from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

from openpi.models import tokenizer
import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None
    fast_tokenizer: str | None = None


def create_policy(config: PolicyConfig) -> _policy.Policy:
    """Create a policy from a policy config."""
    match config.model_type:
        case _model.ModelType.PI0:
            return _create_pi0_policy(config)
        case _model.ModelType.PI0_FAST:
            return _create_pi0_fast_policy(config)
        case _:
            raise ValueError(f"Unsupported model type: {config.model_type}")


def _create_pi0_policy(config: PolicyConfig) -> _policy.Policy:
    sample_kwargs = config.sample_kwargs or {"num_steps": 10}
    return _policy.Policy(
        config.model,
        transforms=[
            *config.input_layers,
            transforms.Normalize(config.norm_stats),
            transforms.TokenizePrompt(
                tokenizer.PaligemmaTokenizer(config.model.max_token_len),
                default_prompt=config.default_prompt,
            ),
        ],
        output_transforms=[
            transforms.Unnormalize(config.norm_stats),
            *config.output_layers,
        ],
        sample_kwargs=sample_kwargs,
    )


def _create_pi0_fast_policy(config: PolicyConfig) -> _policy.Policy:
    """Creates a pi0 FAST policy."""
    tokenizer_path = config.fast_tokenizer or "physical-intelligence/fast"
    sample_kwargs = config.sample_kwargs or {}
    return _policy.Policy(
        config.model,
        transforms=[
            *config.input_layers,
            transforms.Normalize(config.norm_stats, use_quantiles=True),
            transforms.TokenizeFASTInputs(
                tokenizer.FASTTokenizer(config.model.max_token_len, tokenizer_path),
                default_prompt=config.default_prompt,
            ),
        ],
        output_transforms=[
            transforms.ExtractFASTActions(
                tokenizer.FASTTokenizer(config.model.max_token_len, tokenizer_path),
                action_horizon=config.model.action_horizon,
                action_dim=config.model.action_dim,
            ),
            transforms.Unnormalize(config.norm_stats, use_quantiles=True),
            *config.output_layers,
        ],
        sample_kwargs=sample_kwargs,
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model = train_config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16), allow_extra_params=True
    )

    data_config = train_config.data.create(train_config.metadata_dir, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint, instead of the metadata dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets")

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
