from collections.abc import Sequence
import dataclasses
from typing import Any

from openpi.models import tokenizer
import openpi.models.model as _model
import openpi.policies.policy as _policy
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
