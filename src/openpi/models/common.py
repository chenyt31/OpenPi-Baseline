import abc

from flax import struct
import flax.linen as nn

from openpi.base import array_typing as at


@at.typecheck
@struct.dataclass
class Observation:
    """Holds a batch of observations, which is used as input to an encoder at each timestep."""

    images: dict[str, at.Float[at.Array, "b _h _w c"]]
    image_masks: dict[str, at.Bool[at.Array, " b"]]
    state: at.Float[at.Array, "b s"]
    # This contains input mask for tokenized prompts.
    token_input_mask: at.Int[at.Array, "b lt"]
    # This contains optional pre-tokenized inputs, like instruction, state, action....
    tokenized_inputs: at.Int[at.Array, "b lt"] | None = None


class BaseModule(nn.Module, abc.ABC):
    @at.typecheck
    @abc.abstractmethod
    def compute_loss(
        self,
        obs: Observation,
        target_actions: at.Float[at.Array, "b ah ad"],
        *,
        timestep: at.Float[at.Array, " b"] | None = None,
    ) -> at.Float[at.Array, "b ah"]: ...

    @at.typecheck
    @abc.abstractmethod
    def sample_actions(
        self,
        action_horizon: int,
        action_dim: int,
        obs: Observation,
        **sample_kwargs,
    ) -> at.Float[at.Array, "b ah ad"]: ...
