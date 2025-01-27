import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b
