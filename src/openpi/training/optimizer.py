from typing import Protocol

import jax
import optax

import openpi.base.array_typing as at


class ScheduleFactory(Protocol):
    def __call__(self) -> optax.Schedule: ...


class OptimizerFactory(Protocol):
    def __call__(self, lr: optax.ScalarOrSchedule, **kwargs) -> optax.GradientTransformation: ...


def cosine_decay_schedule(
    warmup_steps: int = 1_000, peak_lr: float = 5e-5, decay_steps: int = int(1e9), decay_lr: float = 5e-5
) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=peak_lr / (warmup_steps + 1),
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=decay_lr,
    )


def adamw(
    lr: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    clip_gradient_norm: float | None = 100.0,
    ema_decay: float = 0.999,
    weight_decay: float = 1e-4,
    weight_decay_mask: at.PyTree | None = None,
    freeze_mask: at.PyTree | None = None,
) -> optax.GradientTransformation:
    tx = optax.adamw(lr, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay, mask=weight_decay_mask)

    if freeze_mask is not None:
        tx = optax.multi_transform(
            {"online": tx, "offline": optax.set_to_zero()},
            jax.tree_util.tree_map(lambda x: "offline" if x else "online", freeze_mask),
        )

    if clip_gradient_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(clip_gradient_norm), tx)

    if ema_decay != 0:
        tx = optax.chain(optax.ema(ema_decay), tx)

    return tx
