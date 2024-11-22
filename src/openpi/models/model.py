import abc
from collections.abc import Sequence
import dataclasses
import logging

import augmax
from etils import epath
from flax import struct
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.base import image_tools
import openpi.base.array_typing as at
from openpi.models import common

logger = logging.getLogger("openpi")


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


def preprocess_observation(
    rng: at.KeyArrayLike,
    observation: common.Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> common.Observation:
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return common.Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )


@struct.dataclass
class BaseModel(abc.ABC):
    # Action space dimension.
    action_dim: int = struct.field(pytree_node=False)
    # Action sequence length.
    action_horizon: int = struct.field(pytree_node=False)
    # Tokenized prompt maximum length.
    max_token_len: int = struct.field(pytree_node=False)

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        actions: at.Float[at.Array, "*b ah ad"],
        *,
        train: bool = False,
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        **sample_kwargs,
    ) -> at.Float[at.Array, "*b ah ad"]: ...


@struct.dataclass
class Model(BaseModel):
    module: common.BaseModule = struct.field(pytree_node=False)
    params: at.Params | None = None

    def init_params(self, rng: at.KeyArrayLike, observation: common.Observation, actions: common.Actions) -> "Model":
        preprocess_rng, init_rng = jax.random.split(rng)
        obs = preprocess_observation(preprocess_rng, observation)

        loss_args = (obs, actions)
        return dataclasses.replace(
            self,
            params=self.module.init(init_rng, *loss_args, method=self.module.compute_loss),
        )

    @at.typecheck
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        actions: common.Actions,
        params: at.Params | None = None,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, ""]:
        if params is None:
            params = self.params

        loss_rng, preprocess_rng = jax.random.split(rng)

        obs = preprocess_observation(preprocess_rng, observation, train=train)
        loss_args = (obs, actions)

        return jnp.mean(self.module.apply(params, *loss_args, rngs={"loss": loss_rng}, method=self.module.compute_loss))

    @jax.jit
    @at.typecheck
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        **sample_kwargs,
    ) -> common.Actions:
        if self.params is None:
            raise ValueError("Model parameters not initialized.")

        preprocess_rng, sample_rng = jax.random.split(rng)

        obs = preprocess_observation(preprocess_rng, observation)
        sample_args = (self.action_horizon, self.action_dim, obs)

        actions, _ = self.module.apply(
            self.params,
            *sample_args,
            rngs={"sample": sample_rng},
            method=self.module.sample_actions,
            mutable=["cache"],
            **sample_kwargs,
        )
        return actions


def save_params(model: Model, ckpt_path: epath.Path):
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(ckpt_path, model.params)
        ckptr.wait_until_finished()


def restore_params(model: Model, ckpt_path: epath.Path, *, sharding: jax.sharding.Sharding | None = None) -> Model:
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def to_restore_args(tree):
        return jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), tree)

    with ocp.PyTreeCheckpointer() as ckptr:
        item = ckptr.metadata(ckpt_path) if model.params is None else model.params
        params = ckptr.restore(
            ckpt_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=to_restore_args(item),
            ),
        )
        return dataclasses.replace(model, params=params)


def create_inputs_spec(model: Model, *, batch_size: int = 1) -> tuple[common.Observation, at.Float[at.Array, "ah ad"]]:
    image_spec = jax.ShapeDtypeStruct([batch_size, 224, 224, 3], jnp.float32)
    image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

    observation_spec = common.Observation(
        images={
            "base_0_rgb": image_spec,
            "left_wrist_0_rgb": image_spec,
            "right_wrist_0_rgb": image_spec,
        },
        image_masks={
            "base_0_rgb": image_mask_spec,
            "left_wrist_0_rgb": image_mask_spec,
            "right_wrist_0_rgb": image_mask_spec,
        },
        state=jax.ShapeDtypeStruct([batch_size, model.action_dim], jnp.float32),
        tokenized_prompt=jax.ShapeDtypeStruct([batch_size, model.max_token_len], jnp.int32),
        tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, model.max_token_len], jnp.int32),
    )
    action_spec = jax.ShapeDtypeStruct([batch_size, model.action_horizon, model.action_dim], jnp.float32)

    return observation_spec, action_spec
