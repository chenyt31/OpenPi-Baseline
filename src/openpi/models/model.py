from collections.abc import Sequence
import dataclasses
import logging

import augmax
from etils import epath
from flax import struct
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

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
            out_masks[key] = observation.image_masks[key]

    return common.Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )


@struct.dataclass
class Model:
    module: common.BaseModule = struct.field(pytree_node=False)
    params: at.Params | None = None
    # Action space dimension.
    action_dim: int = struct.field(default=24, pytree_node=False)
    # Action sequence length.
    action_horizon: int = struct.field(default=50, pytree_node=False)

    def init_params(
        self, rng: at.KeyArrayLike, observation: common.Observation, actions: at.Float[at.Array, "*b ah ad"]
    ) -> "Model":
        preprocess_rng, init_rng = jax.random.split(rng)
        obs = preprocess_observation(preprocess_rng, observation)

        loss_args = (obs, actions)
        return dataclasses.replace(
            self,
            params=self.module.init(init_rng, *loss_args, method=self.module.compute_loss),
        )

    @at.typecheck
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        actions: at.Float[at.Array, "*b ah ad"],
        *,
        train: bool = False,
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        if params is None:
            params = self.params

        loss_rng, preprocess_rng = jax.random.split(rng)

        obs = preprocess_observation(preprocess_rng, observation, train=train)
        loss_args = (obs, actions)

        return self.module.apply(params, *loss_args, rngs={"loss": loss_rng}, method=self.module.compute_loss)

    @jax.jit
    @at.typecheck
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        **sample_kwargs,
    ) -> at.Float[at.Array, "*b ah ad"]:
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


# TODO(ury): This is all temporary. We should get it from the checkpoint or the model config instead.

BATCH_SPEC = {
    "actions": jax.ShapeDtypeStruct([512, 50, 24], jnp.float32),
    "image": {
        "base_0_rgb": jax.ShapeDtypeStruct([512, 224, 224, 3], jnp.uint8),
        "left_wrist_0_rgb": jax.ShapeDtypeStruct([512, 224, 224, 3], jnp.uint8),
        "right_wrist_0_rgb": jax.ShapeDtypeStruct([512, 224, 224, 3], jnp.uint8),
    },
    "image_mask": {
        "base_0_rgb": jax.ShapeDtypeStruct([512], jnp.bool_),
        "left_wrist_0_rgb": jax.ShapeDtypeStruct([512], jnp.bool_),
        "right_wrist_0_rgb": jax.ShapeDtypeStruct([512], jnp.bool),
    },
    "tokenized_prompt": jax.ShapeDtypeStruct([512, 48], jnp.int32),
    "tokenized_prompt_mask": jax.ShapeDtypeStruct([512, 48], jnp.int32),
    "state": jax.ShapeDtypeStruct([512, 24], jnp.float32),
}


def make_example_batch(batch_size: int):
    specs = jax.tree.map(lambda spec: jax.ShapeDtypeStruct([batch_size, *spec.shape[1:]], spec.dtype), BATCH_SPEC)
    return jax.tree.map(lambda spec: jnp.zeros(shape=spec.shape, dtype=spec.dtype), specs)
