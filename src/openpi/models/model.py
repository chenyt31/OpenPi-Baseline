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


def preprocess_batch(
    rng: at.KeyArrayLike,
    batch: at.Batch,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> common.Observation:
    state = batch["state"]
    images = batch["image"]

    if not set(image_keys).issubset(images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(images)}")

    batch_size = state.shape[0]

    out_images = {}
    for key in image_keys:
        image = images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)
        else:
            image = jnp.array(image)

        # Normalize to [0, 1].
        image = image.astype(state.dtype) / 255.0

        if train:
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

        # Normalize to [-1, 1].
        image = image * 2.0 - 1.0
        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        mask_name = key + "_mask"
        if mask_name not in images:
            # do not mask by default
            batch_size = jnp.shape(out_images[key])[0]
            out_masks[mask_name] = jnp.ones(batch_size, dtype=jnp.bool)
        else:
            out_masks[mask_name] = images[mask_name]

    tokenized_prompt = None
    tokenized_prompt_mask = None
    if "tokenized_prompt" in batch:
        tokenized_prompt = jnp.array(batch["tokenized_prompt"])
        tokenized_prompt_mask = jnp.array(batch["tokenized_prompt_mask"])

    return common.Observation(
        images=out_images,
        image_masks=out_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


@struct.dataclass
class Model:
    module: common.BaseModule = struct.field(pytree_node=False)
    params: at.Params | None = None
    # Action space dimension.
    action_dim: int = struct.field(default=24, pytree_node=False)
    # Action sequence length.
    action_horizon: int = struct.field(default=50, pytree_node=False)

    def init_params(self, rng: at.KeyArrayLike, batch: at.Batch) -> "Model":
        preprocess_rng, init_rng = jax.random.split(rng)
        obs = preprocess_batch(preprocess_rng, batch)

        loss_args = (obs, batch["actions"])
        return dataclasses.replace(
            self,
            params=self.module.init(init_rng, *loss_args, method=self.module.compute_loss),
        )

    @at.typecheck
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: at.Batch,
        *,
        train: bool = False,
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "b ah"]:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Model parameters not initialized.")

        loss_rng, preprocess_rng = jax.random.split(rng)

        obs = preprocess_batch(preprocess_rng, batch, train=train)
        loss_args = (obs, batch["actions"])

        return self.module.apply(params, *loss_args, rngs={"loss": loss_rng}, method=self.module.compute_loss)

    @jax.jit
    @at.typecheck
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        batch: at.Batch,
        **sample_kwargs,
    ) -> at.Float[at.Array, "b ah ad"]:
        if self.params is None:
            raise ValueError("Model parameters not initialized.")

        preprocess_rng, sample_rng = jax.random.split(rng)

        obs = preprocess_batch(preprocess_rng, batch)
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
        "base_0_rgb_mask": jax.ShapeDtypeStruct([512], jnp.bool_),
        "left_wrist_0_rgb": jax.ShapeDtypeStruct([512, 224, 224, 3], jnp.uint8),
        "left_wrist_0_rgb_mask": jax.ShapeDtypeStruct([512], jnp.bool_),
        "right_wrist_0_rgb": jax.ShapeDtypeStruct([512, 224, 224, 3], jnp.uint8),
        "right_wrist_0_rgb_mask": jax.ShapeDtypeStruct([512], jnp.bool),
    },
    "tokenized_prompt": jax.ShapeDtypeStruct([512, 48], jnp.int32),
    "tokenized_prompt_mask": jax.ShapeDtypeStruct([512, 48], jnp.int32),
    "state": jax.ShapeDtypeStruct([512, 24], jnp.float32),
}


def make_example_batch(batch_size: int):
    specs = jax.tree.map(lambda spec: jax.ShapeDtypeStruct([batch_size, *spec.shape[1:]], spec.dtype), BATCH_SPEC)
    return jax.tree.map(lambda spec: jnp.zeros(shape=spec.shape, dtype=spec.dtype), specs)
