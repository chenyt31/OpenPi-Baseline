from collections.abc import Sequence
import dataclasses
import logging
from etils import epath

import augmax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.base import image_tools
import openpi.base.array_typing as at
from openpi.models import common
from openpi.models import tokenizer

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
    tokenizer: tokenizer.Tokenizer,
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

    if (prompt := batch.get("prompt")) is None:
        prompt = np.array(["be a good robot"] * batch_size)
    tokens, token_masks = tokenizer.tokenize(prompt)

    return common.Observation(
        images=out_images,
        image_masks=out_masks,
        state=state,
        tokenized_inputs=jnp.array(tokens),
        token_input_mask=jnp.array(token_masks),
    )


@dataclasses.dataclass(frozen=True)
class Model:
    module: common.BaseModule
    rng: at.KeyArrayLike
    tokenizer: tokenizer.Tokenizer
    params: at.Params | None = None

    # Action space dimension.
    action_dim: int = 24
    # Action sequence length.
    action_horizon: int = 50

    def init_params(self, batch: at.Batch) -> "Model":
        preprocess_rng, init_rng = jax.random.split(self.rng)
        obs = preprocess_batch(preprocess_rng, batch, self.tokenizer)

        loss_args = (obs, batch["actions"])
        return dataclasses.replace(
            self,
            params=self.module.init(init_rng, *loss_args, method=self.module.compute_loss),
        )

    @at.typecheck
    def compute_loss(
        self,
        batch: at.Batch,
        *,
        train: bool = False,
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "b ah"]:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Model parameters not initialized.")

        loss_rng, preprocess_rng = jax.random.split(self.rng)

        obs = preprocess_batch(preprocess_rng, batch, self.tokenizer, train=train)
        loss_args = (obs, batch["actions"])

        return self.module.apply(params, *loss_args, rngs={"loss": loss_rng}, method=self.module.compute_loss)

    @at.typecheck
    def sample_actions(
        self,
        batch: at.Batch,
        **sample_kwargs,
    ) -> at.Float[at.Array, "b ah ad"]:
        if self.params is None:
            raise ValueError("Model parameters not initialized.")

        preprocess_rng, sample_rng = jax.random.split(self.rng)

        obs = preprocess_batch(preprocess_rng, batch, self.tokenizer)
        sample_args = (self.action_horizon, self.action_dim, obs)

        sample, _ = self.module.apply(
            self.params,
            *sample_args,
            rngs={"sample": sample_rng},
            method=self.module.sample_actions,
            mutable=["cache"],
            **sample_kwargs,
        )

        return sample


def save_params(model: Model, ckpt_path: epath.Path):
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(ckpt_path, model.params)
        ckptr.wait_until_finished()


def restore_params(model: Model, ckpt_path: epath.Path) -> Model:
    with ocp.StandardCheckpointer() as ckptr:
        params = ckptr.restore(ckpt_path)
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
    "mask_input": jax.ShapeDtypeStruct([512, 48], jnp.int32),
    "prompt_tokens": jax.ShapeDtypeStruct([512, 48], jnp.int32),
    "state": jax.ShapeDtypeStruct([512, 24], jnp.float32),
}


def make_example_batch(batch_size: int):
    specs = jax.tree.map(lambda spec: jax.ShapeDtypeStruct([batch_size, *spec.shape[1:]], spec.dtype), BATCH_SPEC)
    return jax.tree.map(lambda spec: jnp.zeros(shape=spec.shape, dtype=spec.dtype), specs)
