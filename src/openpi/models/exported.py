from typing import Any

import etils.epath as epath
import flax.serialization
import flax.struct as struct
import jax
import jax.experimental.export as export
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.base import image_tools
from openpi.base import normalize as _normalize
import openpi.base.array_typing as at
from openpi.models import common
from openpi.models import model as _model

# TODO(ury): Remove before open sourcing and consider replacing with an official export API.
# TODO: Upgrade to official export API.


@struct.dataclass
class PiModel(_model.BaseModel):
    """A model loaded from a monopi checkpoint model directory."""

    params: at.Params

    exported: export.Exported = struct.field(pytree_node=False)
    example_spec: Any = struct.field(pytree_node=False)

    @classmethod
    def from_checkpoint(cls, ckpt_path: epath.Path) -> "PiModel":
        """Load a model from a monopi checkpoint model directory."""
        with (ckpt_path / "graph").open("rb") as f:
            exported = export.deserialize(f.read())

        input_spec = jax.tree.unflatten(exported.in_tree, exported.in_avals)[0]
        params = _load_params(ckpt_path, input_spec[0])
        example_spec = input_spec[2]

        # Extract the action properties from the output spec.
        output_spec = jax.tree.unflatten(exported.out_tree, exported.out_avals)
        actions_spec = output_spec["actions"]
        action_horizon, action_dim = actions_spec.shape

        return cls(
            params=params,
            exported=exported,
            example_spec=example_spec,
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_token_len=48,
        )

    @jax.jit
    @override
    def sample_actions(self, rng: at.KeyArrayLike, obs: common.Observation) -> at.Float[at.Array, "b ah ad"]:
        if obs.state.ndim == 2 and obs.state.shape[0] != 1:
            raise ValueError("Only batch_size=1 is supported.")

        # Convert to the example format.
        example = _obs_to_example(obs)
        example = _unbatch(example)

        # Resize the input images if needed.
        def resize_if_needed(key, image):
            target_shape = self.example_spec["image"][key].shape
            if len(target_shape) == 3 and image.shape != target_shape:
                return image_tools.resize_with_pad(image, *target_shape[-3:-1])
            return image

        example["image"] = {key: resize_if_needed(key, value) for key, value in example["image"].items()}

        rng_data = jax.random.key_data(rng)
        result = export.call(self.exported)(self.params, rng_data, example, {"num_steps": 10})

        return _make_batch(result)["actions"]

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        actions: at.Float[at.Array, "*b ah ad"],
        *,
        train: bool = False,
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        raise NotImplementedError("Not implemented.")

    def fake_obs(self) -> common.Observation:
        example = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), self.example_spec)
        return _example_to_obs(_make_batch(example))


def _load_params(path: epath.Path, params_spec: at.PyTree, sharding: jax.sharding.Sharding | None = None):
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def to_restore_args(tree):
        return jax.tree.map(lambda x: ocp.ArrayRestoreArgs(dtype=x.dtype, sharding=sharding), tree)

    with ocp.PyTreeCheckpointer() as ckptr:
        item = {"params": params_spec}
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=item,
                restore_args=to_restore_args(item),
                # This is needed to read a partial checkpoint.
                transforms={},
            ),
        )["params"]


def _obs_to_example(obs: common.Observation) -> dict:
    def to_uint8(v):
        return (255.0 * (v + 1.0) / 2.0).astype(jnp.uint8)

    images = {k: to_uint8(v) for k, v in obs.images.items()}
    image_masks = {f"{k}_mask": v for k, v in obs.image_masks.items()}

    return {
        "image": {**images, **image_masks},
        "state": obs.state,
        "prompt_tokens": obs.tokenized_prompt,
        "mask_input": obs.tokenized_prompt_mask,
    }


def _example_to_obs(example: dict) -> common.Observation:
    images, image_masks = {}, {}
    for k, v in example["image"].items():
        if k.endswith("_mask"):
            image_masks[k.removesuffix("_mask")] = v
        else:
            images[k] = v

    return common.Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": example["state"],
            "tokenized_prompt": example["prompt_tokens"],
            "tokenized_prompt_mask": example["mask_input"],
        }
    )


def import_norm_stats(ckpt_path: epath.Path, processor_name: str) -> dict[str, _normalize.NormStats]:
    path = ckpt_path / "processors" / processor_name
    if not (found_files := list(path.glob("*/norm_stats.msgpack"))):
        raise FileNotFoundError(f"No norm_stats.msgpack found in {path}")

    with found_files[0].open("rb") as f:
        norm_stats = flax.serialization.msgpack_restore(f.read())

    input_state = norm_stats["input_norms"]["state"]
    state_stats = _normalize.NormStats(mean=input_state["mean"], std=input_state["std"])

    output_actions = norm_stats["output_norms"]["actions"]
    actions_stats = _normalize.NormStats(mean=output_actions["mean"], std=output_actions["std"])

    return {
        "state": state_stats,
        "actions": actions_stats,
    }


def _make_batch(data: dict) -> dict:
    return jax.tree.map(lambda x: x[jnp.newaxis, ...], data)


def _unbatch(data: dict) -> dict:
    return jax.tree.map(lambda x: x[0, ...], data)
