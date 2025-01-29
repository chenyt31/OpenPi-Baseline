"""Functionality to handle internal pi checkpoints.

Used to test internal pi checkpoints and provides utilities to convert them to openpi checkpoints.

TODO(ury): Remove before the public release.
"""

from collections.abc import Mapping
import dataclasses
import pathlib

import flax.nnx as nnx
import flax.serialization
import jax
import jax.export
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override
import yaml

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.shared.array_typing as at
import openpi.shared.download as download
import openpi.shared.image_tools as image_tools
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms


@dataclasses.dataclass
class ConvertConfig:
    # The path to the parameters within the overall param structure. Can include "/" to support nesting.
    param_path: str = ""

    # Transform patterns to use when converting the checkpoint params. Each key maps from the
    # original param name to the openpi param name. If the value is None, the param is removed.
    transform: Mapping[str, str | None] | None = None


def convert_to_openpi(
    ckpt_dir: pathlib.Path | str,
    out_dir: pathlib.Path | str,
    *,
    processor: str | None = None,
    config: ConvertConfig | None = None,
) -> None:
    """Convert an internal checkpoint to an openpi checkpoint.

    Args:
        ckpt_dir: The directory containing the internal exported model.
        out_dir: The directory to save the openpi checkpoint.
        processor: The processor name to use to extract the norm stats. If None, the first processor
            in the checkpoint is used if there's only one available.
        config: The configuration to use when converting the checkpoint params. If not provided, the right
            configuration will be inferred from the checkpoint.
    """
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load params and norm stats.
    ckpt_dir = download.maybe_download(str(ckpt_dir))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    params = _load_params(ckpt_dir, sharding=sharding)

    if not config:
        config = _detect_convert_config(params)

    model = PiModel(ckpt_dir, params=params)
    print("Processors:    ", model.processor_names())
    print("Action dim:    ", model.action_dim)
    print("Action horizon:", model.action_horizon)
    print("Max token len: ", model.max_token_len)

    if processor is None:
        processor_names = model.processor_names()
        if len(processor_names) != 1:
            raise ValueError(
                f"Multiple processors found in the checkpoint. Please specify a processor name: {processor_names}"
            )
        processor = processor_names[0]

    norm_stats = _import_norm_stats(ckpt_dir, processor)

    if config.param_path:
        for part in config.param_path.split("/"):
            if part not in params:
                raise ValueError(f"{part} not found in the checkpoint. Available keys: {list(params)}")
            params = params[part]

    if config.transform is not None:
        params = _transforms.transform_dict(config.transform, params)

    # Make sure that the loaded params are compatible with the openpi model.
    openpi_model = nnx.eval_shape(model.openpi_model_config().create, jax.random.key(0))
    openpi_params = nnx.state(openpi_model).to_pure_dict()

    # Remove any extra params that are not part of the openpi model.
    params = ocp.transform_utils.intersect_trees(openpi_params, params)
    # Make sure that all expected params are present.
    at.check_pytree_equality(expected=openpi_params, got=params, check_shapes=True, check_dtypes=False)

    # Save params.
    ckpt = ocp.StandardCheckpointer()
    ckpt.save(out_dir / "params", {"params": params})
    ckpt.wait_until_finished()

    # Save norm stats.
    _normalize.save(out_dir / "assets", norm_stats)


class PiModel(_model.BaseModel):
    """A model loaded from an internal exported model directory."""

    def __init__(self, ckpt_dir: pathlib.Path | str, params: at.Params | None = None):
        """Load a model from the internal checkpoint directory. Must point at the "model" sub-directory."""
        self.ckpt_dir = download.maybe_download(str(ckpt_dir))
        with (self.ckpt_dir / "graph").open("rb") as f:
            self._exported = jax.export.deserialize(bytearray(f.read()))

        input_spec = jax.tree.unflatten(self._exported.in_tree, self._exported.in_avals)[0]
        if params is None:
            params = _load_params(self.ckpt_dir, input_spec[0])
        self.example_spec = input_spec[2]
        self.sample_spec = input_spec[3]

        # Extract the action properties from the output spec.
        output_spec = jax.tree.unflatten(self._exported.out_tree, self._exported.out_avals)
        if "actions" in output_spec:
            # This is a pi0 model.
            self._output_key = "actions"
            self._model_type = _model.ModelType.PI0
            action_horizon, action_dim = output_spec[self._output_key].shape
        elif "output_tokens" in output_spec:
            # This is a pi0-fast model.
            self._output_key = "output_tokens"
            self._model_type = _model.ModelType.PI0_FAST
            # We can't determine the action shape from the output since the model is returning
            # a tokenized action. Look for this information inside one of the processor specs instead.
            # It doesn't matter which processor is used since all of them should use the same value.
            processor_name = self.processor_names()[0]
            unprocess_spec = _load_processor_msgpack(
                self.ckpt_dir / "processors" / processor_name / "unprocess_spec.msgpack"
            )
            action_shape = unprocess_spec["outputs"]["actions"][0]
            action_horizon, action_dim = action_shape
        else:
            raise ValueError(f"Unknown output spec: {output_spec}")

        max_token_len = self.example_spec["prompt_tokens"].shape[-1]

        super().__init__(action_dim=action_dim, action_horizon=action_horizon, max_token_len=max_token_len)

        # wrap arrays in nnx.Param to make nnx happy
        self.params = jax.tree.map(lambda x: nnx.Param(x), params)

    @property
    def model_type(self) -> _model.ModelType:
        return self._model_type

    def openpi_model_config(self) -> _model.BaseModelConfig:
        match self.model_type:
            case _model.ModelType.PI0:
                return pi0.Pi0Config(
                    action_dim=self.action_dim, action_horizon=self.action_horizon, max_token_len=self.max_token_len
                )
            case _model.ModelType.PI0_FAST:
                return pi0_fast.Pi0FASTConfig(
                    action_dim=self.action_dim, action_horizon=self.action_horizon, max_token_len=self.max_token_len
                )
            case _:
                raise ValueError(f"Unknown model type: {self.model_type}")

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **sample_kwargs) -> _model.Actions:
        if observation.state.ndim == 2 and observation.state.shape[0] != 1:
            raise ValueError("Only batch_size=1 is supported.")

        # Convert to the example format.
        example = _obs_to_example(observation, self.example_spec)
        example = _unbatch(example)

        # Resize the input images if needed.
        def resize_if_needed(key, image):
            target_shape = self.example_spec["image"][key].shape
            if len(target_shape) == 3 and image.shape != target_shape:
                return image_tools.resize_with_pad(image, *target_shape[-3:-1])
            return image

        example["image"] = {key: resize_if_needed(key, value) for key, value in example["image"].items()}

        # Default sample kwargs values for exported models. Will be used if not provided by the user.
        default_kwargs = {"num_steps": 10}

        for key in self.sample_spec:
            if key not in sample_kwargs:
                sample_kwargs[key] = default_kwargs[key]

        if set(sample_kwargs) != set(self.sample_spec):
            raise ValueError(
                f"Sample args {list(sample_kwargs)} do not match the expected args {list(self.sample_spec)}"
            )

        rng_data = jax.random.key_data(rng)
        result = self._exported.call(jax.tree.map(lambda x: x.value, self.params), rng_data, example, sample_kwargs)

        return _make_batch(result)[self._output_key]

    @override
    def compute_loss(self):
        raise NotImplementedError("Not implemented.")

    def fake_obs(self) -> _model.Observation:
        example = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), self.example_spec)
        return _example_to_obs(_make_batch(example))

    def norm_stats(self, processor_name: str) -> dict[str, _normalize.NormStats]:
        """Load the norm stats from the checkpoint."""
        return _import_norm_stats(self.ckpt_dir, processor_name)

    def processor_names(self) -> list[str]:
        """List of processor names available in the checkpoint."""
        processor_dir = self.ckpt_dir / "processors"
        return [x.name for x in processor_dir.iterdir() if x.is_dir()]


def _load_params(
    path: pathlib.Path, params_spec: at.PyTree | None = None, sharding: jax.sharding.Sharding | None = None
):
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def to_restore_args(tree):
        return jax.tree.map(lambda x: ocp.ArrayRestoreArgs(dtype=x.dtype, sharding=sharding), tree)

    with ocp.PyTreeCheckpointer() as ckptr:
        if params_spec is None:
            params_spec = ckptr.metadata(path)["params"]
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


def _obs_to_example(obs: _model.Observation, example_spec: dict) -> dict:
    def to_uint8(v):
        return (255.0 * (v + 1.0) / 2.0).astype(jnp.uint8)

    images = {k: to_uint8(v) for k, v in obs.images.items()}
    image_masks = {f"{k}_mask": v for k, v in obs.image_masks.items()}

    result = {
        "image": {**images, **image_masks},
        "state": obs.state,
        "prompt_tokens": obs.tokenized_prompt,
    }

    if obs.token_ar_mask is not None:
        # This is a pi0-fast model.
        assert obs.tokenized_prompt_mask is not None
        result = {
            **result,
            "mask_prompt_input": obs.tokenized_prompt_mask.astype(jnp.int32),
            "mask_ar": obs.token_ar_mask,
            "tokens": obs.tokenized_prompt,
            "mask_input": obs.tokenized_prompt_mask.astype(jnp.int32),
        }
    else:
        # This is a pi0 model.
        assert obs.tokenized_prompt_mask is not None
        result = {
            **result,
            "mask_input": obs.tokenized_prompt_mask.astype(jnp.int32),
        }

    return result


def _example_to_obs(example: dict) -> _model.Observation:
    images, image_masks = {}, {}
    for k, v in example["image"].items():
        if k.endswith("_mask"):
            image_masks[k.removesuffix("_mask")] = v
        else:
            images[k] = v

    # NOTE(ury): This is used to support the new version with DCT co-training.
    if "mask_prompt_input" in example:
        example["mask_input"] = example["mask_prompt_input"]

    return _model.Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": example["state"],
            "tokenized_prompt": example["prompt_tokens"],
            "tokenized_prompt_mask": example["mask_input"].astype(bool),
            "token_ar_mask": example.get("mask_ar"),
            "token_loss_mask": example["mask_loss"].astype(bool) if "mask_loss" in example else None,
        }
    )


def _import_norm_stats(ckpt_dir: pathlib.Path | str, processor_name: str) -> dict[str, _normalize.NormStats]:
    ckpt_dir = pathlib.Path(ckpt_dir).resolve()

    path = ckpt_dir / "processors" / processor_name
    if not path.exists():
        raise FileNotFoundError(f"Processor {processor_name} not found in {ckpt_dir}")

    if not (found_files := list(path.glob("*/norm_stats.msgpack"))):
        raise FileNotFoundError(f"norm_stats.msgpack not found in {path}")

    outputs = []

    for file in sorted(found_files):
        with file.open("rb") as f:
            norm_stats = flax.serialization.msgpack_restore(f.read())

        # This is the new Normalize processor.
        if "input_norms" in norm_stats:
            actions = norm_stats["output_norms"]["actions"]
            outputs.append(
                _normalize.NormStats(
                    mean=actions["mean"],
                    std=actions["std"],
                    q01=actions.get("q01"),
                    q99=actions.get("q99"),
                )
            )

            state = norm_stats["input_norms"]["state"]
            outputs.append(
                _normalize.NormStats(
                    mean=state["mean"],
                    std=state["std"],
                    q01=state.get("q01"),
                    q99=state.get("q99"),
                )
            )

        # This is to support the old NormalizeActions / NormalizeState processor combo.
        else:
            outputs.append(
                _normalize.NormStats(
                    mean=norm_stats["mean"],
                    std=norm_stats["std"],
                    q01=norm_stats.get("q01"),
                    q99=norm_stats.get("q99"),
                )
            )

    return {
        "actions": outputs[0],
        "state": outputs[1],
    }


def _make_batch(data: at.PyTree) -> at.PyTree:
    return jax.tree.map(lambda x: x[jnp.newaxis, ...], data)


def _unbatch(data: at.PyTree) -> at.PyTree:
    return jax.tree.map(lambda x: x[0, ...], data)


def _detect_convert_config(params: at.PyTree) -> ConvertConfig:
    if "decoder" in params:
        return ConvertConfig(param_path="decoder")

    if "paligemma_model" in params:
        return ConvertConfig(
            transform={
                # Replace the paligemma_model prefix with PaliGemma.
                "paligemma_model/(.+)": r"PaliGemma/\1",
            }
        )

    return ConvertConfig()


class IgnoreCustomTagsLoader(yaml.Loader):
    """A custom loader that ignores custom tags."""


def ignore_custom_tags(loader: yaml.Loader, _tag_suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


# Register the multi-constructor to handle unknown tags
IgnoreCustomTagsLoader.add_multi_constructor("!", ignore_custom_tags)


def _load_processor_msgpack(path: pathlib.Path | str) -> dict:
    """Load a processor spec from a msgpack file."""
    path = pathlib.Path(path)
    with path.open() as f:
        data = f.read()
        return yaml.load(data, Loader=IgnoreCustomTagsLoader)
