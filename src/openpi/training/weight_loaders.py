import collections
import dataclasses
import logging
import os
from typing import Protocol, runtime_checkable

import fsspec
import jax
import numpy as np

import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params: ...


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    ckpt_path: str

    def load(self, params: at.Params) -> at.Params:
        return _checkpoints.restore_params(self.ckpt_path)


def _recover_tree(d: dict) -> dict:
    """Recover a tree from a flat dict delimited by '/'."""
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in d.items():
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        tree[k] = _recover_tree(dict(kv_pairs))
    return tree


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    cache_path: str | None = os.path.expanduser("~/.cache/openpi/paligemma.npz")  # noqa: PTH111, RUF009

    def load(self, params: at.Params) -> at.Params:
        logger.info("Loading PaliGemma weights. This may take a while the first time.")
        with fsspec.open(
            "filecache::gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
            gs={"token": "anon"},
            filecache={"cache_storage": self.cache_path},
        ) as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        # The weights are stored in a special big_vision format, so we need a special function to unflatten them.
        paligemma_params = _recover_tree(flat_params)["params"]

        # Now we will do our own flattening to merge the PaliGemma weights with the action expert weights.
        leaves, treedef = jax.tree_util.tree_flatten_with_path(params["PaliGemma"])
        leaves = dict(leaves)
        for kp, v in jax.tree_util.tree_flatten_with_path(paligemma_params)[0]:
            if kp in leaves:
                logger.info(f"Overwriting {jax.tree_util.keystr(kp)}")
                leaves[kp] = v

        new_paligemma_params = jax.tree_util.tree_unflatten(treedef, leaves.values())
        return {**params, "PaliGemma": new_paligemma_params}
