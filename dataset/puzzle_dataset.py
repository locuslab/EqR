"""Streaming puzzle dataset used by EqR training and evaluation.

This loader follows the public HRM ``PuzzleDataset`` layout from
https://github.com/sapientinc/HRM and keeps the same on-disk arrays. EqR adds
checkpointable sampler state through ``state_dict``/``load_state_dict`` and an
optional shared sampler snapshot so long training jobs can resume without
replaying or reshuffling already-consumed batches.
"""

import os
import json
from typing import Any, Dict, Optional, MutableMapping
import copy
import math

import numpy as np
import pydantic

import torch
from utils.printing import colored_exception
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata


def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + rng.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool

    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.

    rank: int
    num_replicas: int


class PuzzleDataset(IterableDataset):
    def __init__(
        self,
        config: PuzzleDatasetConfig,
        split: str = "train",
        shared_sampler_state: Optional[MutableMapping[str, Any]] = None,
    ):
        super().__init__()
        if not os.path.isdir(os.path.join(config.dataset_path, split)):
            colored_exception(FileNotFoundError, f"Dataset split {split} in {config.dataset_path} does not exist.")
        
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        
        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0
        self._train_state: Optional[Dict[str, Any]] = None
        self._shared_sampler_state = shared_sampler_state

        self._publish_shared_state()

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        metadata_path = os.path.join(self.config.dataset_path, self.split, "dataset.json")
        with open(metadata_path, "r") as f:
            metadata = PuzzleDatasetMetadata(**json.load(f))

        if metadata.total_samples is None:
            total_samples = 0
            for set_name in metadata.sets:
                indices_path = os.path.join(
                    self.config.dataset_path,
                    self.split,
                    f"{set_name}__puzzle_indices.npy",
                )
                indices = np.load(indices_path, mmap_mode="r")
                if indices.size == 0:
                    continue
                total_samples += int(indices[-1])
            metadata.total_samples = total_samples

        return metadata

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _normalize_train_state(self, train_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not train_state:
            return None

        normalized = {
            "set_names": list(train_state.get("set_names", [])),
            "set_idx": int(train_state.get("set_idx", 0)),
            "start_index": int(train_state.get("start_index", 0)),
            "epoch_idx": int(train_state.get("epoch_idx", 0)),
            "group_order": (
                list(train_state.get("group_order", []))
                if train_state.get("group_order") is not None
                else None
            ),
            "rng_state": copy.deepcopy(train_state.get("rng_state")),
        }

        return normalized

    def _build_state_snapshot(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "_iters": self._iters,
            "train_state": self._normalize_train_state(self._train_state),
        }

    def _publish_shared_state(self) -> None:
        if self._shared_sampler_state is None:
            return

        snapshot = self._build_state_snapshot()
        self._shared_sampler_state["state"] = copy.deepcopy(snapshot)

    def _restore_from_shared_state(self) -> None:
        if self._shared_sampler_state is None:
            return

        shared_snapshot = self._shared_sampler_state.get("state")
        if not shared_snapshot:
            return

        if shared_snapshot.get("split") != self.split:
            return

        self._iters = int(shared_snapshot.get("_iters", self._iters))
        shared_train_state = shared_snapshot.get("train_state")
        normalized = self._normalize_train_state(shared_train_state)
        if normalized is not None and self._data is not None:
            expected_set_names = list(self._data.keys())
            if normalized.get("set_names") and normalized["set_names"] != expected_set_names:
                colored_exception(RuntimeError, "Shared sampler state set names do not match loaded dataset.")
        self._train_state = normalized

    def num_batches(self) -> Optional[int]:
        """Return the total number of batches produced per iterator call when known."""
        if not self.config.test_set_mode:
            return None

        self._lazy_load_dataset()
        assert self._data is not None

        total_batches = 0
        for set_name in self.metadata.sets:
            dataset = self._data[set_name]
            total_examples = len(dataset["inputs"])
            total_batches += math.ceil(total_examples / self.config.global_batch_size)

        return total_batches

    def _collate_batch(self, batch):
        converted = {}
        for key, value in batch.items():
            converted[key] = value.astype(np.int32)

        if self.metadata.ignore_label_id is not None:
            converted["labels"][converted["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        if converted["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - converted["puzzle_identifiers"].size
            pad_defaults = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            converted = {
                key: np.pad(
                    value,
                    ((0, pad_size),) + ((0, 0),) * (value.ndim - 1),
                    constant_values=pad_defaults.get(key, 0),
                )
                for key, value in converted.items()
            }

        return {k: torch.from_numpy(v) for k, v in converted.items()}

    def _iter_test(self):
        self._publish_shared_state()
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices],
                })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        """Yield sharded training batches while persisting sampler state.

        The sampler walks through each dataset split (``set_names``) one epoch at
        a time, saving enough information in ``self._train_state`` to resume from
        an arbitrary checkpoint:

        - ``set_idx`` tracks which subset we are currently iterating.
        - ``epoch_idx`` counts how many complete permutations of the subset have
            been processed for the current iterator call, limited by
            ``config.epochs_per_iter``.
        - ``group_order`` is either ``None`` or a permutation of the group ids
            (``dataset["group_indices"]``) that determines which puzzle group to
            sample next.
        - ``start_index`` is the current cursor into ``group_order`` so we can
            resume mid-epoch after yielding a batch.
        - ``rng_state`` caches the Philox bit generator state so that resumed
            runs replay the exact same stochastic choices.

        Inside an epoch, we repeatedly call ``_sample_batch`` to draw puzzles and
        assemble a global batch. We drop any under-filled tail batches to keep
        per-rank slices aligned, collate numpy arrays into tensors, and yield the
        local shard alongside the effective global batch size. After a subset is
        finished we reset its cached state and advance to the next subset. Once
        all subsets are consumed, the iterator state is cleared so the next call
        starts fresh.
        """
        assert self._data is not None

        self._restore_from_shared_state()

        # Initialize or validate train state
        set_names = list(self._data.keys())
        if self._train_state is None:
            self._train_state = {
                "set_names": set_names,
                "set_idx": 0,
                "group_order": None,
                "start_index": 0,
                "rng_state": None,
                "epoch_idx": 0,
            }
        else:
            if self._train_state.get("set_names") != set_names:
                colored_exception(RuntimeError, "Dataset structure changed between checkpoints; cannot resume.")

        self._publish_shared_state()

        while self._train_state["set_idx"] < len(set_names):
            set_name = set_names[self._train_state["set_idx"]]
            dataset = self._data[set_name]

            epoch_idx = int(self._train_state.get("epoch_idx", 0))

            # Restore RNG or seed a new iterator pass
            rng_state = self._train_state.get("rng_state")
            if rng_state is None:
                self._iters += 1
                rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            else:
                rng = np.random.Generator(np.random.Philox())
                rng.bit_generator.state = rng_state  # type: ignore[arg-type]

            group_order_cached = self._train_state.get("group_order")
            if group_order_cached is not None:
                group_order = np.asarray(group_order_cached, dtype=np.int64)
            else:
                group_order = np.empty((0,), dtype=np.int64)

            start_index = int(self._train_state.get("start_index", 0))

            while epoch_idx < self.config.epochs_per_iter:
                if group_order.size == 0 or start_index >= group_order.size:
                    group_order = rng.permutation(dataset["group_indices"].size - 1)
                    start_index = 0
                    epoch_idx += 1

                    # If there are no groups we cannot sample batches
                    if group_order.size == 0:
                        break

                while start_index < group_order.size:
                    start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                        rng,
                        group_order=group_order,
                        puzzle_indices=dataset["puzzle_indices"],
                        group_indices=dataset["group_indices"],
                        start_index=start_index,
                        global_batch_size=self.config.global_batch_size,
                    )

                    # Persist state before yielding
                    self._train_state["start_index"] = start_index
                    self._train_state["rng_state"] = rng.bit_generator.state
                    self._train_state["group_order"] = group_order.tolist()
                    self._train_state["epoch_idx"] = epoch_idx
                    self._publish_shared_state()

                    global_effective_batch_size = batch_puzzle_indices.size

                    # Drop last batch if it does not fill a global batch
                    if global_effective_batch_size < self.config.global_batch_size:
                        start_index = group_order.size
                        break

                    batch_indices = batch_indices[
                        self.config.rank * self.local_batch_size : (self.config.rank + 1) * self.local_batch_size
                    ]
                    batch_puzzle_indices = batch_puzzle_indices[
                        self.config.rank * self.local_batch_size : (self.config.rank + 1) * self.local_batch_size
                    ]
                    batch = self._collate_batch(
                        {
                            "inputs": dataset["inputs"][batch_indices],
                            "labels": dataset["labels"][batch_indices],
                            "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices],
                        }
                    )

                    yield set_name, batch, global_effective_batch_size

                # Force regeneration of a fresh permutation on the next loop
                group_order = np.empty((0,), dtype=np.int64)

            # Finished current set
            self._train_state["group_order"] = None
            self._train_state["start_index"] = 0
            self._train_state["rng_state"] = None
            self._train_state["epoch_idx"] = 0
            self._train_state["set_idx"] += 1
            self._publish_shared_state()

        # Reset state so the next iterator call starts fresh
        self._train_state = None
        self._publish_shared_state()
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
    
    def state_dict(self) -> Dict[str, Any]:
        if self._shared_sampler_state is not None:
            shared_snapshot = self._shared_sampler_state.get("state")
            if shared_snapshot is not None:
                return copy.deepcopy(shared_snapshot)

        return copy.deepcopy(self._build_state_snapshot())

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            self._train_state = None
            self._iters = 0
            self._publish_shared_state()
            return

        if state.get("split", self.split) != self.split:
            colored_exception(ValueError, f"Attempted to load state for split {state.get('split')} into dataset split {self.split}.")

        self._iters = int(state.get("_iters", 0))

        train_state = state.get("train_state")
        if train_state is None:
            self._train_state = None
            self._publish_shared_state()
            return

        self._train_state = self._normalize_train_state(train_state)
        self._publish_shared_state()

