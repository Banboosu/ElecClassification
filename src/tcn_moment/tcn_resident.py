"""GPU-resident batching with the original DataLoader sampling/RNG semantics.

FP32 values, shapes, optimizer and training checks stay in the existing trainer.
This module changes data movement only; no AMP, TF32, compile or batch-size change.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


class ResidentLoader:
    """Gather full batches on device; preserve base-seed and BatchSampler draws."""

    def __init__(self, x, mask, y, batch_size, *, shuffle, generator=None, device="cuda"):
        if len(x) != len(mask) or len(y) != len(x) or not len(y):
            raise ValueError("Require nonempty aligned training tensors")
        self.device = torch.device(device)
        self.values = torch.tensor(x[:, None, :], dtype=torch.float32, device=self.device)
        self.masks = torch.tensor(mask, dtype=torch.float32, device=self.device)
        self.labels = torch.tensor(y, dtype=torch.long, device=self.device)
        self.reference = DataLoader(range(len(y)), batch_size=batch_size, shuffle=shuffle,
                                    num_workers=0, generator=generator)
        self.dataset = self.reference.dataset
        self.generator = generator

    def __len__(self):
        return len(self.reference)

    def __iter__(self):
        # _BaseDataLoaderIter draws this seed even with zero workers. Train uses
        # its supplied CPU generator; validation consumes the default CPU RNG.
        torch.empty((), dtype=torch.int64).random_(generator=self.generator).item()
        batches = list(self.reference.batch_sampler)
        # One index transfer per epoch avoids a blocking host-to-device copy at
        # every microbatch. The CPU sampler draws the same permutations; its
        # final empty randperm is consumed earlier, with identical epoch-end RNG.
        indices = torch.tensor([i for batch in batches for i in batch],
                               dtype=torch.long, device=self.device)
        start = 0
        for batch in batches:
            selected = indices[start:start + len(batch)]
            start += len(batch)
            yield (self.values.index_select(0, selected), self.masks.index_select(0, selected),
                   self.labels.index_select(0, selected))


def make_resident_loader(x: np.ndarray, mask: np.ndarray, y: np.ndarray,
                         batch_size: int, num_workers: int, *, shuffle: bool,
                         generator=None):
    if num_workers != 0:
        raise ValueError("Resident backend requires the frozen num_workers=0 protocol")
    return ResidentLoader(x, mask, y, batch_size, shuffle=shuffle, generator=generator)
