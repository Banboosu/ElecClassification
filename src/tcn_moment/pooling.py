"""Untrained, complete-valid-patch pooling; the encoder/masking recipe is unchanged."""

from __future__ import annotations

import numpy as np
import torch

from tcn_moment.train_moment import masked_pool_embeddings, sequence_mask_to_patch_mask


def pool_embeddings(embeddings, input_mask, patch_len, patch_stride):
    if embeddings.ndim != 3 or not torch.isfinite(embeddings).all():
        raise ValueError("Finite [batch,patch,feature] embeddings required.")
    patch_mask = sequence_mask_to_patch_mask(input_mask, patch_len, patch_stride)
    if patch_mask.shape != embeddings.shape[:2] or (patch_mask.sum(dim=1) == 0).any():
        raise ValueError("Expected at least one complete valid patch and matching mask.")
    mean = masked_pool_embeddings(embeddings, input_mask, patch_len, patch_stride)
    maximum = embeddings.masked_fill(~patch_mask.unsqueeze(-1), -torch.inf).max(dim=1).values
    counts = patch_mask.sum(dim=1, keepdim=True)
    ranks = patch_mask.long().cumsum(dim=1) - 1
    segments = torch.div(ranks * 3, counts, rounding_mode="floor")
    pieces = []
    for index in range(3):
        active = patch_mask & (segments == index)
        weights = active.unsqueeze(-1).to(embeddings.dtype)
        pieces.append((embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1))
    result = {"mean": mean, "max": maximum, "thirds": torch.cat(pieces, dim=1)}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Nonfinite pooled output.")
    return result


def forward_poolings(model, x, mask):
    output = model.embed(x_enc=x, input_mask=mask, reduction="none")
    embeddings = output.embeddings
    if embeddings.ndim != 4:
        raise ValueError("Expected unreduced [batch,channel,patch,feature] output.")
    batch, channels, patches, dimension = embeddings.shape
    head_dimension = int(model.head.linear.in_features)
    if head_dimension == dimension:
        tokens = embeddings.mean(dim=1)
    elif head_dimension == dimension * channels:
        tokens = embeddings.permute(0, 2, 3, 1).reshape(batch, patches, dimension * channels)
    else:
        raise ValueError("Model channel/head dimension mismatch.")
    return pool_embeddings(tokens, mask, int(model.patch_len), int(model.config.patch_stride_len))


def representation(arrays, pooling):
    if pooling == "mean_max":
        return np.concatenate([arrays["mean"], arrays["max"]], axis=1)
    if pooling not in {"mean", "thirds"}:
        raise ValueError("Unknown fixed pooling.")
    return arrays[pooling]


def training_length_edges(lengths):
    lengths = np.asarray(lengths)
    if (
        lengths.ndim != 1
        or not len(lengths)
        or not np.isfinite(lengths).all()
        or (lengths <= 0).any()
    ):
        raise ValueError("Positive selected-training lengths required.")
    return np.quantile(lengths, [1 / 3, 2 / 3], method="linear").tolist()


def length_groups(lengths, edges):
    edges = np.asarray(edges)
    if edges.shape != (2,) or not np.isfinite(edges).all() or edges[0] > edges[1]:
        raise ValueError("Two ordered training-derived cutpoints required.")
    return np.searchsorted(edges, np.asarray(lengths), side="left")
