"""TCN epoch kernels without per-batch GPU-to-host synchronization.

Numerical checks run on CUDA in stream order and abort on any nonfinite result.
Scalar losses accumulate in float64 in the original sequential batch order.
"""
from __future__ import annotations

import numpy as np
import torch

from tcn_moment.audit_protocol import metrics_from_confusion


def train_epoch(model, loader, optimizer, device, gradient_clip_norm=1.):
    model.train()
    total = torch.zeros((), dtype=torch.float64, device=device)
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, mask)
        loss = torch.nn.functional.cross_entropy(logits, y)
        torch._assert_async(torch.isfinite(loss) & torch.isfinite(logits).all(),
                            'Nonfinite TCN loss/logits; no fallback')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        torch._assert_async(torch.isfinite(norm), 'Nonfinite TCN gradient; no skipped steps')
        optimizer.step()
        total.add_(loss.detach().to(torch.float64) * len(y))
    # The CPU read synchronizes all queued assertions before epoch commit.
    return float(total.cpu()) / len(loader.dataset)


def validation(model, loader, device):
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=device)
    probabilities, labels = [], []
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            loss = torch.nn.functional.cross_entropy(logits, y)
            torch._assert_async(torch.isfinite(loss) & torch.isfinite(logits).all(),
                                'Nonfinite TCN validation output')
            total.add_(loss.to(torch.float64) * len(y))
            probabilities.append(logits.softmax(1))
            labels.append(y)
    probs = torch.cat(probabilities).cpu().numpy()
    truth = torch.cat(labels).cpu().numpy()
    cm = np.bincount(3 * truth + probs.argmax(1), minlength=9).reshape(3, 3).tolist()
    return {**metrics_from_confusion(cm), 'confusion_matrix':cm,
            'val_loss':float(total.cpu())/len(truth)}, probs
