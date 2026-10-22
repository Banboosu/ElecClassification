"""E18c: validation-selected TCN at 1/5/10/20/40% (sensitivity, not main protocol)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.run_e12_tcn_stack import _tcn_predict
from tcn_moment.training_utils import seed_everything
from tcn_moment.train_tcn import make_loader

OUT = Path("artifacts/improvements/e18c_tcn_selected/e18c_tcn_selected_20261020_v1")
SEEDS = (42, 44, 45, 46)
FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.4)


def tag(f):
    return f"{int(round(f * 100))}p"


def train_selected(x, mask, y, x_val, mask_val, y_val, seed, device, epochs=50):
    seed_everything(torch, seed)
    model = pilot.build_model("legacy_masked50").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = make_loader(x, mask, y, 32, 0, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_state, best_score, history = None, -1.0, []
    for epoch in range(epochs):
        model.train()
        for bx, bm, by in loader:
            bx, bm, by = bx.to(device), bm.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(bx, bm), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        prob = _tcn_predict(model, x_val, mask_val, device)
        score = f1_score(y_val, prob.argmax(1), average="macro")
        history.append({"epoch": epoch + 1, "val_macro_f1": float(score)})
        if score > best_score:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_score, history


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("E18c requires CUDA")
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"test_evaluated": False, "protocol": "validation-selected TCN, 50 epochs", "conditions": []}
    for seed in SEEDS:
        cfg = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
        raw = load_dataset(replace(cfg.data, normalize="none"))
        for fraction in FRACTIONS:
            t = tag(fraction)
            idx = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
            model, best, history = train_selected(
                raw.x_train[idx], raw.mask_train[idx], raw.y_train[idx],
                raw.x_val, raw.mask_val, raw.y_val, seed * 100 + 7, device,
            )
            prob = _tcn_predict(model, raw.x_val, raw.mask_val, device)
            pred = prob.argmax(1)
            metrics = run_e08._evaluate(raw.y_val, pred, prob, ["0", "1", "2"], 2)
            path = OUT / f"seed{seed}" / "conditions" / t / "tcn_selected"
            path.mkdir(parents=True, exist_ok=True)
            with (path / "predictions.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2"])
                for sid, yy, pp, ss in zip(raw.ids_val, raw.y_val, pred, prob, strict=True):
                    w.writerow([sid, "validation", int(yy), int(pp), *[f"{v:.12f}" for v in ss]])
            rec = {"seed": seed, "fraction": fraction, "tag": t, "method": "tcn_selected",
                   "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history]) + 1),
                   "best_val_macro_f1": best, "validation": metrics,
                   "history": history, "test_evaluated": False}
            summary["conditions"].append(rec)
            atomic_write_json(path / "metrics.json", rec)
            print(f"E18c {seed} {t}: best epoch {rec['best_epoch']} {100*best:.2f}", flush=True)
    atomic_write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    from tcn_moment import run_e08  # local import to avoid circular
    main()
