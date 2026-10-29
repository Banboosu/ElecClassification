"""E12: cross-fitted TCN + DYN + frozen MOMENT/ΔP stacking (validation only).

For each seed and label budget, the 40% training subset is split into five
stratified folds.  A legacy masked TCN is trained inside each fold and predicts
its held-out fold (OOF) and the global validation split.  DYN-HGB and
MOMENT+ΔP-logistic branches are fitted with the same fold assignment.  A logistic
meta-learner is fitted on the OOF probabilities and evaluated once on validation.

No test split is parsed for scoring, and validation is not used to select the TCN
checkpoint: every TCN is trained for a fixed 50 epochs with a cosine schedule.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from tcn_moment import run_e08
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.data import stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.label_semantics import BATTERY_LABEL, class_index
from tcn_moment.train_tcn import make_loader
from tcn_moment.training_utils import seed_everything

DEFAULT_PROTOCOL = Path("configs/experiments/improvements/e12_tcn_stack_protocol.json")
E08_SUITE = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
E01_CACHE = Path("artifacts/improvements/e01_20261017_v2_features")
E09_CACHE = Path("artifacts/improvements/e09_diff_features")
PACKAGES = ("torch", "numpy", "pandas", "scikit-learn")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / p["suite"]


def _validate(p: dict[str, Any]) -> None:
    if p.get("experiment_id") != "E12":
        raise ValueError("Unexpected experiment id")
    if p.get("seeds") != [42, 44, 45, 46]:
        raise ValueError("E12 fixes seeds 42/44/45/46")
    if p.get("fractions") not in ([0.4], [0.1], [0.1, 0.4]):
        raise ValueError("E12 supports 10%, 40%, or 10%/40%")
    if p.get("folds") != 5 or p.get("epochs") != 50:
        raise ValueError("E12 fixes 5 folds and 50 epochs")
    if p.get("test_evaluated") or p.get("automatic_expansion"):
        raise ValueError("E12 is validation-only and does not auto-expand")


def _load(path: Path) -> dict[str, Any]:
    p = _read(path)
    _validate(p)
    return p


def _sources(protocol_path: Path) -> list[Path]:
    directory = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        directory / "run_e08.py",
        directory / "label_semantics.py",
        directory / "run_e09_diff_stream.py",
        directory / "dynamics_features.py",
        directory / "statistical_features.py",
        directory / "tcn_legacy_pilot.py",
        directory / "train_tcn.py",
        directory / "training_utils.py",
        directory / "audit_protocol.py",
        directory / "data.py",
        directory / "io_utils.py",
        directory / "config.py",
        directory / "experiment.py",
        directory / "metrics.py",
        Path("tests/test_e12_tcn_stack.py"),
        Path("scripts/run_e12_remote.sh"),
        protocol_path,
        Path("legacy/改进TCN.py"),
    ]


def freeze(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(protocol_path)
    if list(root.glob("seed*")):
        raise ValueError("Freeze must precede training")
    if sha256_file(Path(p["dataset"])) != p["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    for seed in p["seeds"]:
        manifest = _read(Path(p["split_manifests"][str(seed)]))
        if (
            manifest["protocol"]["random_state"] != seed
            or manifest["protocol"]["dataset_sha256"] != p["dataset_sha256"]
            or manifest["counts"]["train"] != 24569
            or manifest["counts"]["validation"] != 3510
        ):
            raise ValueError(f"Split mismatch for seed {seed}")
    checks = pilot.synthetic_check()
    pilot.configure()
    files = set(_sources(protocol_path))
    files.update(Path(p["split_manifests"][str(seed)]) for seed in p["seeds"])
    files.add(Path(p["dataset"]))
    for seed in p["seeds"]:
        files.update(
            [
                E08_SUITE / f"seed{seed}" / "features.npz",
                E08_SUITE / f"seed{seed}" / "seed_results.json",
                E01_CACHE / f"seed{seed}" / "train.npz",
                E01_CACHE / f"seed{seed}" / "validation.npz",
                E01_CACHE / f"seed{seed}" / "manifest.json",
                E09_CACHE / f"seed{seed}" / "train.npz",
                E09_CACHE / f"seed{seed}" / "validation.npz",
                E09_CACHE / f"seed{seed}" / "manifest.json",
            ]
        )
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing frozen input: {file}")
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": p,
        "protocol_sha256": sha256_file(protocol_path),
        "file_sha256": {str(file): sha256_file(file) for file in sorted(files)},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda,
        },
        "synthetic_checks": checks,
        "test_evaluated": False,
    }
    for file in sorted(files):
        if file.parts[0] in {"src", "configs", "tests", "scripts", "docs", "legacy"}:
            destination = root / "source_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    atomic_write_json(root / "frozen_protocol.json", frozen)
    print(f"Froze E12: {root / 'frozen_protocol.json'}", flush=True)
    return frozen


def verify(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    frozen = _read(_root(p) / "frozen_protocol.json")
    if frozen["protocol"] != p or frozen["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen protocol changed")
    for file, expected in frozen["file_sha256"].items():
        if sha256_file(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    installed = {name: importlib.metadata.version(name) for name in PACKAGES}
    if frozen["packages"] != installed:
        raise ValueError("Dependency versions changed")
    return frozen


def _seed_protocol(p: dict[str, Any], seed: int) -> dict[str, Any]:
    cp = dict(p)
    cp["seed"] = seed
    cp["split_manifest"] = p["split_manifests"][str(seed)]
    return cp


def _tcn_predict(model: torch.nn.Module, x: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = make_loader(x, mask, np.zeros(len(x), dtype=np.int64), 128, 0, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch_x, batch_mask, _ in loader:
            logits = model(batch_x.to(device), batch_mask.to(device))
            chunks.append(logits.softmax(-1).cpu().numpy())
    probabilities = np.concatenate(chunks)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite TCN probabilities")
    return probabilities


def _train_tcn(
    x_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    seed_everything(torch, seed)
    model = pilot.build_model("legacy_masked50").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    loader = make_loader(x_train, mask_train, y_train, 32, 0, shuffle=True, generator=generator)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_mask, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_mask)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite TCN training output")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model


def _meta(
    oof: np.ndarray,
    y_train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    grid = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    selected, cv_results = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        grid,
        oof,
        y_train,
        folds=3,
        seed=seed,
    )
    model = LogisticRegression(C=selected["C"], max_iter=5000, random_state=seed).fit(oof, y_train)
    return model.predict(validation), model.predict_proba(validation), selected, cv_results


def run_seed(protocol_path: Path, seed: int) -> None:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    if seed not in p["seeds"]:
        raise ValueError("Seed not in frozen protocol")
    cp = _seed_protocol(p, seed)
    data = pilot.load_train_validation(cp)
    root = _root(p) / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    with np.load(E08_SUITE / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
        dyn_train = archive["train_features"]
        dyn_val = archive["validation_features"]
        if not np.array_equal(archive["train_ids"].astype(str), data["train"]["ids"]):
            raise ValueError("E08 train IDs mismatch")
        if not np.array_equal(archive["validation_ids"].astype(str), data["validation"]["ids"]):
            raise ValueError("E08 validation IDs mismatch")
    moment_train, _ = run_e08._load_moment_split(E01_CACHE, seed=seed, split="train", ids=data["train"]["ids"])
    moment_val, _ = run_e08._load_moment_split(E01_CACHE, seed=seed, split="validation", ids=data["validation"]["ids"])
    diff_train, _ = run_e08._load_moment_split(E09_CACHE, seed=seed, split="train", ids=data["train"]["ids"])
    diff_val, _ = run_e08._load_moment_split(E09_CACHE, seed=seed, split="validation", ids=data["validation"]["ids"])
    moment_diff_train = np.hstack([moment_train, diff_train])
    moment_diff_val = np.hstack([moment_val, diff_val])
    seed_results = _read(E08_SUITE / f"seed{seed}" / "seed_results.json")
    hgb_params = {
        record["tag"]: record["selected_params"]
        for record in seed_results["conditions"]
        if record["method"] == "dyn_hgb"
    }
    class_names = ["0", "1", "2"]
    battery_index = class_index(class_names, BATTERY_LABEL)
    if not torch.cuda.is_available():
        raise RuntimeError("E12 requires CUDA")
    device = torch.device("cuda")
    pilot.configure()
    conditions: list[dict[str, Any]] = []
    for fraction in p["fractions"]:
        tag = run_e08._tag(fraction)
        indices = stratified_train_subset_indices(
            data["train"]["y"], data["train"]["ids"], fraction, seed
        )
        x_tcn = data["train"]["x"][indices]
        mask_tcn = data["train"]["mask"][indices]
        y_train = data["train"]["y"][indices]
        x_tcn_val = data["validation"]["x"]
        mask_tcn_val = data["validation"]["mask"]
        dyn_subset = dyn_train[indices]
        md_subset = moment_diff_train[indices]
        fold_assignment = np.full(len(indices), -1, dtype=np.int8)
        oof_dyn = np.zeros((len(indices), 3), dtype=np.float64)
        oof_md = np.zeros((len(indices), 3), dtype=np.float64)
        oof_tcn = np.zeros((len(indices), 3), dtype=np.float64)
        fold_validation_tcn = []
        splitter = StratifiedKFold(n_splits=p["folds"], shuffle=True, random_state=seed)
        models_dir = root / "models" / tag
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, (train_index, valid_index) in enumerate(splitter.split(x_tcn, y_train)):
            fold_assignment[valid_index] = fold
            tcn_model = _train_tcn(
                x_tcn[train_index],
                mask_tcn[train_index],
                y_train[train_index],
                epochs=p["epochs"],
                seed=seed * 100 + fold,
                device=device,
            )
            torch.save(
                {key: value.cpu() for key, value in tcn_model.state_dict().items()},
                models_dir / f"fold{fold}.pt",
            )
            oof_tcn[valid_index] = _tcn_predict(tcn_model, x_tcn[valid_index], mask_tcn[valid_index], device)
            fold_validation_tcn.append(_tcn_predict(tcn_model, x_tcn_val, mask_tcn_val, device))
            hgb = run_e08._hgb_factory(hgb_params[tag], seed).fit(dyn_subset[train_index], y_train[train_index])
            oof_dyn[valid_index] = hgb.predict_proba(dyn_subset[valid_index])
            logistic = run_e08._logit_factory({"C": 1.0}, seed).fit(
                md_subset[train_index], y_train[train_index]
            )
            oof_md[valid_index] = logistic.predict_proba(md_subset[valid_index])
            print(f"E12 seed={seed} {tag} fold={fold} complete", flush=True)
        tcn_foldavg_val = np.mean(fold_validation_tcn, axis=0)
        tcn_full = _train_tcn(
            x_tcn,
            mask_tcn,
            y_train,
            epochs=p["epochs"],
            seed=seed * 100 + 99,
            device=device,
        )
        torch.save(
            {key: value.cpu() for key, value in tcn_full.state_dict().items()},
            models_dir / "full.pt",
        )
        tcn_full_val = _tcn_predict(tcn_full, x_tcn_val, mask_tcn_val, device)
        hgb_full = run_e08._hgb_factory(hgb_params[tag], seed).fit(dyn_subset, y_train)
        dyn_val_prob = hgb_full.predict_proba(dyn_val)
        logistic_full = run_e08._logit_factory({"C": 1.0}, seed).fit(md_subset, y_train)
        md_val_prob = logistic_full.predict_proba(moment_diff_val)

        oof_all = np.hstack([oof_dyn, oof_md, oof_tcn])
        val_all = np.hstack([dyn_val_prob, md_val_prob, tcn_full_val])
        pred_no_tcn, prob_no_tcn, params_no_tcn, cv_no_tcn = _meta(
            oof_all[:, :6], y_train, val_all[:, :6], seed=seed
        )
        pred_tcn, prob_tcn, params_tcn, cv_tcn = _meta(oof_all, y_train, val_all, seed=seed)
        for method, prediction, scores, selected, cv_results in (
            ("tcn_full", tcn_full_val.argmax(1), tcn_full_val, {}, []),
            ("tcn_foldavg", tcn_foldavg_val.argmax(1), tcn_foldavg_val, {}, []),
            ("stack_no_tcn", pred_no_tcn, prob_no_tcn, params_no_tcn, cv_no_tcn),
            ("stack_tcn", pred_tcn, prob_tcn, params_tcn, cv_tcn),
        ):
            metrics = run_e08._evaluate(
                data["validation"]["y"], prediction, scores, class_names, battery_index
            )
            prediction_path = run_e08._write_condition(
                root,
                tag=tag,
                method=method,
                ids=data["validation"]["ids"],
                y_true=data["validation"]["y"],
                y_pred=prediction,
                scores=scores,
                battery_index=battery_index,
            )
            record = {
                "seed": seed,
                "tag": tag,
                "fraction": fraction,
                "method": method,
                "n_train": len(indices),
                "selected_params": selected,
                "cv_results": cv_results,
                "validation": metrics,
                "predictions": prediction_path,
                "oof_macro_f1": (
                    float(f1_score(y_train, oof_tcn.argmax(1), average="macro"))
                    if method == "tcn_full"
                    else None
                ),
            }
            conditions.append(record)
            atomic_write_json(root / "conditions" / tag / method / "metrics.json", record)
            print(
                f"E12 seed={seed} {method} {tag}: macro-F1={100*metrics['macro_f1']:.4f}",
                flush=True,
            )
        np.savez_compressed(
            root / f"oof_{tag}.npz",
            indices=indices,
            fold_assignment=fold_assignment,
            oof_dyn=oof_dyn,
            oof_md=oof_md,
            oof_tcn=oof_tcn,
            y_train=y_train,
        )
    summary = {
        "seed": seed,
        "fractions": list(p["fractions"]),
        "conditions": conditions,
        "test_evaluated": False,
    }
    atomic_write_json(root / "seed_results.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol)
    elif args.action == "verify":
        verify(args.protocol)
    else:
        if args.seed is None:
            parser.error("run requires --seed")
        run_seed(args.protocol, args.seed)


if __name__ == "__main__":
    main()
