"""CPU-only E07 learned-state, optimizer, RNG and prepared-input audit; no inference."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from audit_e05c_saved_outputs import digest, read


def audit(root):
    import numpy as np
    import torch

    frozen = read(root / "frozen_protocol.json")
    p = frozen["protocol"]
    manifest = read(root / "output_manifest.json")
    for name, want in manifest.items():
        if digest(root / name) != want:
            raise ValueError(f"Saved output changed: {name}")
    if read(root / "suite_status.json")["status"] != "completed":
        raise ValueError("Suite incomplete")
    steps = math.ceil(p["train_count"] / p["training"]["effective_batch_size"]) * 30
    records = []

    def same_state(a, b):
        return a.keys() == b.keys() and all(
            a[part].keys() == b[part].keys()
            and all(torch.equal(a[part][k], b[part][k]) for k in a[part]) for part in a
        )

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        return True

    for seed in p["seeds"]:
        prep = root / "prepared" / f"seed{seed}"
        pm = read(prep / "manifest.json")
        if pm["protocol_sha256"] != frozen["protocol_sha256"] or pm["test_encoded"]:
            raise ValueError("Prepared scope changed")
        arrays = {}
        for split, count in (("train", 9827), ("validation", 3510)):
            file = prep / f"{split}.npz"
            if digest(file) != pm["splits"][split]:
                raise ValueError("Prepared array hash changed")
            with np.load(file, allow_pickle=False) as archive:
                arrays[split] = {k: archive[k] for k in archive.files}
            a = arrays[split]
            if (a["x"].shape != (count, 1024) or a["mask"].shape != (count, 1024)
                or a["statistics"].shape != (count, 15) or a["labels"].shape != (count,)
                or len(set(a["sample_ids"])) != count
                or not all(np.isfinite(a[k]).all() for k in ("x", "mask", "statistics"))):
                raise ValueError("Invalid prepared shape/value/count")
        split = read(prep / "split_manifest.json")["splits"]
        train_ids, val_ids = arrays["train"]["sample_ids"], arrays["validation"]["sample_ids"]
        if (not np.array_equal(val_ids, split["validation"]) or not set(train_ids) <= set(split["train"])
            or set(train_ids) & set(val_ids) or set(train_ids) & set(split["test"])
            or set(val_ids) & set(split["test"])):
            raise ValueError("Prepared split overlap/order mismatch")
        parent_rel = f"artifacts/improvements/e03e_gain_replication/e03e_gain_replication_20261018_v1_seed{seed}_fusion"
        parent = root / "input_snapshot" / parent_rel
        if digest(parent / "model_best.pt") != frozen["file_sha256"][parent_rel + "/model_best.pt"]:
            raise ValueError("Parent checkpoint snapshot changed")
        original = torch.load(parent / "model_best.pt", map_location="cpu", weights_only=True)
        np.testing.assert_array_equal(train_ids, np.load(parent / "train_ids.npy", allow_pickle=False))
        for condition in ("frozen_fusion", "lora8_fusion"):
            d = root / f"seed{seed}" / condition
            m = read(d / "metrics.json")
            metadata, history = read(d / "experiment_plan.json"), read(d / "history.json")
            # Only trusted, hash-verified local experiment artifacts are loaded.
            last = torch.load(d / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            best = torch.load(d / "model_best.pt", map_location="cpu", weights_only=True)
            if (last["metadata"] != metadata or best["metadata"] != metadata
                or last["epoch"] != 30 or last["history"] != history
                or best["epoch"] != m["best_epoch"] or last["best_epoch"] != m["best_epoch"]
                or abs(best["validation_macro_f1"] - m["validation_metrics"]["macro_f1"]) > 1e-12
                or best["validation_macro_f1"] != last["best"]
                or not same_state(best["learned_state"], last["best_learned_state"])):
                raise ValueError("Checkpoint selection/history/metadata mismatch")
            np.testing.assert_array_equal(train_ids, np.load(d / "train_ids.npy", allow_pickle=False))
            for state in (best["learned_state"], last["learned_state"]):
                if not finite(state):
                    raise ValueError("Nonfinite learned parameters")
                for name in ("stats_mean", "stats_scale"):
                    if not torch.equal(state["classifier"][name], original["learned_state"]["classifier"][name]):
                        raise ValueError("Parent normalizer changed")
                adapters = state["adapters"]
                if condition == "frozen_fusion":
                    if adapters:
                        raise ValueError("Frozen control has adapter parameters")
                elif len(adapters) != 96 or sum(v.numel() for v in adapters.values()) != 786432:
                    raise ValueError("Unexpected rank8 q/v adapter state")
                elif any(not bool(v.abs().sum() > 0) for v in adapters.values()):
                    raise ValueError("Unupdated/zero adapter tensor")
            opt = last["optimizer"]
            roles = [g["role"] for g in opt["param_groups"]]
            expected_roles = ["head"] if condition == "frozen_fusion" else ["head", "adapters"]
            optimizer_steps = sorted({int(v["step"]) for v in opt["state"].values()})
            if (roles != expected_roles or optimizer_steps != [steps] or not finite(opt["state"])
                or any(g["lr"] != 1e-5 or g["weight_decay"] != .01 or g["fused"] is not False for g in opt["param_groups"])):
                raise ValueError("Optimizer roles/steps/schedule/state mismatch")
            scaler = last["scaler"]
            if scaler["scale"] != 1024 or scaler["growth_interval"] != 1000000000 or scaler["_growth_tracker"] != steps:
                raise ValueError("Fixed scaler or committed update count mismatch")
            rng = last["rng"]
            required = {"python", "numpy", "torch_cpu", "torch_cuda", "data_loader_generator"}
            if not required <= rng.keys() or len(rng["torch_cuda"]) != 1:
                raise ValueError("Missing CPU/CUDA/shuffle RNG state")
            if not all(isinstance(rng[k], torch.Tensor) and rng[k].numel() > 0 for k in ("torch_cpu", "data_loader_generator")):
                raise ValueError("Empty saved RNG state")
            records.append({"seed": seed, "condition": condition, "selected_epoch": best["epoch"],
                            "last_epoch": last["epoch"], "optimizer_steps_per_parameter": optimizer_steps,
                            "optimizer_roles": roles, "finite_state": True, "normalizer_unchanged": True,
                            "best_matches_committed_checkpoint": True, "rng_saved": sorted(required)})
    return {"status": "passed", "device": "cpu", "model_inference_performed": False,
            "test_evaluated": False, "test_ids_used_only_for_disjointness": True,
            "checkpoint_files_audited": 16, "case_count": len(records),
            "total_optimizer_updates": len(records) * steps, "per_case": records,
            "summary_sha256": digest(root / "summary.json"), "audit_source_sha256": digest(Path(__file__))}


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = audit(args.analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
