"""E13 zero-perturbation consistency check.

Feed clean raw validation through the same online inference path used for perturbed
conditions (recompute DYN features, MOMENT/diff embeddings and TCN forward), then
compare against the cached/saved clean E13b conditions.  Validation only, no test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcn_moment import run_e08
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.dynamics_features import extract_dynamics_features
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.run_e13_robustness import (
    _diff_pad,
    _load_tcn,
    _meta_model,
    _moment_embed,
    _normalize_pad,
    _read_predictions,
    _tcn_prob,
)
from tcn_moment.train_moment import (
    build_model,
    require_torch_and_moment,
    set_num_classes,
)

SEEDS = (42, 44, 45, 46)
TAGS = ("10p", "40p")
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
E01 = Path("artifacts/improvements/e01_20261017_v2_features")
E09 = Path("artifacts/improvements/e09_diff_features")
E12 = {
    "10p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_10p_20261020_v1"),
    "40p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_20261020_v1"),
}
E13B = Path("artifacts/improvements/e13_robustness/e13b_robustness_20261020_v1")


def _compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    diff = np.abs(reference - candidate)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "argmax_agreement": float((reference.argmax(1) == candidate.argmax(1)).mean()),
    }


def _e13_clean(tag: str, seed: int, method: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    path = E13B / f"seed{seed}" / "conditions" / tag / "clean" / method / "predictions.csv"
    return _read_predictions(path)


def main() -> None:
    protocol = json.loads(Path("configs/experiments/improvements/e13b_robustness_protocol.json").read_text())
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires CUDA")
    config = load_config(Path(protocol["config"]))
    torch_mod, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    moments = build_model(config, MOMENTPipeline, 3)
    set_num_classes(moments, 3)
    moments.init()
    moments.to(device)
    moments.eval()
    for parameter in moments.parameters():
        parameter.requires_grad = False

    records = []
    for seed in SEEDS:
        seed_config = with_random_seed(load_config(Path(protocol["config"])), seed)
        raw = load_dataset(replace(seed_config.data, normalize="none"))
        with np.load(E08 / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
            dyn_train = archive["train_features"]
            dyn_val_cached = archive["validation_features"]
        moment_train, _ = run_e08._load_moment_split(E01, seed=seed, split="train", ids=raw.ids_train)
        moment_val_cached, _ = run_e08._load_moment_split(E01, seed=seed, split="validation", ids=raw.ids_val)
        diff_train, _ = run_e08._load_moment_split(E09, seed=seed, split="train", ids=raw.ids_train)
        diff_val_cached, _ = run_e08._load_moment_split(E09, seed=seed, split="validation", ids=raw.ids_val)
        seed_results = json.loads((E08 / f"seed{seed}" / "seed_results.json").read_text())
        hgb_params = {
            record["tag"]: record["selected_params"]
            for record in seed_results["conditions"]
            if record["method"] == "dyn_hgb"
        }
        for tag, fraction in (("10p", 0.1), ("40p", 0.4)):
            indices = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
            hgb = run_e08._hgb_factory(hgb_params[tag], seed).fit(dyn_train[indices], raw.y_train[indices])
            md_model = run_e08._logit_factory({"C": 1.0}, seed).fit(
                np.hstack([moment_train[indices], diff_train[indices]]), raw.y_train[indices]
            )
            dyn_online = extract_dynamics_features(raw.x_val.astype(np.float32), raw.mask_val.astype(bool))
            x_norm = np.zeros_like(raw.x_val)
            diff_rows = []
            diff_masks = []
            for index in range(len(raw.x_val)):
                values = raw.x_val[index, raw.mask_val[index].astype(bool)]
                x_norm[index] = _normalize_pad(values, raw.x_val.shape[1])
                row, row_mask = _diff_pad(values, raw.x_val.shape[1])
                diff_rows.append(row)
                diff_masks.append(row_mask)
            moment_online = _moment_embed(moments, x_norm, raw.mask_val.astype(bool), device)
            diff_online = _moment_embed(moments, np.stack(diff_rows), np.stack(diff_masks), device)
            tcn_model = _load_tcn(E12[tag] / f"seed{seed}" / "models" / tag / "full.pt", device)
            tcn_online = _tcn_prob(tcn_model, raw.x_val.astype(np.float32), raw.mask_val, device)
            dyn_prob = hgb.predict_proba(dyn_online)
            md_prob = md_model.predict_proba(np.hstack([moment_online, diff_online]))

            with np.load(E12[tag] / f"seed{seed}" / f"oof_{tag}.npz", allow_pickle=False) as archive:
                oof_dyn = archive["oof_dyn"]
                oof_md = archive["oof_md"]
                oof_tcn = archive["oof_tcn"]
                y_oof = archive["y_train"]
            metas = {}
            for name, matrix in (
                ("tcn_dyn", np.hstack([oof_tcn, oof_dyn])),
                ("dyn_md", np.hstack([oof_dyn, oof_md])),
                ("full", np.hstack([oof_dyn, oof_md, oof_tcn])),
            ):
                model, selected, cv_results = _meta_model(matrix, y_oof, seed)
                metas[name] = model
            base = {
                "dyn_hgb": dyn_prob,
                "md_logit": md_prob,
                "tcn": tcn_online,
                "tcn_dyn": metas["tcn_dyn"].predict_proba(np.hstack([tcn_online, dyn_prob])),
                "dyn_md": metas["dyn_md"].predict_proba(np.hstack([dyn_prob, md_prob])),
                "full": metas["full"].predict_proba(np.hstack([dyn_prob, md_prob, tcn_online])),
            }
            checks = {
                "dyn_features": _compare(dyn_val_cached, dyn_online),
                "moment_orig": _compare(moment_val_cached, moment_online),
                "moment_diff": _compare(diff_val_cached, diff_online),
            }
            for method, probabilities in base.items():
                ids, y, reference, _ = _e13_clean(tag, seed, method)
                if not np.array_equal(y, raw.y_val):
                    raise ValueError(f"E13 clean labels differ: {tag}/{seed}/{method}")
                reference = reference[[ids.index(sample_id) for sample_id in raw.ids_val.astype(str).tolist()]]
                checks[method] = _compare(reference, probabilities)
            records.append({"seed": seed, "tag": tag, **checks})
            print(
                f"E13-zero seed={seed} {tag}: "
                f"dyn={checks['dyn_features']['max_abs_diff']:.2e} "
                f"orig={checks['moment_orig']['max_abs_diff']:.2e} "
                f"diff={checks['moment_diff']['max_abs_diff']:.2e} "
                f"full={checks['full']['max_abs_diff']:.2e} "
                f"full_argmax={checks['full']['argmax_agreement']:.3f}",
                flush=True,
            )
    root = Path("artifacts/improvements/e13_zero_perturbation")
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "test_evaluated": False,
        "conditions": len(records),
        "records": records,
        "max": {
            method: max(record[method]["max_abs_diff"] for record in records)
            for method in ("dyn_features", "moment_orig", "moment_diff", "dyn_hgb", "md_logit", "tcn", "tcn_dyn", "dyn_md", "full")
        },
        "min_full_argmax_agreement": min(record["full"]["argmax_agreement"] for record in records),
    }
    atomic_write_json(root / "summary.json", summary)
    lines = [
        "# E13 零扰动在线推理一致性检查",
        "",
        "以 clean raw validation 重新走完整在线推理（DYN、MOMENT、ΔP、TCN、meta），与 E13b clean 保存的缓存预测对比。",
        "",
        "| seed | tag | DYN 特征 maxdiff | orig emb maxdiff | diff emb maxdiff | full maxdiff | full argmax 一致率 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['seed']} | {record['tag']} | {record['dyn_features']['max_abs_diff']:.2e} | "
            f"{record['moment_orig']['max_abs_diff']:.2e} | {record['moment_diff']['max_abs_diff']:.2e} | "
            f"{record['full']['max_abs_diff']:.2e} | {record['full']['argmax_agreement']:.3f} |"
        )
    lines += [
        "",
        "## 汇总",
        "",
        f"- {len(records)} 个 seed×预算条件；0 test。",
        f"- DYN 特征最大差异：{summary['max']['dyn_features']:.3e}",
        f"- MOMENT orig / diff 最大差异：{summary['max']['moment_orig']:.3e} / {summary['max']['moment_diff']:.3e}",
        f"- TCN 概率最大差异：{summary['max']['tcn']:.3e}",
        f"- full 概率最大差异：{summary['max']['full']:.3e}；full argmax 最低一致率：{summary['min_full_argmax_agreement']:.3f}",
        "",
        "## 解释",
        "",
        "- 该检查验证 E13 的扰动条件确实走同一条在线路径；clean 行不是“只复用缓存”的捷径。",
        "- TCN 前向与保存概率存在 float32/批处理级微小差异，因此 full 概率允许 ~1e-3 量级差异；argmax 应完全一致。",
        "- 仍然只是 validation 内部一致性检查，不是独立泛化或部署保证。",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
