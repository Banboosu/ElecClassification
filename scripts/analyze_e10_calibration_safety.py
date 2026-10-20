"""E10: calibration, error complementarity, length strata, and battery risk-coverage.

Uses the saved E12 validation predictions and OOF probabilities.  Temperature scaling
is fitted on OOF probabilities (cross-fitted training folds) and evaluated on
validation; no test split is read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

E12_ROOT = Path("artifacts/improvements/e12_tcn_stack")
SUITES = {
    "40p": E12_ROOT / "e12_tcn_stack_20261020_v1",
    "10p": E12_ROOT / "e12_tcn_stack_10p_20261020_v1",
}
E08_SUITE = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
SEEDS = (42, 44, 45, 46)
BATTERY = 2


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return ids, y_true, y_pred, probabilities


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _nll(probabilities: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def _brier(probabilities: np.ndarray, y: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[y]
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def _ece(probabilities: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def _fit_temperature(probabilities: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))

    def objective(log_temperature: float) -> float:
        return _nll(_softmax(logits / np.exp(log_temperature)), y)

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimisation failed")
    return float(np.exp(result.x))


def _calibration_for_seed(tag: str, seed: int) -> dict[str, Any]:
    suite = SUITES[tag]
    seed_results = _read_json(suite / f"seed{seed}" / "seed_results.json")
    record = next(
        row for row in seed_results["conditions"] if row["method"] == "stack_tcn"
    )
    c_value = float(record["selected_params"]["C"])
    with np.load(suite / f"seed{seed}" / f"oof_{tag}.npz", allow_pickle=False) as archive:
        oof = np.hstack([archive["oof_dyn"], archive["oof_md"], archive["oof_tcn"]])
        y_train = archive["y_train"]
    meta = LogisticRegression(C=c_value, max_iter=5000, random_state=seed).fit(oof, y_train)
    oof_prob = meta.predict_proba(oof)
    ids, y_val, _, val_prob = _read_predictions(
        suite / f"seed{seed}" / "conditions" / tag / "stack_tcn" / "predictions.csv"
    )
    if len(ids) != 3510:
        raise ValueError("Unexpected validation row count")
    temperature = _fit_temperature(oof_prob, y_train)
    val_logits = np.log(np.clip(val_prob, 1e-12, 1.0))
    calibrated = _softmax(val_logits / temperature)
    return {
        "seed": seed,
        "C": c_value,
        "temperature": temperature,
        "before": {
            "nll": _nll(val_prob, y_val),
            "brier": _brier(val_prob, y_val),
            "ece": _ece(val_prob, y_val),
            "accuracy": float((val_prob.argmax(1) == y_val).mean()),
        },
        "after": {
            "nll": _nll(calibrated, y_val),
            "brier": _brier(calibrated, y_val),
            "ece": _ece(calibrated, y_val),
            "accuracy": float((calibrated.argmax(1) == y_val).mean()),
        },
        "_y_val": y_val,
        "_val_prob": val_prob,
        "_calibrated": calibrated,
    }


def _length_strata(tag: str, seed: int) -> dict[str, Any]:
    with np.load(E08_SUITE / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
        train_lengths = archive["train_features"][:, 0]
        val_lengths = archive["validation_features"][:, 0]
    edges = np.quantile(train_lengths, [1 / 3, 2 / 3])
    suite = SUITES[tag]
    metrics: dict[str, Any] = {"edges": edges.tolist(), "by_method": {}}
    for method in ("stack_tcn", "tcn_full"):
        _, y_true, _, _ = _read_predictions(
            suite / f"seed{seed}" / "conditions" / tag / method / "predictions.csv"
        )
        # Predictions are in the same validation order as E08 features.
        strata = [
            val_lengths <= edges[0],
            (val_lengths > edges[0]) & (val_lengths <= edges[1]),
            val_lengths > edges[1],
        ]
        metrics["by_method"][method] = []
        for mask in strata:
            metrics["by_method"][method].append(
                {
                    "n": int(mask.sum()),
                    "macro_f1": float(f1_score(y_true[mask], _read_predictions(
                        suite / f"seed{seed}" / "conditions" / tag / method / "predictions.csv"
                    )[2][mask], average="macro")),
                }
            )
    return metrics


def _mcnemar(tag: str, seed: int) -> dict[str, Any]:
    suite = SUITES[tag]
    _, y_true, stack_pred, _ = _read_predictions(
        suite / f"seed{seed}" / "conditions" / tag / "stack_tcn" / "predictions.csv"
    )
    _, _, tcn_pred, _ = _read_predictions(
        suite / f"seed{seed}" / "conditions" / tag / "tcn_full" / "predictions.csv"
    )
    stack_correct = stack_pred == y_true
    tcn_correct = tcn_pred == y_true
    b = int(np.sum(stack_correct & ~tcn_correct))
    c = int(np.sum(~stack_correct & tcn_correct))
    p_value = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {
        "seed": seed,
        "stack_fixes": b,
        "stack_breaks": c,
        "both_correct": int(np.sum(stack_correct & tcn_correct)),
        "both_wrong": int(np.sum(~stack_correct & ~tcn_correct)),
        "exact_binomial_p": p_value,
    }


def _battery_curve(tag: str, seed: int) -> dict[str, Any]:
    suite = SUITES[tag]
    _, y_true, _, probabilities = _read_predictions(
        suite / f"seed{seed}" / "conditions" / tag / "stack_tcn" / "predictions.csv"
    )
    positive = y_true == BATTERY
    scores = probabilities[:, BATTERY]
    order = np.argsort(-scores)
    total_positive = int(positive.sum())
    curve = []
    for rate in (0.05, 0.10, 0.20, 0.30, 0.50):
        count = int(round(rate * len(y_true)))
        selected = order[:count]
        selected_positive = positive[selected]
        tp = int(selected_positive.sum())
        fp = count - tp
        curve.append(
            {
                "alert_rate": rate,
                "coverage": count / len(y_true),
                "recall": tp / total_positive,
                "fpr": fp / int((~positive).sum()),
                "precision": tp / max(count, 1),
            }
        )
    return {"seed": seed, "curve": curve}


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for stage in ("before", "after"):
        for metric in ("nll", "brier", "ece", "accuracy"):
            array = np.asarray([record[stage][metric] for record in records])
            values[f"{stage}_{metric}_mean"] = float(array.mean())
            values[f"{stage}_{metric}_sd"] = float(array.std(ddof=1))
    values["temperature_mean"] = float(np.mean([record["temperature"] for record in records]))
    return values


def _plot_reliability(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tcn_moment.paper_style import apply_paper_style

    apply_paper_style(plt)
    bins = np.linspace(0.0, 1.0, 16)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for axis, stage, color, label in (
        (axes[0], "_val_prob", "#D55E00", "before"),
        (axes[1], "_calibrated", "#0072B2", "after temperature scaling"),
    ):
        accuracies = np.full(len(centers), np.nan)
        confidences = np.full(len(centers), np.nan)
        for index in range(len(centers)):
            correct = []
            confidence = []
            for record in records:
                probabilities = record[stage]
                values = probabilities.max(1)
                prediction = probabilities.argmax(1)
                mask = (values > bins[index]) & (values <= bins[index + 1])
                if mask.any():
                    correct.extend((prediction[mask] == record["_y_val"][mask]).tolist())
                    confidence.extend(values[mask].tolist())
            if correct:
                accuracies[index] = float(np.mean(correct))
                confidences[index] = float(np.mean(confidence))
        axis.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        axis.plot(centers, accuracies, marker="o", color=color, label=label)
        axis.set_xlabel("Mean confidence")
        axis.set_ylabel("Empirical accuracy")
        axis.set_title(label)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_risk_coverage(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tcn_moment.paper_style import apply_paper_style

    apply_paper_style(plt)
    coverages = np.linspace(0.05, 1.0, 20)
    accuracies = []
    battery_recall = []
    battery_fpr = []
    for coverage in coverages:
        correct = []
        recalls = []
        fprs = []
        for record in records:
            probabilities = record["_val_prob"]
            y = record["_y_val"]
            confidence = probabilities.max(1)
            order = np.argsort(-confidence)
            count = max(int(round(coverage * len(y))), 1)
            selected = order[:count]
            correct.append(float((probabilities[selected].argmax(1) == y[selected]).mean()))
            positive = y == BATTERY
            score_order = np.argsort(-probabilities[:, BATTERY])[:count]
            tp = int((positive[score_order]).sum())
            fp = count - tp
            recalls.append(tp / max(int(positive.sum()), 1))
            fprs.append(fp / max(int((~positive).sum()), 1))
        accuracies.append(float(np.mean(correct)))
        battery_recall.append(float(np.mean(recalls)))
        battery_fpr.append(float(np.mean(fprs)))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(coverages, accuracies, marker="o", color="#009E73")
    axes[0].set_xlabel("Coverage (fraction auto-decided)")
    axes[0].set_ylabel("Macro accuracy on accepted")
    axes[0].set_title("Selective prediction")
    axes[1].plot(coverages, battery_recall, marker="o", color="#D55E00", label="Recall")
    axes[1].plot(coverages, battery_fpr, marker="o", color="#0072B2", label="FPR")
    axes[1].set_xlabel("Alert rate (top battery scores)")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Battery alarm risk-coverage")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _markdown(tag: str, records: list[dict[str, Any]], aggregation: dict[str, Any], mcnemar: list[dict[str, Any]], strata: dict[str, Any], battery: list[dict[str, Any]]) -> str:
    lines = [
        f"# E10：E12 最终 stack 的校准、错误互补与电池风险（{tag}，validation-only）",
        "",
        "温度缩放只在 OOF 概率上拟合，validation 只评一次；没有 test。",
        "",
        "## 校准（四 seed 均值 ± SD）",
        "",
        "| 指标 | 校准前 | 温度缩放后 |",
        "|---|---:|---:|",
    ]
    for metric in ("nll", "brier", "ece", "accuracy"):
        lines.append(
            f"| {metric.upper()} | {aggregation[f'before_{metric}_mean']:.4f} ± {aggregation[f'before_{metric}_sd']:.4f} | "
            f"{aggregation[f'after_{metric}_mean']:.4f} ± {aggregation[f'after_{metric}_sd']:.4f} |"
        )
    lines += [
        "",
        f"- 平均温度：{aggregation['temperature_mean']:.3f}；ECE 为 15 等宽置信度分箱。",
        "",
        "## McNemar（stack vs TCN full）",
        "",
        "| Seed | stack 修对 | stack 弄错 | 都对 | 都错 | 精确二项 p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in mcnemar:
        lines.append(
            f"| {record['seed']} | {record['stack_fixes']} | {record['stack_breaks']} | "
            f"{record['both_correct']} | {record['both_wrong']} | {record['exact_binomial_p']:.3g} |"
        )
    lines += ["", "## 长度分层 Macro-F1（train 三分位边界）", "",
              "| Seed | 边界 | 方法 | ≤低 | 中 | >高 |", "|---:|---|---|---:|---:|---:|"]
    for seed, record in strata.items():
        for method, values in record["by_method"].items():
            lines.append(
                f"| {seed} | {record['edges'][0]:.0f}/{record['edges'][1]:.0f} | {method} | "
                f"{values[0]['macro_f1']:.2f} | {values[1]['macro_f1']:.2f} | {values[2]['macro_f1']:.2f} |"
            )
    lines += ["", "## 电池风险—覆盖率（stack，四 seed 均值）", "",
              "| Alert rate | Recall | FPR | Precision |", "|---:|---:|---:|---:|"]
    for index in range(len(battery[0]["curve"])):
        rates = [record["curve"][index] for record in battery]
        lines.append(
            f"| {rates[0]['alert_rate']*100:.0f}% | {100*np.mean([r['recall'] for r in rates]):.2f} | "
            f"{100*np.mean([r['fpr'] for r in rates]):.2f} | {100*np.mean([r['precision'] for r in rates]):.2f} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- OOF 来自 5 折交叉拟合，validation 曾用于早前的模型/预算选择；温度参数只拟合 OOF，但 OOF 与 validation 不完全同分布。",
        "- ECE/Brier 是描述性校准指标，不构成概率可信保证；风险—覆盖率曲线不是保形预测的覆盖保证。",
        "- 四 seed split 重叠，McNemar 的 p 值按 seed 报告，不合并为独立样本总体推断。",
        "- 没有 test；没有做幅值/噪声/缺失/前缀鲁棒性。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["40p", "10p"])
    parser.add_argument("--output-dir", default="artifacts/improvements/e10_calibration_safety")
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"test_evaluated": False, "tags": {}}
    for tag in args.tags:
        records = [_calibration_for_seed(tag, seed) for seed in SEEDS]
        aggregation = _aggregate(records)
        mcnemar = [_mcnemar(tag, seed) for seed in SEEDS]
        strata = {str(seed): _length_strata(tag, seed) for seed in SEEDS}
        battery = [_battery_curve(tag, seed) for seed in SEEDS]
        tag_dir = output_root / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        _plot_reliability(records, tag_dir / "reliability")
        _plot_risk_coverage(records, tag_dir / "risk_coverage")
        markdown = _markdown(tag, records, aggregation, mcnemar, strata, battery)
        (tag_dir / "summary.md").write_text(markdown + "\n", encoding="utf-8")
        summary["tags"][tag] = {
            "calibration": aggregation,
            "per_seed": [
                {key: value for key, value in record.items() if not key.startswith("_")}
                for record in records
            ],
            "mcnemar": mcnemar,
            "length_strata": strata,
            "battery_curve": battery,
        }
        print(markdown)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
