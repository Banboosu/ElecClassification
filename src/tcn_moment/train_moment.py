from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

from tcn_moment.config import ExperimentConfig, load_config
from tcn_moment.data import load_dataset, save_label_encoder


def require_torch_and_moment() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm
        from momentfm import MOMENTPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing MOMENT training dependencies. Install them with:\n"
            "  uv sync --extra moment\n"
        ) from exc
    return torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline


def select_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: ExperimentConfig, moment_pipeline: Any, num_classes: int) -> Any:
    model = moment_pipeline.from_pretrained(
        config.model.model_id,
        model_kwargs={
            "task_name": "classification",
            "seq_len": config.data.max_length,
            "n_channels": config.model.num_channels,
            # MOMENT examples use num_class; some downstream wrappers use num_classes.
            "num_class": num_classes,
            "num_classes": num_classes,
            "freeze_embedder": config.model.freeze_backbone,
            "freeze_encoder": config.model.freeze_backbone,
            "enable_gradient_checkpointing": not config.model.freeze_backbone,
        },
    )
    return model


def set_num_classes(model: Any, num_classes: int) -> None:
    # MOMENT versions have used both num_class and num_classes in examples/configs.
    if hasattr(model, "num_class"):
        model.num_class = num_classes
    if hasattr(model, "num_classes"):
        model.num_classes = num_classes
    if hasattr(model, "config"):
        if hasattr(model.config, "num_class"):
            model.config.num_class = num_classes
        if hasattr(model.config, "num_classes"):
            model.config.num_classes = num_classes


def forward_logits(model: Any, batch_x: Any) -> Any:
    output = model(x_enc=batch_x)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output


def train(config: ExperimentConfig) -> None:
    torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()

    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(config.data)
    save_label_encoder(bundle.label_encoder, output_dir)

    device = select_device(torch, config.training.device)
    model = build_model(config, MOMENTPipeline, bundle.num_classes)
    set_num_classes(model, bundle.num_classes)
    model.init()
    model.to(device)

    if config.model.freeze_backbone:
        for name, parameter in model.named_parameters():
            if "head" not in name and "classification" not in name:
                parameter.requires_grad = False

    x_train = torch.tensor(bundle.x_train[:, None, :], dtype=torch.float32)
    y_train = torch.tensor(bundle.y_train, dtype=torch.long)
    x_test = torch.tensor(bundle.x_test[:, None, :], dtype=torch.float32)
    y_test = torch.tensor(bundle.y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    report: dict[str, Any] | None = None
    try:
        for epoch in range(1, config.training.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = forward_logits(model, batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.detach().cpu()) * len(batch_y)

            metrics = evaluate(torch, model, test_loader, loss_fn, device)
            metrics["train_loss"] = train_loss / len(train_loader.dataset)
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            print(
                f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f}"
            )
            save_checkpoint(torch, model, optimizer, output_dir, epoch, history)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Keeping the latest completed epoch checkpoint.")

    if history:
        report = final_report(torch, model, test_loader, device, bundle.label_encoder.classes_.tolist())
        torch.save(model.state_dict(), output_dir / "moment_classifier.pt")

    (output_dir / "metrics.json").write_text(
        json.dumps({"history": history, "report": report}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved artifacts to {output_dir}")


def save_checkpoint(
    torch: Any,
    model: Any,
    optimizer: Any,
    output_dir: Path,
    epoch: int,
    history: list[dict[str, float]],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
    torch.save(model.state_dict(), output_dir / f"moment_classifier_epoch_{epoch}.pt")
    (output_dir / "metrics_partial.json").write_text(
        json.dumps({"history": history}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def evaluate(torch: Any, model: Any, loader: Any, loss_fn: Any, device: Any) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = forward_logits(model, batch_x)
            loss = loss_fn(logits, batch_y)
            total_loss += float(loss.detach().cpu()) * len(batch_y)
            y_true.extend(batch_y.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=1,
    )
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {
        "val_loss": total_loss / len(loader.dataset),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def final_report(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = forward_logits(model, batch_x.to(device))
            y_true.extend(batch_y.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=[str(name) for name in class_names],
        zero_division=1,
        output_dict=True,
    )
    print(classification_report(y_true, y_pred, target_names=[str(name) for name in class_names], zero_division=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MOMENT classifier on charging power data.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    train(load_config(Path(args.config)))


if __name__ == "__main__":
    main()
