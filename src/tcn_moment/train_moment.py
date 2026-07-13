from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from tcn_moment.config import ExperimentConfig, load_config
from tcn_moment.data import load_dataset, save_label_encoder
from tcn_moment.metrics import classification_metrics


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


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    seed_everything(torch, config.data.random_state)

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
    x_val = torch.tensor(bundle.x_val[:, None, :], dtype=torch.float32)
    y_val = torch.tensor(bundle.y_val, dtype=torch.long)
    x_test = torch.tensor(bundle.x_test[:, None, :], dtype=torch.float32)
    y_test = torch.tensor(bundle.y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=config.training.batch_size,
        shuffle=False,
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
    test_metrics: dict[str, Any] | None = None
    class_names = [str(name) for name in bundle.label_encoder.classes_.tolist()]
    best_macro_f1 = -1.0
    best_path = output_dir / "moment_classifier_best.pt"
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

            metrics = evaluate(torch, model, val_loader, loss_fn, device, class_names)
            metrics["train_loss"] = train_loss / len(train_loader.dataset)
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            print(
                f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                torch.save(model.state_dict(), best_path)
            save_checkpoint(torch, model, optimizer, output_dir, epoch, history)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Keeping the latest completed epoch checkpoint.")

    if history:
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate(
            torch,
            model,
            test_loader,
            loss_fn,
            device,
            class_names,
            include_details=True,
        )
        print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
        torch.save(model.state_dict(), output_dir / "moment_classifier.pt")

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": "MOMENT",
                "split": {
                    "train": len(bundle.y_train),
                    "validation": len(bundle.y_val),
                    "test": len(bundle.y_test),
                    "random_state": config.data.random_state,
                },
                "history": history,
                "best_validation_macro_f1": best_macro_f1,
                "test_metrics": test_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
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


def evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: Any,
    class_names: list[str],
    *,
    include_details: bool = False,
) -> dict[str, Any]:
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

    result = classification_metrics(
        y_true,
        y_pred,
        class_names,
        include_details=include_details,
    )
    result["loss"] = total_loss / len(loader.dataset)
    if not include_details:
        result["val_loss"] = result.pop("loss")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MOMENT classifier on charging power data.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    train(load_config(Path(args.config)))


if __name__ == "__main__":
    main()
