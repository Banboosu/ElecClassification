from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from tcn_moment.config import ExperimentConfig, load_config
from tcn_moment.data import load_dataset, save_label_encoder
from tcn_moment.metrics import classification_metrics


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        self.causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        if self.causal_padding:
            return output[..., : -self.causal_padding]
        return output


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(inputs) + self.residual(inputs))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        return self.classifier(self.pool(features).squeeze(-1))


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
) -> DataLoader:
    tensors = TensorDataset(
        torch.tensor(x[:, None, :], dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
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
            logits = model(batch_x)
            total_loss += float(loss_fn(logits, batch_y).cpu()) * len(batch_y)
            y_true.extend(batch_y.cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

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


def train(config: ExperimentConfig) -> None:
    seed_everything(config.data.random_state)
    bundle = load_dataset(config.data)
    training = config.tcn_training
    output_dir = training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_label_encoder(bundle.label_encoder, output_dir)

    train_loader = make_loader(
        bundle.x_train,
        bundle.y_train,
        training.batch_size,
        training.num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        bundle.x_val,
        bundle.y_val,
        training.batch_size,
        training.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        bundle.x_test,
        bundle.y_test,
        training.batch_size,
        training.num_workers,
        shuffle=False,
    )

    device = select_device(training.device)
    model = TCNClassifier(
        bundle.num_classes,
        config.tcn_model.channels,
        config.tcn_model.kernel_size,
        config.tcn_model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    class_names = [str(name) for name in bundle.label_encoder.classes_.tolist()]
    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    best_path = output_dir / "tcn_classifier_best.pt"

    for epoch in range(1, training.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(batch_y)

        metrics = evaluate(model, val_loader, loss_fn, device, class_names)
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

        (output_dir / "metrics_partial.json").write_text(
            json.dumps({"history": history}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        class_names,
        include_details=True,
    )
    result = {
        "model": "TCN",
        "split": {
            "train": len(bundle.y_train),
            "validation": len(bundle.y_val),
            "test": len(bundle.y_test),
            "random_state": config.data.random_state,
        },
        "history": history,
        "best_validation_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"Saved artifacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN on the unified data protocol.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    train(load_config(Path(args.config)))


if __name__ == "__main__":
    main()
