from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcn_moment.config import load_config
from tcn_moment.train_moment import (
    build_model,
    forward_logits,
    require_torch_and_moment,
    select_device,
    set_num_classes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load MOMENT and run one forward-pass smoke test.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    parser.add_argument("--num-classes", type=int, default=3)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    torch, _, _, _, moment_pipeline = require_torch_and_moment()
    device = select_device(torch, config.training.device)
    model = build_model(config, moment_pipeline, args.num_classes)
    set_num_classes(model, args.num_classes)
    model.init()
    model.to(device)
    model.eval()

    sequence = torch.zeros((1, 1, config.data.max_length), dtype=torch.float32, device=device)
    input_mask = torch.ones((1, config.data.max_length), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = forward_logits(model, sequence, input_mask)
    result = {
        "status": "ok",
        "model_id": config.model.model_id,
        "model_config": str(config.model.config_path),
        "device": str(device),
        "logits_shape": list(logits.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
