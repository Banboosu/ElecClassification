"""E03F-v2: online frozen encoder compatibility, preserving the failed v1 freeze."""

from __future__ import annotations

import argparse
import copy
import json
import signal
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from tcn_moment import strong_head_lora as original
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.moment_peft import PeftClassifier
from tcn_moment.train_moment import forward_features

DEFAULT_PROTOCOL = "configs/experiments/improvements/e03f_strong_head_lora_online_protocol.json"
OriginalMakeModel = original.make_model
OriginalTrainingProtocol = original.training_protocol


class OnlineFrozenClassifier(PeftClassifier):
    """A frozen online feature provider is NOT an adapter-bearing trainable backbone.

    The old loop uses .backbone for adapter witnesses/optimizer roles. Register the
    online encoder separately so those roles are empty, while forward still runs
    the exact same public encoder and gradients still train the shared head.
    """

    def __init__(self, encoder, classifier):
        if any(p.requires_grad for p in encoder.parameters()):
            raise ValueError("Online feature provider must be completely frozen.")
        super().__init__(None, classifier)
        self.online_frozen_encoder = encoder

    def forward(self, x, statistics, mask=None):
        features = forward_features(self.online_frozen_encoder, x, mask).float()
        return self.classifier(features, statistics)

    def train_selected(self):
        self.train()
        self.online_frozen_encoder.eval()
        self.classifier.train()


def training_protocol(p):
    pp = OriginalTrainingProtocol(p)
    # Same physical64/effective128 as declared for EVERY treatment. The generic
    # loop chooses frozen_head_batch_size when there is no trainable backbone.
    pp["training"]["frozen_head_batch_size"] = p["training"]["physical_batch_size"]
    return pp


def make_model(p, branch, rank):
    model, parent = OriginalMakeModel(p, branch, rank)
    if not rank:
        model = OnlineFrozenClassifier(model.backbone, model.classifier)
    return model, parent


@contextmanager
def online_hooks():
    with ExitStack() as stack:
        stack.enter_context(patch.object(original, "make_model", make_model))
        stack.enter_context(patch.object(original, "training_protocol", training_protocol))
        yield


def read_protocol(path):
    p = original.read_protocol(path)
    if p.get("execution_revision") != "v2_online_frozen_provider" or not p["suite"].endswith("_v2"):
        raise ValueError("Separate v2 suite/protocol required.")
    return p


def freeze(path):
    p = read_protocol(path)
    root = original.plan_directory(p)
    if (root / "frozen_protocol.json").exists():
        return verify(path)
    failed = original.verify(Path(original.DEFAULT_PROTOCOL))
    failed_root = original.plan_directory(failed["protocol"])
    status = json.loads((failed_root / "suite_status.json").read_text())
    failed_run = original.run_directory(failed["protocol"], "frozen_fusion")
    if status["status"] != "failed" or json.loads((failed_run / "history.json").read_text()):
        raise ValueError("Expected preserved v1 failure before any committed epoch.")
    smoke_report = json.loads(Path(p["smoke_report"]).read_text())
    if smoke_report.get("online_revision_source_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Synthetic smoke must exercise this exact compatibility layer.")
    inputs = [
        Path(__file__),
        Path("tests/test_strong_head_lora_online.py"),
        Path("scripts/run_e03f_online_remote.sh"),
        Path("docs/experiment_records/e03f_online_interface_revision_20261019.md"),
        Path(original.DEFAULT_PROTOCOL),
        failed_root / "frozen_protocol.json",
        failed_root / "suite_status.json",
        failed_run / "history.json",
        failed_run / "status.json",
        failed_run / "checkpoint_last.pt",
    ]
    # Inherited freeze validates the same exact synthetic smoke/core sources.
    # Seal this compatibility layer as extra inputs before the command returns.
    f = original.freeze(path)
    f = copy.deepcopy(f)
    f["file_sha256"].update({str(k): sha256_file(k) for k in inputs})
    f["execution_revision"] = p["execution_revision"]
    f["original_failure"] = (
        "Online rank0 encoder erroneously entered an adapter-gradient witness; zero committed epochs; no scientific change."
    )
    atomic_write_json(root / "frozen_protocol.json", f)
    return f


def verify(path):
    p = read_protocol(path)
    f = original.verify(path)
    if f.get("execution_revision") != p["execution_revision"]:
        raise ValueError("Compatibility-layer freeze incomplete.")
    return f


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Interrupted; resume committed epoch.")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("smoke", "freeze", "verify", "prepare", "run", "summarize")
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--condition")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = Path(args.protocol)
    read_protocol(path)
    with online_hooks():
        if args.action == "freeze":
            freeze(path)
        elif args.action == "verify":
            verify(path)
        elif args.action == "run":
            verify(path)
            original.run(path, args.condition, args.resume)
        elif args.action in {"prepare", "summarize"}:
            verify(path)
            getattr(original, args.action)(path)
        else:
            original.smoke(path)
            p = read_protocol(path)
            report = json.loads(Path(p["smoke_report"]).read_text())
            report["online_revision_source_sha256"] = sha256_file(Path(__file__))
            atomic_write_json(Path(p["smoke_report"]), report)


if __name__ == "__main__":
    main()
