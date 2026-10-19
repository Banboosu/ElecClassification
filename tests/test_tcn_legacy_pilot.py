"""Scientific controls for E05B; runs with stdlib unittest (no pytest dependency)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from tcn_moment.tcn_legacy_pilot import (
    CONDITIONS, LegacyTCN, build_model, load_train_validation, protocol,
    state_hash, synthetic_check, DEFAULT,
)
from tcn_moment.training_utils import (
    seed_everything, save_training_checkpoint, resume_training_checkpoint,
)


class TCNLegacyPilotTests(unittest.TestCase):
    def test_causality_padding_and_pair_initialization(self):
        self.assertEqual(synthetic_check()["paired_initialization"], "passed")
        for a, b in ((0, 1), (2, 3)):
            seed_everything(torch, 43)
            left = state_hash(build_model(CONDITIONS[a]))
            seed_everything(torch, 43)
            right = state_hash(build_model(CONDITIONS[b]))
            self.assertEqual(left, right)

    def test_padding_pooling_contrast(self):
        model = LegacyTCN(masked=True)
        model.encoder = nn.Identity()
        model.classifier = nn.Linear(1, 3, bias=False)
        nn.init.ones_(model.classifier.weight)
        x = torch.tensor([[[2., 4., 10., 20.]]])
        mask = torch.tensor([[1., 1., 0., 0.]])
        torch.testing.assert_close(model(x, mask), torch.full((1, 3), 3.))
        model.masked = False
        torch.testing.assert_close(model(x, mask), torch.full((1, 3), 9.))

    def test_heldout_sequences_are_never_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = {"splits": {"train": ["a", "b", "c"],
                                   "validation": ["d", "e", "f"], "test": ["heldout"]}}
            (directory / "split.json").write_text(json.dumps(manifest))
            rows = [{"charging_station_id": sid, "InsertedColumn": str(i % 3),
                     "charging_powers_str": ",".join([str(i + 1)] * 18)}
                    for i, sid in enumerate("abcdef")]
            rows.append({"charging_station_id": "heldout", "InsertedColumn": "forbidden",
                         "charging_powers_str": "DO_NOT_PARSE"})
            pd.DataFrame(rows).to_csv(directory / "data.csv", sep=" ", index=False)
            p = {"split_manifest": str(directory / "split.json"),
                 "dataset": str(directory / "data.csv"), "train_count": 3,
                 "validation_count": 3, "max_length": 32, "classes": ["0", "1", "2"]}
            from tcn_moment.data import parse_power_sequence
            with patch("tcn_moment.tcn_legacy_pilot.parse_power_sequence", wraps=parse_power_sequence) as parse:
                data = load_train_validation(p)
            self.assertEqual(set(data), {"train", "validation"})
            self.assertEqual(parse.call_count, 6)
            self.assertTrue(all("DO_NOT_PARSE" not in c.args for c in parse.call_args_list))
            self.assertEqual(data["train"]["x"].shape, (3, 32))
            manifest["splits"]["test"] = ["a"]
            (directory / "split.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                load_train_validation(p)

    def test_test_and_expansion_are_rejected(self):
        p = json.loads(DEFAULT.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "protocol.json"
            for key in ("test_evaluated", "automatic_expansion", "automatic_shutdown"):
                file.write_text(json.dumps({**p, key: True}))
                with self.assertRaises(ValueError):
                    protocol(file)

    def test_checkpoint_restores_dropout_optimizer_and_shuffle(self):
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as tmp:
            seed_everything(torch, 43)
            model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(.3), nn.Linear(8, 3))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            generator = torch.Generator().manual_seed(43)
            x = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
            y = torch.tensor([0, 1, 2, 0, 1, 2])
            def step(net, opt, gen):
                for batch in torch.randperm(6, generator=gen).split(3):
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(net(x[batch]), y[batch])
                    loss.backward()
                    opt.step()
            step(model, optimizer, generator)
            scheduler.step(.8)
            file = Path(tmp) / "checkpoint.pt"
            save_training_checkpoint(torch=torch, path=file, epoch=1, model=model,
                                     optimizer=optimizer, history=[{"epoch": 1}],
                                     best_macro_f1=.8, epochs_without_improvement=0,
                                     data_generator=generator, scheduler=scheduler,
                                     metadata={"scope": "validation_only"})
            step(model, optimizer, generator)
            expected = state_hash(model)
            restored = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(.3), nn.Linear(8, 3))
            opt = torch.optim.AdamW(restored.parameters(), lr=.001)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
            gen = torch.Generator()
            start, _, _, _ = resume_training_checkpoint(
                torch=torch, path=file, model=restored, optimizer=opt, data_generator=gen,
                device=torch.device("cpu"), scheduler=sch,
                expected_metadata={"scope": "validation_only"})
            self.assertEqual(start, 2)
            step(restored, opt, gen)
            self.assertEqual(state_hash(restored), expected)


if __name__ == "__main__":
    unittest.main()
