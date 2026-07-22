from __future__ import annotations

import hashlib
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from tcn_moment.experiment import _git_state


class GitStateTests(unittest.TestCase):
    @patch("tcn_moment.experiment.subprocess.run")
    def test_dirty_worktree_records_status_and_diff_hash(self, run: MagicMock) -> None:
        diff = b"diff --git a/example.py b/example.py\n"
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=b"abc123\n", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=b" M example.py\n", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=diff, stderr=b""),
        ]

        metadata, actual_diff = _git_state()

        self.assertEqual(metadata["git_commit"], "abc123")
        self.assertTrue(metadata["git_dirty"])
        self.assertEqual(metadata["git_status"], [" M example.py"])
        self.assertEqual(metadata["git_diff_sha256"], hashlib.sha256(diff).hexdigest())
        self.assertEqual(actual_diff, diff)

    @patch("tcn_moment.experiment.subprocess.run")
    def test_clean_worktree_does_not_create_a_patch(self, run: MagicMock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=b"abc123\n", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""),
        ]

        metadata, diff = _git_state()

        self.assertFalse(metadata["git_dirty"])
        self.assertIsNone(metadata["git_diff_sha256"])
        self.assertIsNone(diff)


if __name__ == "__main__":
    unittest.main()
