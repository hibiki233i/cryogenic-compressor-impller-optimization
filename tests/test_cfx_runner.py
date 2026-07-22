from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cfx_runner import solver_exit_failure_message


class CfxRunnerFailureTests(unittest.TestCase):
    def test_solver_exit_preserves_fatal_overflow_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "case_001.out"
            out_file.write_text(
                "Solver stopped after a FATAL OVERFLOW in the equation solver.",
                encoding="utf-8",
            )
            message = solver_exit_failure_message(str(out_file), 2)
            self.assertIn("FATAL OVERFLOW", message)

    def test_generic_solver_exit_is_not_promoted_to_fatal_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "case_001.out"
            out_file.write_text("Solver process stopped.", encoding="utf-8")
            message = solver_exit_failure_message(str(out_file), 2)
            self.assertNotIn("FATAL OVERFLOW", message)
            self.assertIn("计算发散或崩溃", message)


if __name__ == "__main__":
    unittest.main()
