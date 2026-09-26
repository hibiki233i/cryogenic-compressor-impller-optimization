from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cfx_runner import (
    CfxBlockageParser,
    read_new_log_content,
    solver_exit_failure_message,
)


class CfxRunnerFailureTests(unittest.TestCase):
    def test_blockage_parser_requires_same_iteration_and_explicit_ratios(self):
        parser = CfxBlockageParser()
        statuses = parser.feed(
            "OUTER LOOP ITERATION = 21\n"
            "A wall has been placed at portion(s) of an INLET boundary condition\n"
            "100.0% of the faces, 100.0% of the area\n"
            "A wall has been placed at portion(s) of an OUTLET boundary condition\n"
            "100.0% of the faces, 100.0% of the area\n"
        )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0].iteration, 21)
        self.assertTrue(statuses[0].inlet_is_100_percent)
        self.assertTrue(statuses[0].outlet_is_100_percent)
        self.assertTrue(statuses[0].both_are_100_percent)

    def test_blockage_parser_does_not_mix_message_blocks_or_iterations(self):
        parser = CfxBlockageParser()
        statuses = parser.feed(
            "OUTER LOOP ITERATION = 7\n"
            "A wall has been placed at portion(s) of an INLET boundary condition\n"
            "100.0% of the faces, 100.0% of the area\n"
            "OUTER LOOP ITERATION = 8\n"
            "A wall has been placed at portion(s) of an OUTLET boundary condition\n"
            "100.0% of the faces, 100.0% of the area\n"
            "OUTER LOOP ITERATION = 9\n"
        )
        self.assertEqual([status.iteration for status in statuses], [7, 8])
        self.assertFalse(any(status.both_are_100_percent for status in statuses))

    def test_blockage_parser_does_not_take_ratio_from_another_warning_block(self):
        parser = CfxBlockageParser()
        statuses = parser.feed(
            "OUTER LOOP ITERATION = 4\n"
            "WARNING #100\n"
            "A wall has been placed at portion(s) of an INLET boundary condition\n"
            "WARNING #200\n"
            "100.0% of the faces, 100.0% of the area\n"
            "A wall has been placed at portion(s) of an OUTLET boundary condition\n"
            "100.0% of the faces, 100.0% of the area\n"
            "OUTER LOOP ITERATION = 5\n"
        )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0].iteration, 4)
        self.assertFalse(statuses[0].inlet_is_100_percent)
        self.assertTrue(statuses[0].outlet_is_100_percent)
        self.assertFalse(statuses[0].both_are_100_percent)

    def test_incremental_log_reader_returns_only_appended_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "case_001.out"
            out_file.write_text("old\n", encoding="utf-8")
            _, offset = read_new_log_content(str(out_file), out_file.stat().st_size)
            with out_file.open("a", encoding="utf-8") as stream:
                stream.write("new\n")
            content, offset = read_new_log_content(str(out_file), offset)
            self.assertEqual(content.replace("\r\n", "\n"), "new\n")
            self.assertEqual(offset, out_file.stat().st_size)

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
