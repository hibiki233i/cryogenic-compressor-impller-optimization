"""Small OUT/RES fixture used by the offline recovery regression test."""
from pathlib import Path


def make_pair(directory):
    root = Path(directory)
    out = root / "result.out"
    out.write_text(
        "OUTER LOOP ITERATION = 10\n"
        "| Equation | Rate | RMS Res | Max Res | Linear | State |\n"
        + "".join(
            f"| {equation} | 0.95 | 1e-5 | 1e-1 | 2 | OK |\n"
            for equation in ("U-Mom", "V-Mom", "W-Mom", "P-Mass", "H-Energy", "K-TurbKE", "O-TurbFreq")
        )
        + "This run of the ANSYS CFX Solver has finished.\n",
        encoding="utf-8",
    )
    res = out.with_suffix(".res")
    res.write_bytes(b"offline-test-fixture")
    return out, res
