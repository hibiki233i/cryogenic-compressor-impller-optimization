#!/usr/bin/env python3
"""
Export selected Pareto-front points into per-case folders, with:
- geometry parameter files
- optional copied base .cft project file
- optional generated run_cfturbo.cft-batch file from a template

Typical usage:
  python3 pareto_export_cft_cases.py --top-n 5
  python3 pareto_export_cft_cases.py --front-indices 9,18,26
  python3 pareto_export_cft_cases.py --curve-fractions 0.2,0.5,0.8 --front-csv pareto_front_points.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from design_variables import variable_names

VAR_NAMES = variable_names()

DEFAULT_ENGINEERING_CSV = "pareto_engineering_ranked.csv"
DEFAULT_FRONT_CSV = "pareto_front_points.csv"
DEFAULT_OUTPUT_DIR = "pareto_cft_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-point case folders and optional .cft/.cft-batch files from Pareto front points.")
    parser.add_argument("--engineering-csv", default=DEFAULT_ENGINEERING_CSV, help="Engineering-ranked Pareto CSV.")
    parser.add_argument("--front-csv", default=DEFAULT_FRONT_CSV, help="Plain Pareto front CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--top-n", type=int, default=None, help="Export top N engineering-ranked points.")
    parser.add_argument("--front-indices", default=None, help="Comma-separated front_index values to export.")
    parser.add_argument("--curve-fractions", default=None, help="Comma-separated fractions in [0,1], mapped to nearest front point.")
    parser.add_argument("--case-prefix", default="ParetoCase", help="Per-folder name prefix.")
    parser.add_argument("--base-cft", default=None, help="Optional base .cft file to copy into each case folder.")
    parser.add_argument("--cft-batch-template", default=None, help="Optional CFturbo XML batch template to modify into run_cfturbo.cft-batch.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing case folders if they already exist.")
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(p)


def parse_csv_ints(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str | None) -> list[float]:
    if not raw:
        return []
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    for v in vals:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Curve fraction must be in [0,1], got {v}")
    return vals


def nearest_front_rows_by_fraction(front: pd.DataFrame, fractions: list[float]) -> list[pd.Series]:
    xy = front[["Efficiency", "totalpressureratio"]].to_numpy(dtype=float)
    if len(front) == 0:
        return []
    if len(front) == 1:
        return [front.iloc[0] for _ in fractions]

    seg = xy[1:] - xy[:-1]
    seg_len = ((seg ** 2).sum(axis=1)) ** 0.5
    total = float(seg_len.sum())
    if total <= 0.0:
        return [front.iloc[0] for _ in fractions]

    out = []
    cum = seg_len.cumsum()
    for frac in fractions:
        target = frac * total
        seg_idx = int((cum >= target).argmax())
        out.append(front.iloc[seg_idx])
    return out


def select_rows(args: argparse.Namespace, engineering_df: pd.DataFrame, front_df: pd.DataFrame) -> list[pd.Series]:
    selected: list[pd.Series] = []

    if args.top_n is not None:
        if "engineering_rank" not in engineering_df.columns:
            raise ValueError("engineering CSV must contain engineering_rank for --top-n")
        selected.extend([row for _, row in engineering_df.head(args.top_n).iterrows()])

    front_indices = parse_csv_ints(args.front_indices)
    if front_indices:
        for idx in front_indices:
            matched = engineering_df.loc[engineering_df["front_index"] == idx] if "front_index" in engineering_df.columns else pd.DataFrame()
            if len(matched) > 0:
                selected.append(matched.iloc[0])
                continue
            matched_front = front_df.loc[front_df["front_index"] == idx]
            if len(matched_front) == 0:
                raise ValueError(f"front_index {idx} not found in provided CSVs")
            selected.append(matched_front.iloc[0])

    curve_fractions = parse_csv_floats(args.curve_fractions)
    if curve_fractions:
        selected.extend(nearest_front_rows_by_fraction(front_df, curve_fractions))

    if not selected:
        if "engineering_rank" in engineering_df.columns and len(engineering_df) > 0:
            selected.append(engineering_df.iloc[0])
        elif len(front_df) > 0:
            selected.append(front_df.iloc[0])

    # de-duplicate by front_index if available, else by geometry tuple
    deduped: list[pd.Series] = []
    seen = set()
    for row in selected:
        if "front_index" in row.index:
            key = ("front_index", int(row["front_index"]))
        else:
            key = tuple(round(float(row[name]), 10) for name in VAR_NAMES)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def set_xml_text(root: ET.Element, xpath: str, value: str) -> None:
    node = root.find(xpath)
    if node is None:
        raise ValueError(f"Template XML missing required node: {xpath}")
    node.text = value


def create_cft_batch_from_template(template_path: str, dest_path: str, row: pd.Series) -> None:
    tree = ET.parse(template_path)
    root = tree.getroot()

    set_xml_text(root, ".//dS", f"{float(row['d1s']):.5f}")
    set_xml_text(root, ".//d2", f"{float(row['d2']):.5f}")
    set_xml_text(root, ".//b2", f"{float(row['b2']):.5f}")
    set_xml_text(root, ".//DeltaZ", f"{float(row['Lz']):.5f}")
    set_xml_text(root, ".//nBl", f"{int(round(float(row['nBl'])))}")
    set_xml_text(root, ".//dH", f"{float(row['dH']):.5f}")
    set_xml_text(root, ".//xTipInlet", f"{float(row['TipClear']):.5f}")
    set_xml_text(root, ".//xTipOutlet", f"{float(row['TipClear']):.5f}")
    set_xml_text(root, ".//sLEH", f"{float(row['t']):.5f}")
    set_xml_text(root, ".//sLES", f"{float(row['t']):.5f}")
    set_xml_text(root, ".//sTEH", f"{float(row['t']):.5f}")
    set_xml_text(root, ".//sTES", f"{float(row['t']):.5f}")

    beta2hb_rad = math.radians(float(row["beta2hb"]))
    beta2sb_rad = math.radians(float(row["beta2sb"]))
    rake_te_s_rad = math.radians(float(row["rake_te_s"]))
    beta1hb_rad = math.radians(float(row["beta1hb"]))
    beta1sb_rad = math.radians(float(row["beta1sb"]))

    set_xml_text(root, ".//Beta2/Value[@Index='0']", f"{beta2hb_rad:.6f}")
    set_xml_text(root, ".//Beta2/Value[@Index='1']", f"{beta2sb_rad:.6f}")
    set_xml_text(root, ".//RakeTE/Value[@Index='1']", f"{rake_te_s_rad:.6f}")
    set_xml_text(root, ".//Beta1/Value[@Index='0']", f"{beta1hb_rad:.6f}")
    set_xml_text(root, ".//Beta1/Value[@Index='1']", f"{beta1sb_rad:.6f}")

    # keep the same constants as Run-GeometryMeshing.ps1
    set_xml_text(root, ".//mFlow", f"{0.0036:.6f}")
    set_xml_text(root, ".//nRot", f"{(10000.0 / 60.0):.6f}")

    tree.write(dest_path, encoding="utf-8", xml_declaration=True)


def row_summary(row: pd.Series) -> dict:
    summary = {
        "front_index": int(row["front_index"]) if "front_index" in row.index else None,
        "engineering_rank": int(row["engineering_rank"]) if "engineering_rank" in row.index and pd.notna(row["engineering_rank"]) else None,
        "engineering_score": float(row["engineering_score"]) if "engineering_score" in row.index and pd.notna(row["engineering_score"]) else None,
        "geometry": {name: (int(round(float(row[name]))) if name == "nBl" else float(row[name])) for name in VAR_NAMES},
    }
    for col in ["Efficiency", "totalpressureratio", "Power", "MassFlow", "geom_margin", "knee_score", "stability_penalty"]:
        if col in row.index and pd.notna(row[col]):
            summary[col] = float(row[col])
    return summary


def write_case_files(case_dir: Path, row: pd.Series, base_cft: str | None, cft_batch_template: str | None) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)

    summary = row_summary(row)
    with open(case_dir / "case_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame([{name: summary["geometry"][name] for name in VAR_NAMES}]).to_csv(case_dir / "geometry_parameters.csv", index=False)

    if base_cft:
        shutil.copy2(base_cft, case_dir / Path(base_cft).name)

    if cft_batch_template:
        create_cft_batch_from_template(cft_batch_template, str(case_dir / "run_cfturbo.cft-batch"), row)

    return summary


def main() -> int:
    args = parse_args()

    engineering_df = load_csv(args.engineering_csv) if Path(args.engineering_csv).exists() else pd.DataFrame()
    front_df = load_csv(args.front_csv)

    if engineering_df.empty and front_df.empty:
        raise RuntimeError("No valid Pareto CSV input found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = select_rows(args, engineering_df, front_df)
    report = {
        "output_dir": str(output_dir.resolve()),
        "case_count": len(rows),
        "base_cft": args.base_cft,
        "cft_batch_template": args.cft_batch_template,
        "cases": [],
    }

    for i, row in enumerate(rows, start=1):
        front_index = int(row["front_index"]) if "front_index" in row.index else i
        engineering_rank = None
        if "engineering_rank" in row.index and pd.notna(row["engineering_rank"]):
            engineering_rank = int(row["engineering_rank"])

        parts = [args.case_prefix]
        if engineering_rank is not None:
            parts.append(f"R{engineering_rank:02d}")
        parts.append(f"F{front_index:02d}")
        case_name = "_".join(parts)
        case_dir = output_dir / case_name

        if case_dir.exists() and not args.force:
            raise FileExistsError(f"Case folder already exists: {case_dir}. Use --force to overwrite.")
        if case_dir.exists() and args.force:
            shutil.rmtree(case_dir)

        summary = write_case_files(case_dir, row, args.base_cft, args.cft_batch_template)
        summary["case_dir"] = str(case_dir.resolve())
        report["cases"].append(summary)

    with open(output_dir / "export_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Created {len(rows)} case folder(s) under: {output_dir}")
    for case in report["cases"]:
        print(f"- {case['case_dir']}")
    if args.base_cft:
        print(f"Copied base .cft from: {args.base_cft}")
    if args.cft_batch_template:
        print(f"Generated run_cfturbo.cft-batch from template: {args.cft_batch_template}")
    else:
        print("No cft-batch template supplied, so only parameter files were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
