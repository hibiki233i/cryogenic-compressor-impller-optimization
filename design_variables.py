from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd


DESIGN_VARIABLES_ENV = "IMPELLER_DESIGN_VARIABLES_PATH"

DEFAULT_VARIABLE_SPECS = [
    {"name": "d1s", "lower": 0.34, "upper": 0.42, "decimals": 5, "is_integer": False, "role": "geometry"},
    {"name": "dH", "lower": 0.044, "upper": 0.056, "decimals": 5, "is_integer": False, "role": "geometry"},
    {"name": "beta1hb", "lower": 70.0, "upper": 84.0, "decimals": 3, "is_integer": False, "role": "geometry"},
    {"name": "beta1sb", "lower": 20.0, "upper": 35.0, "decimals": 3, "is_integer": False, "role": "geometry"},
    {"name": "d2", "lower": 0.45, "upper": 0.53, "decimals": 5, "is_integer": False, "role": "geometry"},
    {"name": "b2", "lower": 0.044, "upper": 0.056, "decimals": 5, "is_integer": False, "role": "geometry"},
    {"name": "beta2hb", "lower": 40.0, "upper": 54.0, "decimals": 3, "is_integer": False, "role": "geometry"},
    {"name": "beta2sb", "lower": 40.0, "upper": 55.0, "decimals": 3, "is_integer": False, "role": "geometry"},
    {"name": "Lz", "lower": 0.185, "upper": 0.215, "decimals": 5, "is_integer": False, "role": "geometry"},
    {"name": "t", "lower": 0.0015, "upper": 0.0035, "decimals": 6, "is_integer": False, "role": "geometry"},
    {"name": "TipClear", "lower": 0.0007, "upper": 0.0015, "decimals": 6, "is_integer": False, "role": "geometry"},
    {"name": "nBl", "lower": 9.0, "upper": 12.0, "decimals": 0, "is_integer": True, "role": "geometry"},
    {"name": "rake_te_s", "lower": -25.0, "upper": -15.0, "decimals": 3, "is_integer": False, "role": "geometry"},
    {"name": "P_out", "lower": 8.0, "upper": 13.0, "decimals": 3, "is_integer": False, "role": "operating_condition"},
]

OUTPUT_COLUMNS = ["Efficiency", "PressureRatio", "Power", "MassFlow", "totalpressureratio", "is_boundary"]
PERFORMANCE_DATA_SCHEMA_VERSION = 2
TOTAL_PRESSURE_DEFINITION = "stationary_frame_mass_flow_averaged_total_to_total"


def default_design_variables_path() -> Path:
    return Path(os.environ.get(DESIGN_VARIABLES_ENV, "design_variables.json"))


def _base_specs() -> list[dict]:
    return deepcopy(DEFAULT_VARIABLE_SPECS)


def load_variable_specs(config_path: str | Path | None = None) -> list[dict]:
    path = Path(config_path) if config_path is not None else default_design_variables_path()
    specs = _base_specs()
    if not path.exists():
        return specs

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_specs = payload.get("variables", [])
    by_name = {item["name"]: item for item in raw_specs if "name" in item}
    for spec in specs:
        override = by_name.get(spec["name"])
        if not override:
            continue
        if "lower" in override:
            spec["lower"] = float(override["lower"])
        if "upper" in override:
            spec["upper"] = float(override["upper"])
    validate_variable_specs(specs)
    return specs


def save_variable_specs(specs: list[dict], config_path: str | Path | None = None) -> Path:
    validate_variable_specs(specs)
    path = Path(config_path) if config_path is not None else default_design_variables_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variables": [
            {
                "name": spec["name"],
                "lower": float(spec["lower"]),
                "upper": float(spec["upper"]),
                "role": spec.get("role", "geometry"),
            }
            for spec in specs
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_variable_specs(specs: list[dict]) -> None:
    expected_names = [item["name"] for item in DEFAULT_VARIABLE_SPECS]
    provided_names = [item["name"] for item in specs]
    if provided_names != expected_names:
        raise ValueError(f"Design variable names must match {expected_names}, got {provided_names}")
    for spec in specs:
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {spec['name']}: lower={lower} must be < upper={upper}")


def variable_names(specs: list[dict] | None = None) -> list[str]:
    active_specs = specs or load_variable_specs()
    return [spec["name"] for spec in active_specs]


def geometry_variable_names(specs: list[dict] | None = None) -> list[str]:
    active_specs = specs or load_variable_specs()
    return [spec["name"] for spec in active_specs if spec.get("role") == "geometry"]


def operating_condition_names(specs: list[dict] | None = None) -> list[str]:
    active_specs = specs or load_variable_specs()
    return [
        spec["name"]
        for spec in active_specs
        if spec.get("role") == "operating_condition"
    ]


def lower_bounds(specs: list[dict] | None = None):
    import numpy as np

    active_specs = specs or load_variable_specs()
    return np.array([float(spec["lower"]) for spec in active_specs], dtype=float)


def upper_bounds(specs: list[dict] | None = None):
    import numpy as np

    active_specs = specs or load_variable_specs()
    return np.array([float(spec["upper"]) for spec in active_specs], dtype=float)


def training_csv_columns(specs: list[dict] | None = None) -> list[str]:
    return variable_names(specs) + OUTPUT_COLUMNS


def ensure_training_csv(path: str | Path, specs: list[dict] | None = None) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        pd.DataFrame(columns=training_csv_columns(specs)).to_csv(csv_path, index=False)
        write_performance_data_metadata(csv_path)
    return csv_path


def performance_data_metadata_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_suffix(path.suffix + ".meta.json")


def write_performance_data_metadata(csv_path: str | Path) -> Path:
    metadata_path = performance_data_metadata_path(csv_path)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": PERFORMANCE_DATA_SCHEMA_VERSION,
                "total_pressure_definition": TOTAL_PRESSURE_DEFINITION,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def is_current_performance_data(csv_path: str | Path) -> bool:
    metadata_path = performance_data_metadata_path(csv_path)
    if not Path(csv_path).exists() or not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        int(payload.get("schema_version", 0)) == PERFORMANCE_DATA_SCHEMA_VERSION
        and payload.get("total_pressure_definition") == TOTAL_PRESSURE_DEFINITION
    )


def require_current_performance_data(csv_path: str | Path) -> None:
    if is_current_performance_data(csv_path):
        return
    raise ValueError(
        f"{csv_path} uses the legacy or unknown total-pressure definition. "
        "Re-run CFX-Post recovery from the existing .res files before training "
        "or Pareto extraction."
    )
