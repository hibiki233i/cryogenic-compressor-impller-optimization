from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


GEOMETRY_FAILURE_STAGES = frozenset({"geometry", "mesh", "geometry_mesh"})
OPERATING_FAILURE_STAGES = frozenset(
    {"fatal_overflow", "blockage", "physical_invalid"}
)
NUMERICAL_FAILURE_STAGES = frozenset({"cfx_solver", "residual_unconverged"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _int_or_zero(value) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _sample_dict(variable_names: Sequence[str], sample) -> dict[str, float]:
    if isinstance(sample, Mapping):
        return {name: float(sample[name]) for name in variable_names}
    values = np.asarray(sample, dtype=float).reshape(-1)
    if len(values) != len(variable_names):
        raise ValueError(
            f"Expected {len(variable_names)} failure-point values, got {len(values)}."
        )
    return {name: float(value) for name, value in zip(variable_names, values)}


def load_run_outcome(
    csv_path: str | Path,
    *,
    source: str,
    run_id: str,
) -> dict | None:
    """Return the latest normalized audit record for one run, if present."""
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        records = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    if records.empty or not {"source", "run_id"}.issubset(records.columns):
        return None
    matched = records[
        (records["source"].astype(str) == str(source))
        & (records["run_id"].astype(str) == str(run_id))
    ]
    if matched.empty:
        return None
    record = matched.iloc[-1].to_dict()
    record["status"] = str(record.get("status", "")).strip().lower()
    record["failure_stage"] = str(
        record.get("failure_stage", "")
    ).strip().lower()
    record["confirmed"] = str(record.get("confirmed", "")).strip().lower() in {
        "1", "true", "yes"
    }
    return record


def record_run_outcome(
    csv_path: str | Path,
    variable_names: Sequence[str],
    sample,
    *,
    source: str,
    run_id: str,
    status: str,
    failure_stage: str = "",
    reason: str = "",
    confirmed: bool | None = None,
) -> dict:
    """Upsert the latest outcome for one DOE or active-learning run.

    Repeated attempts increment ``attempt_count``. Non-deterministic failures
    are only confirmed after the same stage fails twice consecutively;
    deterministic fatal-overflow/blockage/physical-invalid points are confirmed
    immediately.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = str(status).strip().lower()
    if status not in {"failed", "succeeded"}:
        raise ValueError(f"Unsupported run outcome status: {status}")

    if path.exists():
        try:
            existing = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    previous_attempts = 0
    previous_status = ""
    previous_stage = ""
    previous_consecutive_failures = 0
    if not existing.empty and {"source", "run_id"}.issubset(existing.columns):
        matched = existing[
            (existing["source"].astype(str) == str(source))
            & (existing["run_id"].astype(str) == str(run_id))
        ]
        if not matched.empty:
            previous = matched.iloc[-1]
            previous_attempts = _int_or_zero(previous.get("attempt_count", 0))
            previous_status = str(previous.get("status", "")).lower()
            previous_stage = str(previous.get("failure_stage", "")).lower()
            previous_consecutive_failures = _int_or_zero(
                previous.get("consecutive_failure_count", 0)
            )

    attempt_count = previous_attempts + 1
    stage = str(failure_stage or "").strip().lower()
    if status == "succeeded":
        stage = ""
        is_confirmed = False
        consecutive_failures = 0
    elif confirmed is not None:
        is_confirmed = bool(confirmed)
        consecutive_failures = (
            previous_consecutive_failures + 1
            if previous_status == "failed" and previous_stage == stage
            else 1
        )
    else:
        consecutive_failures = (
            previous_consecutive_failures + 1
            if previous_status == "failed" and previous_stage == stage
            else 1
        )
        is_confirmed = (
            stage in {"fatal_overflow", "blockage", "physical_invalid"}
            or consecutive_failures >= 2
        )

    record = {
        "source": str(source),
        "run_id": str(run_id),
        "status": status,
        "failure_stage": stage,
        "reason": str(reason or ""),
        "attempt_count": int(attempt_count),
        "consecutive_failure_count": int(consecutive_failures),
        "confirmed": bool(is_confirmed),
        "updated_at_utc": _timestamp(),
        **_sample_dict(variable_names, sample),
    }
    merged = pd.concat([existing, pd.DataFrame([record])], ignore_index=True, sort=False)
    if {"source", "run_id"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["source", "run_id"], keep="last")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    merged.to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    return record


def load_failure_training_sets(
    csv_path: str | Path,
    variable_names: Sequence[str],
    geometry_variable_names: Sequence[str],
    sources: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return design-intrinsic geometry and operating failures.

    Repeated numerical solver failures remain in the returned audit table but
    are intentionally excluded from classifier training: repetition confirms
    reproducibility, not that the design rather than solver setup caused it.
    """
    path = Path(csv_path)
    empty_geom = np.empty((0, len(geometry_variable_names)), dtype=float)
    empty_full = np.empty((0, len(variable_names)), dtype=float)
    if not path.exists():
        return empty_geom, empty_full, pd.DataFrame()

    try:
        records = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return empty_geom, empty_full, pd.DataFrame()
    if sources is not None:
        records = (records[records["source"].astype(str).str.lower().isin(sources)].copy()
                   if "source" in records else records.iloc[:0].copy())
    required = {
        "status",
        "failure_stage",
        "confirmed",
        *variable_names,
    }
    if not required.issubset(records.columns):
        return empty_geom, empty_full, records

    confirmed = records["confirmed"].astype(str).str.lower().isin({"1", "true", "yes"})
    failed = records[
        (records["status"].astype(str).str.lower() == "failed") & confirmed
    ].copy()
    for name in variable_names:
        failed[name] = pd.to_numeric(failed[name], errors="coerce")
    failed = failed.dropna(subset=list(variable_names))

    succeeded = records[
        records["status"].astype(str).str.lower() == "succeeded"
    ].copy()
    for name in variable_names:
        succeeded[name] = pd.to_numeric(succeeded[name], errors="coerce")
    succeeded = succeeded.dropna(subset=list(variable_names))

    stage = failed["failure_stage"].astype(str).str.lower()
    geometry_rows = failed[stage.isin(GEOMETRY_FAILURE_STAGES)]
    operating_rows = failed[stage.isin(OPERATING_FAILURE_STAGES)]
    if not succeeded.empty:
        successful_geometry = {
            tuple(row)
            for row in succeeded[list(geometry_variable_names)].to_numpy(dtype=float)
        }
        successful_operating = {
            tuple(row)
            for row in succeeded[list(variable_names)].to_numpy(dtype=float)
        }
        geometry_rows = geometry_rows.loc[
            np.asarray([
                tuple(row) not in successful_geometry
                for row in geometry_rows[list(geometry_variable_names)].to_numpy(dtype=float)
            ], dtype=bool)
        ]
        operating_rows = operating_rows.loc[
            np.asarray([
                tuple(row) not in successful_operating
                for row in operating_rows[list(variable_names)].to_numpy(dtype=float)
            ], dtype=bool)
        ]
    geometry_rows = geometry_rows.drop_duplicates(
        subset=list(geometry_variable_names), keep="last"
    )
    operating_rows = operating_rows.drop_duplicates(
        subset=list(variable_names), keep="last"
    )
    return (
        geometry_rows[list(geometry_variable_names)].to_numpy(dtype=float),
        operating_rows[list(variable_names)].to_numpy(dtype=float),
        failed,
    )


def classify_cfx_failure_stage(reason: str) -> str:
    """Separate physical/solver failures from setup and post-processing faults."""
    message = str(reason or "").lower()
    if any(
        token in message
        for token in ("fatal overflow", "floating point overflow", "overflow error")
    ):
        return "fatal_overflow"
    if "residual_unconverged" in message:
        return "residual_unconverged"
    if "residual_unavailable" in message or "convergence" in message:
        return "cfx_solver"
    if "100%堵塞" in message or "blockage" in message:
        return "blockage"
    if any(
        token in message
        for token in ("计算发散", "求解器异常", "未生成任何 .res", "solver diverged")
    ):
        return "cfx_solver"
    if "cfx-post" in message or "结果文件" in message or "postprocess" in message:
        return "postprocess"
    if "cfx-pre" in message or ".def" in message:
        return "solver_setup"
    return "infrastructure"


def classify_geometry_failure_stage(reason: str, diagnostic: str = "") -> str:
    """Keep environment/licensing faults out of the geometry negative class."""
    message = f"{reason}\n{diagnostic}".lower()
    infrastructure_tokens = (
        "license",
        "licensing",
        "flexlm",
        "checkout failed",
        "not found",
        "cannot find",
        "no such file",
        "access denied",
        "permission denied",
        "找不到",
        "不存在",
        "拒绝访问",
    )
    if any(token in message for token in infrastructure_tokens):
        return "infrastructure"
    # Without the PowerShell stage file, a non-zero process exit is ambiguous.
    # Fail closed for labeling: record it, but do not teach it as bad geometry.
    return "infrastructure"


def load_structured_failure_status(work_dir: str | Path) -> dict | None:
    path = Path(work_dir) / "failure_status.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return None
    stage = str(payload.get("stage", "")).strip().lower()
    if stage not in {"geometry", "mesh", "infrastructure"}:
        return None
    return {
        "stage": stage,
        "reason": str(payload.get("reason", "")),
    }
