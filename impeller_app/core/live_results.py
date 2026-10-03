"""Read-only, schema-checked snapshots for the desktop results monitor.

Never launches a solver, repairs a file, or treats surrogate predictions as CFD.
The snapshot reader runs outside the GUI thread and rejects changing files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from design_variables import (
    PERFORMANCE_DATA_SCHEMA_VERSION, TOTAL_PRESSURE_DEFINITION,
    load_variable_specs, variable_names, performance_data_metadata_path,
)
from ..config import AppConfig

DISPLAY_COLUMNS = ("Efficiency", "totalpressureratio", "MassFlow", "Power", "P_out", "nBl", "is_boundary")
MAX_FILE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ResultIssue:
    source: str
    code: str
    detail: str = ""


@dataclass
class SampleSnapshot:
    source: str
    modified: float | None = None
    count: int | None = None
    excluded: int = 0
    duplicates: int = 0
    best_efficiency: float | None = None
    best_pressure_ratio: float | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)


@dataclass
class ResultsSnapshot:
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    samples: dict[str, SampleSnapshot] = field(default_factory=dict)
    hv: list[tuple[int, float | None, float | None]] = field(default_factory=list)
    hv_source: str = ""
    hv_modified: float | None = None
    stage: str = ""
    issues: list[ResultIssue] = field(default_factory=list)


def _stamp(path: Path) -> tuple | None:
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 64 MiB monitor limit")
    return content


def _json(path: Path) -> dict:
    payload = json.loads(_read_bytes(path).decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _csv(path: Path) -> pd.DataFrame:
    raw = _read_bytes(path)
    if not raw or not raw.endswith((b"\n", b"\r")):
        raise ValueError("Empty or incomplete CSV write; waiting for a complete file")
    return pd.read_csv(BytesIO(raw))


class LiveResultsReader:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()

    def read(self) -> ResultsSnapshot:
        cfg = self.config
        ws = cfg.workspace
        snapshot = ResultsSnapshot(hv_source=str(ws.hv_history_csv))
        for name, path in (("doe", ws.training_csv), ("pool", ws.pool_checkpoint_csv)):
            snapshot.samples[name] = SampleSnapshot(source=str(path))
            try:
                snapshot.samples[name] = self._samples(path)
            except FileNotFoundError as exc:
                snapshot.issues.append(ResultIssue(name, "missing", str(exc)))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                snapshot.issues.append(ResultIssue(name, "unavailable", str(exc)))
        try:
            self._history(snapshot)
        except FileNotFoundError as exc:
            snapshot.issues.append(ResultIssue("hv", "missing", str(exc)))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            snapshot.issues.append(ResultIssue("hv", "unavailable", str(exc)))
        return snapshot

    def performance_frame(self, path: Path) -> tuple[pd.DataFrame, int, int, float]:
        """Return validated, deduplicated observations for display and analysis."""
        metadata = performance_data_metadata_path(path)
        before = (_stamp(path), _stamp(metadata))
        meta = _json(metadata)
        if (meta.get("schema_version") != PERFORMANCE_DATA_SCHEMA_VERSION
                or meta.get("total_pressure_definition") != TOTAL_PRESSURE_DEFINITION):
            raise ValueError("Performance schema/total-pressure definition mismatch")
        frame = _csv(path)
        names = variable_names(load_variable_specs(self.config.workspace.design_variables_json))
        required = list(dict.fromkeys(names + list(DISPLAY_COLUMNS)))
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        numeric = frame[required].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric.to_numpy()).all(axis=1)
        runtime = self.config.runtime
        valid &= (numeric.P_out - runtime.optimization_outlet_static_pressure_pa).abs() <= runtime.operating_point_pressure_tolerance_pa
        # Display boundary cases explicitly; do not invent a new feasibility rule.
        valid &= numeric.is_boundary.isin([0, 1])
        filtered = numeric.loc[valid]
        accepted = filtered.drop_duplicates(subset=names, keep="last")
        modified = path.stat().st_mtime
        if before != (_stamp(path), _stamp(metadata)):
            raise ValueError("Files changed while reading; retrying on the next refresh")
        return accepted, len(frame) - len(filtered), len(filtered) - len(accepted), modified

    def _samples(self, path: Path) -> SampleSnapshot:
        accepted, excluded, duplicates, modified = self.performance_frame(path)
        result = SampleSnapshot(
            source=str(path), modified=modified, count=len(accepted),
            excluded=excluded, duplicates=duplicates,
        )
        if len(accepted):
            result.best_efficiency = float(accepted.Efficiency.max())
            result.best_pressure_ratio = float(accepted.totalpressureratio.max())
            # Bounds rendering cost only; summary values use every accepted row.
            chart_rows = accepted.iloc[np.linspace(0, len(accepted) - 1, min(5000, len(accepted)), dtype=int)]
            result.points = list(chart_rows[["Efficiency", "totalpressureratio"]].itertuples(index=False, name=None))
            result.rows = list(accepted[list(DISPLAY_COLUMNS)].tail(200).iloc[::-1].itertuples(index=False, name=None))
        return result

    def _history(self, snapshot: ResultsSnapshot) -> None:
        ws = self.config.workspace
        stage_path = ws.checkpoint_meta_json.parent / "al_stage.json"
        paths = (ws.hv_history_csv, ws.checkpoint_meta_json, stage_path)
        before = tuple(_stamp(p) for p in paths)
        meta = _json(ws.checkpoint_meta_json)
        stage = _json(stage_path)
        if (meta.get("performance_data_schema_version") != PERFORMANCE_DATA_SCHEMA_VERSION
                or meta.get("total_pressure_definition") != TOTAL_PRESSURE_DEFINITION):
            raise ValueError("HV checkpoint has incompatible performance metadata")
        runtime = self.config.runtime
        saved = stage.get("runtime", {})
        for key in ("optimization_outlet_static_pressure_pa", "operating_point_pressure_tolerance_pa"):
            if saved.get(key) != getattr(runtime, key):
                raise ValueError("HV stage operating condition differs from the monitored configuration")
        stage_id = stage.get("stage_id")
        policy = stage.get("hv_policy_version")
        if not stage_id or meta.get("stage_id") != stage_id or policy is None or meta.get("hv_policy_version") != policy:
            raise ValueError("HV stage/checkpoint provenance mismatch")
        frame = _csv(ws.hv_history_csv)
        required = {"iter", "true_hv", "surrogate_hv", "stage_id", "hv_policy_version"}
        if not required.issubset(frame.columns):
            raise ValueError("HV columns/provenance missing")
        if not frame.stage_id.eq(stage_id).all() or not frame.hv_policy_version.eq(policy).all():
            raise ValueError("Mixed HV stages or policies; refusing to join their curves")
        iterations = pd.to_numeric(frame["iter"], errors="coerce")
        if (not np.isfinite(iterations).all() or iterations.duplicated().any()
                or (iterations < 0).any() or (iterations % 1 != 0).any()):
            raise ValueError("Invalid or duplicate HV iteration")
        # A history can be written just before its checkpoint. Publish committed rounds only.
        frame = frame.loc[iterations <= int(meta["completed_iters"])].copy()
        frame["iter"] = iterations.loc[frame.index]
        history = []
        for _, row in frame.sort_values("iter").iterrows():
            values = []
            for column in ("true_hv", "surrogate_hv"):
                value = row[column]
                if pd.isna(value):
                    values.append(None)
                else:
                    value = float(value)
                    if not np.isfinite(value) or value < 0:
                        raise ValueError("Invalid HV value")
                    values.append(value)
            history.append((int(row["iter"]), *values))
        if before != tuple(_stamp(p) for p in paths):
            raise ValueError("HV files changed while reading; retrying on the next refresh")
        snapshot.hv = history
        snapshot.stage = str(stage_id)
        snapshot.hv_modified = ws.hv_history_csv.stat().st_mtime
