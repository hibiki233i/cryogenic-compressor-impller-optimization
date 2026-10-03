"""Read-only case inventory, bounded file previews and CFX evidence inspection."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re

import pandas as pd

from cfx_convergence import inspect_out, latest_pair
from cfx_runner import is_current_cfx_result, RESULT_METADATA_FILENAME
from design_variables import load_variable_specs, variable_names
from ..config import AppConfig
from .analysis import stable_csv
from .live_results import _json, _stamp

TEXT_SUFFIXES = {".txt", ".log", ".out", ".err", ".lst", ".trn", ".json", ".csv", ".ccl", ".xml", ".tse", ".ps1"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def safe_child(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    child = (root / relative).resolve()
    if child == root or not child.is_relative_to(root):
        raise ValueError("Path is outside the selected case")
    return child


def _run_name(value) -> str:
    if value is None or pd.isna(value):
        raise ValueError("Invalid run ID")
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", text):
        raise ValueError("Invalid run ID")
    return text


def _clean(value):
    return "" if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)) else str(value)


@dataclass
class CaseRecord:
    source: str
    run_id: str
    path: Path
    status: str = "unknown"
    failure_stage: str = ""
    reason: str = ""
    attempts: int | None = None
    updated: str = ""
    exists: bool = False
    parameters: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    workspace_match: bool = True

    @property
    def key(self):
        return self.source, self.run_id, str(self.path)


@dataclass
class CaseInventory:
    records: list[CaseRecord] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    roots: dict[str, str] = field(default_factory=dict)


@dataclass
class CaseDetail:
    record: CaseRecord
    parameters: dict = field(default_factory=dict)
    parameter_source: str = ""
    metrics: dict = field(default_factory=dict)
    result_state: str = "missing"
    effective_blades: int | None = None
    convergence: dict = field(default_factory=dict)
    documents: dict = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class FilePreview:
    path: str
    kind: str
    text: str = ""
    data: bytes = b""
    truncated: bool = False


def preview_file(case_dir: Path, relative: str) -> FilePreview:
    path = safe_child(case_dir, relative)
    size = path.stat().st_size
    if path.suffix.lower() in IMAGE_SUFFIXES:
        if size > 8 * 1024 * 1024:
            raise ValueError("Image exceeds the 8 MiB preview limit")
        with path.open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Image grew beyond the preview limit")
        return FilePreview(str(path), "image", data=data)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return FilePreview(str(path), "binary")
    limit = 256 * 1024
    with path.open("rb") as stream:
        stream.seek(max(0, size - limit))
        raw = stream.read(limit)
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030", errors="replace")
    lines = text.splitlines()
    return FilePreview(str(path), "text", text="\n".join(lines[-800:]), truncated=size > limit or len(lines) > 800)


class CaseReader:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()
        self.names = variable_names(load_variable_specs(self.config.workspace.design_variables_json))

    def inventory(self) -> CaseInventory:
        ws = self.config.workspace
        result = CaseInventory()
        audits = pd.DataFrame()
        queries = pd.DataFrame()
        for path, kind in ((ws.failure_records_csv, "audit"), (ws.al_query_validation_csv, "query")):
            try:
                frame = stable_csv(path)
                if "run_id" not in frame:
                    raise ValueError("Missing run_id")
                if kind == "audit":
                    if not {"source", "status"}.issubset(frame):
                        raise ValueError("Missing source/status")
                    audits = frame.drop_duplicates(["source", "run_id"], keep="last")
                else:
                    queries = frame.drop_duplicates("run_id", keep="last")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                result.issues.append(f"{path}: {exc}")
        records = {}
        for source, configured, nested, prefix in (
            ("doe", ws.doe_runs_dir, "Runs", "Run_"),
            ("active_learning", ws.active_learning_runs_dir, "ActiveLearning_Runs", "AL_"),
        ):
            root = configured.resolve()
            if (root / nested).is_dir() and not any(p.is_dir() for p in root.glob(prefix + "*")):
                root = (root / nested).resolve()
                result.issues.append(f"Nested case directory: {root}")
            result.roots[source] = str(root)
            # A configured case root may point at a different experiment. Never
            # attach the current workspace's same-named IDs to that experiment.
            associate = True
            own_stage_path = ws.checkpoint_meta_json.parent / "al_stage.json"
            for marker in (root / "al_stage.json", root.parent / "al_stage.json"):
                if marker.exists() and marker.resolve() != own_stage_path.resolve():
                    try:
                        own_stage = _json(own_stage_path).get("stage_id") if own_stage_path.exists() else None
                        associate = _json(marker).get("stage_id") == own_stage and own_stage is not None
                    except (OSError, ValueError, TypeError):
                        associate = False
                    if not associate:
                        result.issues.append(f"Different stage at {root}; audit associations disabled. Open that stage's configuration to inspect its records.")
                        break
            try:
                if not root.is_dir():
                    result.issues.append(f"Case directory missing: {root}")
                else:
                    count = 0
                    for child in root.iterdir():
                        if not child.name.startswith(prefix) or not child.is_dir():
                            continue
                        child = safe_child(root, child.name)
                        record = CaseRecord(source, _run_name(child.name), child, exists=True, workspace_match=associate)
                        records[(source, child.name)] = record
                        count += 1
                        if count >= 10000:
                            result.issues.append("Case scan capped at 10,000 directories per source")
                            break
            except (OSError, ValueError) as exc:
                result.issues.append(f"{root}: {exc}")
            if not associate:
                continue
            for kind, frame in (("query", queries if source == "active_learning" else pd.DataFrame()), ("audit", audits)):
                if kind == "audit" and not frame.empty:
                    frame = frame.loc[frame.source.astype(str) == source]
                for _, row in frame.iterrows():
                    try:
                        run_id = _run_name(row.run_id)
                        path = safe_child(root, run_id)
                        record = records.setdefault((source, run_id), CaseRecord(source, run_id, path, exists=path.is_dir()))
                        if kind == "audit":
                            record.audit = {k: _clean(v) for k, v in row.to_dict().items()}
                        updated = (_clean(row.get("updated_at_utc")) or _clean(row.get("completed_at_utc"))
                                   or _clean(row.get("submitted_at_utc")))
                        previous_time = pd.to_datetime(record.updated, errors="coerce", utc=True)
                        incoming_time = pd.to_datetime(updated, errors="coerce", utc=True)
                        if pd.notna(previous_time) and pd.notna(incoming_time) and incoming_time < previous_time:
                            continue
                        status = _clean(row.get("status")).lower()
                        # Audit is read after query and wins on completed outcomes.
                        record.status = {"success": "succeeded"}.get(status, status) or "unknown"
                        record.failure_stage = _clean(row.get("failure_stage")) if record.status == "failed" else ""
                        record.reason = _clean(row.get("reason", row.get("failure_reason")))
                        record.updated = updated
                        attempt = row.get("attempt_count", row.get("evaluation_attempts"))
                        if pd.notna(attempt):
                            record.attempts = int(attempt)
                        for name in self.names:
                            value = float(row.get(name, float("nan")))
                            if math.isfinite(value):
                                record.parameters[name] = value
                    except (OSError, ValueError, TypeError, OverflowError) as exc:
                        result.issues.append(f"{source}/{row.get('run_id')}: {exc}")
        result.records = sorted(records.values(), key=lambda r: (r.source, r.run_id))
        return result

    def detail(self, record: CaseRecord) -> CaseDetail:
        detail = CaseDetail(record, parameters=dict(record.parameters), parameter_source="audit")
        root = record.path.resolve()
        if not root.is_dir():
            detail.issues.append(f"Case directory missing: {root}")
            return detail
        for name in ("input_parameters.json", "geometry_summary.json", "failure_status.json", RESULT_METADATA_FILENAME, "CFX_INVALID.json", "DOE_INVALID.json"):
            path = safe_child(root, name)
            if not path.exists():
                continue
            try:
                payload = _json(path)
                detail.documents[name] = payload
                if name == "input_parameters.json":
                    detail.parameters = {key: float(payload[key]) for key in self.names}
                    if not all(math.isfinite(v) for v in detail.parameters.values()):
                        raise ValueError("Nonfinite input parameter")
                    detail.parameter_source = name
            except (OSError, ValueError, TypeError, KeyError) as exc:
                detail.issues.append(f"{name}: {exc}")
                if name == "input_parameters.json":
                    detail.parameters = {}
        try:
            out, _ = latest_pair(root)
            if out:
                safe_child(root, out.name)
                detail.convergence = inspect_out(out, self.config.runtime.cfx_residual_threshold)
            result = safe_child(root, "CFX_Results.txt")
            if result.exists():
                detail.result_state = "unverified"
                before = _stamp(result)
                if is_current_cfx_result(result, self.config.runtime.cfx_residual_threshold):
                    nbl = detail.parameters.get("nBl")
                    if nbl is None or not math.isfinite(nbl) or nbl <= 0:
                        raise ValueError("Blade count unavailable; full-impeller flow/power cannot be interpreted")
                    # Historical DOE inputs retain the unsnapped LHS coordinate;
                    # both geometry and CFX runners execute int(round(nBl)).
                    nbl = int(round(nbl))
                    if nbl < 1:
                        raise ValueError("Invalid executed blade count")
                    detail.effective_blades = nbl
                    if result.stat().st_size > 4096:
                        raise ValueError("CFX result text is larger than a five-value performance record")
                    values = [float(x) for x in result.read_text(encoding="utf-8-sig").strip().split(",")]
                    if len(values) != 5 or not all(math.isfinite(x) for x in values):
                        raise ValueError("Invalid CFX result values")
                    # CFX-Post stores one passage's power/flow; use the same
                    # full-impeller conversion as the external runner.
                    detail.metrics = dict(zip(("Efficiency", "PressureRatio", "Power", "MassFlow", "totalpressureratio"),
                                              (values[0], values[1], values[2] * nbl, values[3] * nbl, values[4])))
                    if before != _stamp(result):
                        raise ValueError("CFX results changed during reading")
                    detail.result_state = "verified"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            detail.metrics = {}
            detail.result_state = "unverified"
            detail.issues.append(str(exc))
        count = 0
        try:
            for directory, directories, files in os.walk(root, followlinks=False):
                parent = Path(directory)
                depth = len(parent.relative_to(root).parts)
                allowed = []
                for name in directories:
                    try:
                        candidate = safe_child(root, (parent / name).relative_to(root))
                        if not candidate.is_symlink() and depth < 4:
                            allowed.append(name)
                    except (OSError, ValueError):
                        detail.issues.append(f"External directory skipped: {name}")
                directories[:] = allowed
                for name in files:
                    rel = (parent / name).relative_to(root)
                    try:
                        path = safe_child(root, rel)
                        stat = path.stat()
                        detail.files.append({"file": str(rel), "bytes": stat.st_size, "modified": stat.st_mtime})
                    except (OSError, ValueError) as exc:
                        detail.issues.append(str(exc))
                    count += 1
                    if count >= 2000:
                        detail.issues.append("File inventory capped at 2,000 files / 5 directory levels")
                        return detail
        except OSError as exc:
            detail.issues.append(str(exc))
        return detail
