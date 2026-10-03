"""Resolve a user-selected inspection configuration without changing task settings."""
from pathlib import Path

from ..config import AppConfig
from .live_results import _json


def load_review_config(path: Path) -> AppConfig:
    path = path.resolve()
    payload = _json(path)
    workspace = payload.get("workspace")
    if not isinstance(workspace, dict) or not workspace.get("project_root"):
        raise ValueError("Select an application/stage configuration containing workspace.project_root")
    root = Path(workspace["project_root"])
    if not root.is_absolute():
        workspace["project_root"] = str((path.parent / root).resolve())
    config = AppConfig.from_dict(payload).resolved()
    if not config.workspace.project_root.is_dir():
        raise ValueError(f"Workspace directory is not available: {config.workspace.project_root}")
    return config
