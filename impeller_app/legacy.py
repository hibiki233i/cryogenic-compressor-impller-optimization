from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _project_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()


def _load_module(module_name: str, file_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, ROOT / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy module: {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def active_learning_module() -> ModuleType:
    return _load_module("legacy_active_learning", "NN_NSGA2_ActiveLearning_refactored.py")


@lru_cache(maxsize=1)
def pareto_module() -> ModuleType:
    return _load_module("legacy_pareto_query", "pareto_front_query.py")


@lru_cache(maxsize=1)
def export_module() -> ModuleType:
    return _load_module("legacy_pareto_export", "pareto_export_cft_cases.py")
