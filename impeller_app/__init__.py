"""Desktop application package for the impeller optimization workflow."""

from .config import AppConfig
from .tasks import export_cases, query_pareto, run_nsga2_only, start_active_learning, start_doe

__all__ = [
    "AppConfig",
    "start_doe",
    "start_active_learning",
    "run_nsga2_only",
    "query_pareto",
    "export_cases",
]
