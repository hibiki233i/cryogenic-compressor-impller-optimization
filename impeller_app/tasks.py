from __future__ import annotations

from .config import AppConfig
from .core import ActiveLearningService, ParetoService
from .runner import RunnerAPI


def start_doe(config: AppConfig, progress_callback=None):
    return RunnerAPI(config).run_doe_batch(progress_callback=progress_callback)


def start_active_learning(config: AppConfig, progress_callback=None):
    service = ActiveLearningService(config)
    return service.run_active_learning_iteration(
        additional_iters=config.runtime.active_learning_additional_iters,
        progress_callback=progress_callback,
    )


def query_pareto(config: AppConfig, selection: dict, progress_callback=None):
    if progress_callback:
        progress_callback("Computing Pareto front...")
    return ParetoService(config).query_front(selection)


def export_cases(config: AppConfig, top_n: int = 1, force: bool = False):
    service = ParetoService(config)
    service.compute_pareto_front()
    return service.export_cases(top_n=top_n, force=force)
