from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskUpdate:
    status: str
    message: str
    progress: float | None = None
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    status: str
    message: str
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
