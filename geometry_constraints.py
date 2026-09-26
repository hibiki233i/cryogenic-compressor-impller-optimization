from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GeometryConstraintConfig:
    min_d2_d1s_gap: float = 0.070
    max_le_sweep_diff: float = 52.0
    max_exit_angle_diff: float = 13.5
    min_rake_te_s_by_nbl: Mapping[int, float] | None = None

    def rake_limits(self) -> Mapping[int, float]:
        return self.min_rake_te_s_by_nbl or {
            9: -18.0,
            10: -19.0,
            11: -20.0,
            12: -21.0,
        }


def _matrix(X: np.ndarray, variable_names: Sequence[str]) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(variable_names):
        raise ValueError(
            f"Expected {len(variable_names)} design values, got {values.shape[1]}."
        )
    return values


def geometry_rule_violations(
    X: np.ndarray,
    variable_names: Sequence[str],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: GeometryConstraintConfig | None = None,
) -> np.ndarray:
    """Return the shared DOE/active-learning hard-rule violations.

    Every column uses the convention ``violation <= 0``.  Keeping this logic in
    one module prevents DOE from evaluating geometries that active learning
    would reject before CFD.
    """
    cfg = config or GeometryConstraintConfig()
    values = _matrix(X, variable_names)
    index = {name: i for i, name in enumerate(variable_names)}
    required = {
        "d1s", "beta1hb", "beta1sb", "d2", "b2", "beta2hb",
        "beta2sb", "Lz", "t", "TipClear", "rake_te_s",
    }
    missing = required.difference(index)
    if missing:
        raise ValueError(f"Missing geometry variables: {sorted(missing)}")

    def column(name: str) -> np.ndarray:
        return values[:, index[name]]

    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    d1s = column("d1s")
    d2 = column("d2")
    beta1hb = column("beta1hb")
    beta1sb = column("beta1sb")
    beta2hb = column("beta2hb")
    beta2sb = column("beta2sb")
    b2 = column("b2")
    length_z = column("Lz")
    thickness = column("t")
    tip_clearance = column("TipClear")
    rake = column("rake_te_s")

    return np.column_stack([
        cfg.min_d2_d1s_gap - (d2 - d1s),
        lower[index["b2"]] - b2,
        thickness - upper[index["t"]],
        lower[index["TipClear"]] - tip_clearance,
        tip_clearance - upper[index["TipClear"]],
        (beta1hb - beta1sb) - cfg.max_le_sweep_diff,
        np.abs(beta2hb - beta2sb) - cfg.max_exit_angle_diff,
        lower[index["Lz"]] - length_z,
        length_z - upper[index["Lz"]],
        rake - upper[index["rake_te_s"]],
        lower[index["rake_te_s"]] - rake,
    ])


def explicit_geometry_safe_mask(
    X: np.ndarray,
    variable_names: Sequence[str],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: GeometryConstraintConfig | None = None,
) -> np.ndarray:
    return np.all(
        geometry_rule_violations(
            X, variable_names, lower_bounds, upper_bounds, config=config
        ) <= 0.0,
        axis=1,
    )


def overlap_proxy_violation(
    X: np.ndarray,
    variable_names: Sequence[str],
    config: GeometryConstraintConfig | None = None,
) -> np.ndarray:
    """Return the shared soft overlap-risk proxy in rake-angle degrees."""
    cfg = config or GeometryConstraintConfig()
    values = _matrix(X, variable_names)
    index = {name: i for i, name in enumerate(variable_names)}
    nbl = np.rint(values[:, index["nBl"]]).astype(int)
    rake = values[:, index["rake_te_s"]]
    limits = cfg.rake_limits()

    def limit_for(value: int) -> float:
        if value in limits:
            return float(limits[value])
        nearest = min(limits, key=lambda item: abs(int(item) - value))
        return float(limits[nearest])

    minimum_rake = np.array([limit_for(value) for value in nbl], dtype=float)
    return np.maximum(0.0, minimum_rake - rake)
