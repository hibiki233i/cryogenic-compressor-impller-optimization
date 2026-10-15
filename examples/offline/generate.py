"""Synthetic analytic schema-v2 data. These values are not CFD observations."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from design_variables import (
    DEFAULT_VARIABLE_SPECS, lower_bounds, upper_bounds, variable_names,
    write_performance_data_metadata,
)
from geometry_constraints import explicit_geometry_safe_mask


def generate_training_data(path: Path, samples: int = 384, seed: int = 42) -> pd.DataFrame:
    specs = DEFAULT_VARIABLE_SPECS
    names = variable_names(specs)
    lower, upper = lower_bounds(specs), upper_bounds(specs)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    batches = []
    count = 0
    while count < samples:
        values = qmc.scale(sampler.random(samples), lower, upper)
        for index, spec in enumerate(specs):
            values[:, index] = np.round(values[:, index], spec["decimals"])
        values[:, names.index("P_out")] = 12.0
        values = values[explicit_geometry_safe_mask(values, names, lower, upper)]
        batches.append(values)
        count += len(values)
    values = np.concatenate(batches)[:samples]
    z = (values - lower) / (upper - lower)
    smooth_loss = np.mean((z[:, 1:11] - 0.5) ** 2, axis=1)
    frame = pd.DataFrame(values, columns=names)
    frame["Efficiency"] = 0.68 + 0.10*z[:, 0] - 0.025*smooth_loss + 0.005*z[:, 12]
    frame["PressureRatio"] = 1.05 + 0.20*z[:, 4]
    frame["Power"] = 80.0 + 25.0*z[:, 4] + 8.0*z[:, 11]
    frame["MassFlow"] = 3.40 + 0.90*z[:, 5] + 0.30*z[:, 11]
    frame["totalpressureratio"] = 1.35 + 0.55*(1.0-z[:, 0]) + 0.15*z[:, 4]
    frame["is_boundary"] = (frame["MassFlow"] < 3.6).astype(int)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_performance_data_metadata(path)
    return frame
