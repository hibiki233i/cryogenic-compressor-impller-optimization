from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class CategoricalNBlScaler:
    """Scale continuous inputs and one-hot encode the discrete blade count."""

    def __init__(
        self,
        variable_names: Sequence[str],
        nbl_categories: Sequence[int],
    ):
        self.variable_names = tuple(variable_names)
        self.nbl_categories = np.asarray(tuple(nbl_categories), dtype=int)
        if "nBl" not in self.variable_names:
            raise ValueError("CategoricalNBlScaler requires an nBl input.")
        if len(self.nbl_categories) == 0:
            raise ValueError("At least one nBl category is required.")
        self.nbl_index = self.variable_names.index("nBl")
        self.continuous_indices = np.array(
            [i for i in range(len(self.variable_names)) if i != self.nbl_index],
            dtype=int,
        )
        self.continuous_scaler = MinMaxScaler()
        self.n_features_in_ = len(self.variable_names)
        self.n_features_out_ = len(self.continuous_indices) + len(self.nbl_categories)

    def fit(self, X: np.ndarray):
        values = self._validate(X)
        self.continuous_scaler.fit(values[:, self.continuous_indices])
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = self._validate(X)
        continuous = self.continuous_scaler.transform(
            values[:, self.continuous_indices]
        )
        nbl = np.rint(values[:, self.nbl_index]).astype(int)
        known = np.isin(nbl, self.nbl_categories)
        if not np.all(known):
            raise ValueError(
                f"Unknown nBl categories: {sorted(set(nbl[~known].tolist()))}"
            )
        one_hot = (nbl[:, None] == self.nbl_categories[None, :]).astype(float)
        return np.column_stack([continuous, one_hot])

    def get_feature_names_out(self) -> np.ndarray:
        continuous_names = [
            self.variable_names[i] for i in self.continuous_indices
        ]
        category_names = [f"nBl={value}" for value in self.nbl_categories]
        return np.asarray(continuous_names + category_names, dtype=object)

    def _validate(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != len(self.variable_names):
            raise ValueError(
                f"Expected {len(self.variable_names)} raw inputs, got {values.shape[1]}."
            )
        return values


def make_categorical_nbl_scaler(
    variable_names: Sequence[str],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> CategoricalNBlScaler:
    nbl_index = list(variable_names).index("nBl")
    categories = range(
        int(np.ceil(float(lower_bounds[nbl_index]))),
        int(np.floor(float(upper_bounds[nbl_index]))) + 1,
    )
    return CategoricalNBlScaler(variable_names, categories)
