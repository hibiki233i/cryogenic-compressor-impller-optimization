from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from ..config import AppConfig
from ..models import TaskResult, TaskUpdate

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
except ImportError:
    saltelli = None
    sobol = None

warnings.filterwarnings("ignore", category=UserWarning)

def _emit(progress_callback, message: str, progress=None, metrics=None):
    if progress_callback:
        progress_callback(TaskUpdate(status="running", message=message, progress=progress, metrics=metrics or {}))

class PerformanceSurrogate(nn.Module):
    def __init__(self, input_dim=14, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SobolService:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()
        from design_variables import load_variable_specs, variable_names, lower_bounds, upper_bounds
        self.variable_specs = load_variable_specs(self.config.workspace.design_variables_json)
        self.var_names = variable_names(self.variable_specs)
        self.l_bounds = lower_bounds(self.variable_specs)
        self.u_bounds = upper_bounds(self.variable_specs)
        
        self.nbl_idx = self.var_names.index('nBl')
        self.p_out_idx = self.var_names.index('P_out')
        self.output_names = ['Efficiency', 'totalpressureratio', 'MassFlow']

    def ensure_salib(self):
        if saltelli is None or sobol is None:
            raise ImportError("SALib is required for Sobol analysis. Please install it first.")

    def load_model_and_scalers(self, model_path: Path, scaler_x_path: Path, scaler_y_path: Path):
        model = PerformanceSurrogate(input_dim=len(self.var_names), output_dim=len(self.output_names))
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        if not scaler_x_path.exists() or not scaler_y_path.exists():
            raise FileNotFoundError("Scalers not found.")

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_x, scaler_y

    def run_analysis(self, progress_callback=None) -> TaskResult:
        try:
            self.ensure_salib()
            _emit(progress_callback, "Running Sobol analysis...")
            
            ws = self.config.workspace
            return self._run_single_core(
                self.config.workspace.training_csv,
                ws.best_regressor_pth,
                ws.scaler_x_pkl,
                ws.scaler_y_pkl,
                progress_callback
            )

        except Exception as exc:
            import traceback
            return TaskResult(status="failed", message=f"Sobol analysis failed: {str(exc)}", metrics={"traceback": traceback.format_exc()})

    def _run_single_core(self, csv_path: Path, model_path: Path, sx_path: Path, sy_path: Path, progress_callback=None) -> TaskResult:
        model, scaler_x, scaler_y = self.load_model_and_scalers(model_path, sx_path, sy_path)

        df = self._load_analysis_data(progress_callback, csv_path)
        fixed_nbl = self.config.runtime.sobol_fixed_nbl
        base_n = self.config.runtime.sobol_base_n
        tag = self.config.runtime.sobol_tag

        X_data = df[self.var_names].values.astype(float)
        sub_mask = (X_data[:, self.nbl_idx] == fixed_nbl)
        if not np.any(sub_mask):
            p_out_ref = float(np.median(X_data[:, self.p_out_idx]))
            _emit(progress_callback, f"Warning: No samples with nBl={fixed_nbl} found. Using global median P_out.")
        else:
            p_out_ref = float(np.median(X_data[sub_mask, self.p_out_idx]))

        # Build trust region (5% to 95%)
        cont_idx = [i for i in range(len(self.var_names)) if i not in (self.nbl_idx, self.p_out_idx)]
        cont_names = [self.var_names[i] for i in cont_idx]
        
        if np.any(sub_mask):
            X_sub = X_data[sub_mask]
        else:
            X_sub = X_data
            
        lower_trust = np.percentile(X_sub[:, cont_idx], 5, axis=0)
        upper_trust = np.percentile(X_sub[:, cont_idx], 95, axis=0)
        lower_trust = np.maximum(lower_trust, self.l_bounds[cont_idx])
        upper_trust = np.minimum(upper_trust, self.u_bounds[cont_idx])

        problem = {
            'num_vars': len(cont_names),
            'names': cont_names,
            'bounds': list(zip(lower_trust, upper_trust))
        }

        _emit(progress_callback, f"Generating {base_n} Saltelli samples...")
        X_cont = saltelli.sample(problem, base_n, calc_second_order=False)
        
        X_full = np.zeros((len(X_cont), len(self.var_names)), dtype=float)
        for j, idx in enumerate(cont_idx):
            X_full[:, idx] = X_cont[:, j]
        X_full[:, self.nbl_idx] = fixed_nbl
        X_full[:, self.p_out_idx] = p_out_ref
        
        _emit(progress_callback, f"Evaluating {len(X_full)} samples on surrogate...")
        X_norm = scaler_x.transform(X_full)
        with torch.no_grad():
            Y_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
        Y_pred = scaler_y.inverse_transform(Y_norm)

        results = {}
        for i, name in enumerate(['Efficiency', 'PressureRatio']):
            _emit(progress_callback, f"Calculating Sobol indices for {name}...")
            Si = sobol.analyze(problem, Y_pred[:, i], calc_second_order=False, print_to_console=False)
            res_df = pd.DataFrame({
                'Variable': cont_names,
                'S1': Si['S1'],
                'ST': Si['ST'],
                'Interaction': Si['ST'] - Si['S1']
            }).sort_values('ST', ascending=False)
            results[name] = res_df.to_dict(orient='records')
            
            csv_path_out = self.config.workspace.project_root / f"{tag}_sobol_{name}_nBl{fixed_nbl}.csv"
            res_df.to_csv(csv_path_out, index=False)

        return TaskResult(
            status="succeeded",
            message=f"Sobol analysis completed for nBl={fixed_nbl}.",
            metrics={"sobol_results": results},
            artifacts={"csv_files": [str(self.config.workspace.project_root / f"{tag}_sobol_{n}_nBl{fixed_nbl}.csv") for n in ['Efficiency', 'PressureRatio']]}
        )

    def _load_analysis_data(self, progress_callback, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"Training CSV not found at {csv_path}")

        df_main = pd.read_csv(csv_path)
        required_cols = self.var_names + self.output_names + ["is_boundary"]
        for col in required_cols:
            if col not in df_main.columns:
                raise ValueError(f"CSV missing required column: {col}")
        df_main = df_main[required_cols].copy()
        df_main["nBl"] = np.clip(np.round(df_main["nBl"]), 9, 12).astype(int)
        df_main = df_main.drop_duplicates(subset=self.var_names, keep="first").reset_index(drop=True)

        if not self.config.runtime.sobol_use_al_samples:
            _emit(progress_callback, f"Using base Sobol dataset only: {csv_path.name} ({len(df_main)} samples)")
            return df_main

        pool_csv = self.config.workspace.pool_checkpoint_csv
        if not pool_csv.exists():
            raise FileNotFoundError(f"Active-learning pool CSV not found at {pool_csv}")

        df_al = pd.read_csv(pool_csv)
        for col in required_cols:
            if col not in df_al.columns:
                raise ValueError(f"AL pool CSV missing required column: {col}")
        df_al = df_al[required_cols].copy()
        df_al["nBl"] = np.clip(np.round(df_al["nBl"]), 9, 12).astype(int)
        df_al = df_al.drop_duplicates(subset=self.var_names, keep="first").reset_index(drop=True)

        df_merged = pd.concat([df_main, df_al], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=self.var_names, keep="first").reset_index(drop=True)
        _emit(
            progress_callback,
            f"Using base + active-learning Sobol dataset: {csv_path.name} + {pool_csv.name} "
            f"(base={len(df_main)}, al={len(df_al)}, merged={len(df_merged)})"
        )
        return df_merged
