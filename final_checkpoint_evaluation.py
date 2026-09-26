from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from impeller_app.config import AppConfig
from impeller_app.core.active_learning import ActiveLearningService


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint without running CFD.")
    parser.add_argument("--output-dir", type=Path, help="Save all newly trained artifacts here for offline review.")
    args = parser.parse_args()
    service = ActiveLearningService(AppConfig.load())
    legacy = service.legacy
    checkpoint_meta = json.loads(
        Path(legacy.CHECKPOINT_META_PATH).read_text(encoding="utf-8")
    )
    completed_iter = int(checkpoint_meta.get("completed_iters", 0))
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_regressor.pth" if output_dir else Path(legacy.BEST_REG_PATH)
    x_path = output_dir / "scaler_X.pkl" if output_dir else Path(legacy.SCALER_X_PATH)
    y_path = output_dir / "scaler_Y.pkl" if output_dir else Path(legacy.SCALER_Y_PATH)

    pool_df = legacy.load_and_clean_data(legacy.POOL_CHECKPOINT_CSV)
    test_df = legacy.load_and_clean_data(legacy.TEST_SET_CSV)

    X_pool = legacy.snap_discrete_vars(
        pool_df[legacy.VAR_NAMES].to_numpy(dtype=float)
    )
    Y_pool = pool_df[legacy.ALL_OUTPUT_NAMES].to_numpy(dtype=float)
    W_pool = pool_df["is_boundary"].to_numpy(dtype=float)
    X_test = legacy.snap_discrete_vars(
        test_df[legacy.VAR_NAMES].to_numpy(dtype=float)
    )
    Y_test = test_df[legacy.ALL_OUTPUT_NAMES].to_numpy(dtype=float)

    scaler_X = legacy.make_categorical_nbl_scaler(
        legacy.VAR_NAMES, legacy.L_BOUNDS, legacy.U_BOUNDS
    )
    scaler_Y = MinMaxScaler()
    Y_pool_surr = Y_pool[:, legacy.SURROGATE_OUTPUT_IDX]
    Y_test_surr = Y_test[:, legacy.SURROGATE_OUTPUT_IDX]

    train_idx, val_idx, split_method = legacy.split_training_validation_indices(
        X_pool, W_pool,
    )
    scaler_X.fit(X_pool[train_idx])
    scaler_Y.fit(Y_pool_surr[train_idx])
    X_pool_norm = scaler_X.transform(X_pool)
    Y_pool_norm = scaler_Y.transform(Y_pool_surr)
    nbl_idx = legacy.VAR_NAMES.index("nBl")
    sample_weights, weight_diagnostics = (
        legacy.compute_density_nbl_sample_weights(
            X_pool_norm[train_idx],
            X_pool[train_idx, nbl_idx],
            k_neighbors=legacy.CFG.density_k_neighbors,
            min_weight=legacy.CFG.sample_weight_min,
            max_weight=legacy.CFG.sample_weight_max,
        )
    )

    model, history = legacy.train_regressor(
        X_pool_norm[train_idx],
        Y_pool_norm[train_idx],
        (Y_pool_surr[train_idx, 2] < legacy.BOUNDARY_FLOW_G_S).astype(float),
        X_pool_norm[val_idx],
        Y_pool_norm[val_idx],
        (Y_pool_surr[val_idx, 2] < legacy.BOUNDARY_FLOW_G_S).astype(float),
        sample_weights_train=sample_weights,
        save_path=model_path,
        random_seed=1000,
    )
    joblib.dump(scaler_X, x_path)
    joblib.dump(scaler_Y, y_path)

    fixed_pred = legacy.deterministic_predict(
        model, scaler_X.transform(X_test), scaler_Y
    )
    fixed_metrics = legacy.regression_metrics(
        Y_test_surr, fixed_pred, prefix="fixed_test_"
    )

    cv_rows, cv_summary = legacy.run_kfold_surrogate_validation(
        X_raw=X_pool,
        Y_surr=Y_pool_surr,
        W_boundary=W_pool,
        al_iter=completed_iter,
        cfg=legacy.CFG,
    )

    true_hv, true_front_y, true_front_x = legacy.compute_true_cumulative_hv(
        X_pool=X_pool,
        Y_pool=Y_pool,
        W_pool=W_pool,
        geom_warn_clf=None,
        geom_safe_threshold=None,
        ref_eff=legacy.TRUE_HV_REF_EFF,
        ref_pr=legacy.TRUE_HV_REF_PR,
    )
    local_metrics = legacy.compute_local_online_metrics(
        query_csv_path=legacy.AL_QUERY_VALIDATION_CSV,
        current_pareto_X=true_front_x,
        scaler_X=scaler_X,
        completed_iter=completed_iter,
        cfg=legacy.CFG,
    )
    reliability, ehvi_quota, reliability_score = (
        legacy.assess_model_reliability(local_metrics, legacy.CFG)
    )

    summary = {
        "evaluation_kind": f"post_round_{completed_iter}_final_checkpoint",
        "completed_iters": completed_iter,
        "pool_samples": int(len(X_pool)),
        "fixed_test_samples": int(len(X_test)),
        "split_method": split_method,
        "split_random_state": 42,
        "model_random_seed": 1000,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "epochs": int(len(history)),
        "true_hv": float(true_hv),
        "true_front_size": int(len(true_front_y)),
        "model_reliability": reliability,
        "model_reliability_score": float(reliability_score),
        "ehvi_quota_before_hv_review": int(ehvi_quota),
        "preprocessing_protocol": "training_partition_only_v2",
        "best_regressor_path": str(model_path),
        "scaler_x_path": str(x_path),
        "scaler_y_path": str(y_path),
        **fixed_metrics,
        **cv_summary,
        **local_metrics,
        **weight_diagnostics,
    }
    output_path = (output_dir or Path(__file__).parent) / "active_learning_final_evaluation.json"
    if output_dir:
        pd.DataFrame(cv_rows).to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    output_path.write_text(
        json.dumps(_finite_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_finite_json(summary), ensure_ascii=False, indent=2))
    print(f"[final-evaluation] saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
