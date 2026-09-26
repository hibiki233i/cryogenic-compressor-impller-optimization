"""Prepare independent, auditable AL experiments without launching CFD."""
from pathlib import Path
import json
import re
import shutil
import platform
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from cfx_convergence import atomic_json, digest
from design_variables import write_performance_data_metadata
from ..config import AppConfig, WorkspacePaths
from .artifacts import data_digest

def export_true_hv_curve(history_path, output_dir):
    """Export genuine observed HV only, with a round-zero baseline; no smoothing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(history_path)
    frame = frame.dropna(subset=['true_hv']).sort_values('iter')
    if frame.empty:
        return
    if 'stage_id' in frame and frame.stage_id.nunique() > 1:
        raise ValueError('Refusing to join different AL experiments into one HV curve')
    out = Path(output_dir)
    columns = [c for c in ('stage_id','iter','n_samples','attempted_queries','accepted_al_queries','true_hv','hv_policy_version','pool_digest','seed') if c in frame]
    frame[columns].to_csv(out/'hv_true_curve.csv',index=False)
    fig, axes = plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,x,label in [(axes[0],'iter','AL iteration'),(axes[1],'n_samples','Accepted training samples')]:
        ax.plot(frame[x],frame.true_hv,'o-',color='#126782',linewidth=1.7,markersize=4)
        ax.set(xlabel=label,ylabel='Verified CFD hypervolume')
        ax.grid(alpha=.2)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/f'hv_true_curve.{ext}',dpi=300)
    plt.close(fig)

def prepare_stage(config, destination, seed_mode='doe', seed=42):
    """DOE-only is a fresh experiment; combined is a warm start, explicitly labeled.

    Output is a new workspace. No defaults are switched and no old histories are
    changed. The caller opens its stage_config.json or sets IMPELLER_APP_CONFIG.
    """
    from .active_learning import ActiveLearningService
    if seed_mode not in ('doe','combined'):
        raise ValueError('seed_mode must be doe or combined')
    source = config.resolved()
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Stage destination already exists; refusing to overwrite an experiment')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',destination.name):
        raise ValueError('Use letters, digits, underscore or hyphen in the stage name')
    l = ActiveLearningService(source).legacy
    master = l.load_and_clean_data(str(source.workspace.training_csv))
    fixed_path = source.workspace.project_root/'fixed_test_set.csv'
    if not fixed_path.exists():
        raise ValueError('An existing independent fixed test set is required')
    # Read-only: existing test set must have valid schema; no implicit repartition.
    fixed = l.load_and_clean_data(str(fixed_path))
    keys = lambda df:{tuple(row) for row in np.round(df[l.VAR_NAMES].to_numpy(float),10)}
    fixed_keys = keys(fixed)
    if not fixed_keys.issubset(keys(master)):
        raise ValueError('Fixed tests must be retained clean DOE samples')
    pool = (master[[tuple(row) not in fixed_keys for row in np.round(master[l.VAR_NAMES].to_numpy(float),10)]].copy()
            if seed_mode == 'doe' else l.load_pool_checkpoint(str(source.workspace.pool_checkpoint_csv)))
    if pool is None or pool.empty or keys(pool)&fixed_keys:
        raise ValueError('Missing training pool or train/test overlap')
    if len(keys(pool)) != len(pool) or len(fixed_keys) != len(fixed):
        raise ValueError('Duplicate training/test designs')
    pool_hash = data_digest(pool,l.VAR_NAMES,l.SURROGATE_OUTPUT_NAMES)
    test_hash = data_digest(fixed,l.VAR_NAMES,l.SURROGATE_OUTPUT_NAMES)
    hv,_,_ = l.compute_true_cumulative_hv(pool[l.VAR_NAMES].to_numpy(float),pool[l.ALL_OUTPUT_NAMES].to_numpy(float),pool.is_boundary.to_numpy(float))
    if not np.isfinite(hv):
        raise ValueError('No verified feasible baseline front; cannot initialize a paper HV experiment')
    cfg = AppConfig.from_dict(source.to_dict())
    cfg.workspace = WorkspacePaths(project_root=destination)
    cfg = cfg.resolved()
    # Source solver paths were fully resolved before replacing workspace.
    source_files = [source.workspace.training_csv, fixed_path, source.workspace.design_variables_json]
    if seed_mode == 'combined':
        source_files.append(source.workspace.pool_checkpoint_csv)
    sources = {str(p):digest(p) for p in source_files}
    packages = {}
    for package in ('numpy','pandas','torch','scikit-learn','scipy','pymoo'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = 'unavailable'
    failures = None
    if source.workspace.failure_records_csv.exists():
        failures = pd.read_csv(source.workspace.failure_records_csv)
        sources[str(source.workspace.failure_records_csv)] = digest(source.workspace.failure_records_csv)
        if seed_mode == 'doe':
            if 'source' not in failures:
                raise ValueError('Failure history has no source; cannot isolate DOE evidence')
            failures = failures[failures.source.astype(str).str.lower().eq('doe')]
    stage = {'stage_id':destination.name,'stage_protocol_version':1,
        'acquisition_policy_version':4,'hv_policy_version':l.HV_POLICY_VERSION,
        'experiment_type':'fresh_from_clean_doe' if seed_mode=='doe' else 'warm_start_with_prior_al',
        'seed_mode':seed_mode,'seed':int(seed),'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_workspace':str(source.workspace.project_root),'source_files':sources,
        'initial_pool_digest':pool_hash,'fixed_test_digest':test_hash,
        'initial_pool_samples':len(pool),'fixed_test_samples':len(fixed),
        'hv_reference':[l.TRUE_HV_REF_EFF,l.TRUE_HV_REF_PR],
        'initial_true_hv':float(hv),'runtime':cfg.to_dict()['runtime'],
        'al_config':asdict(l.CFG),'python':platform.python_version(),'packages':packages,
        'code_sha256':{p.name:digest(p) for p in (Path(__file__).resolve().parents[1]/'config.py',
                          Path(l.__file__),Path(__file__).resolve().parents[2]/'cfx_runner.py')},
        'note':'Historical queried AL values are excluded in fresh mode. No retrospective model errors or invented CFD gains.'}
    # Make the stage only after all source checks have succeeded. Source is untouched.
    destination.mkdir(parents=True)
    for frame,path in [(master,cfg.workspace.training_csv),(pool,cfg.workspace.pool_checkpoint_csv),
                       (pool,destination/'initial_training_pool.csv'),(fixed,destination/'fixed_test_set.csv')]:
        frame.to_csv(path,index=False);write_performance_data_metadata(path)
    shutil.copy2(source.workspace.design_variables_json,cfg.workspace.design_variables_json)
    if failures is not None:
        failures.to_csv(cfg.workspace.failure_records_csv,index=False)
    atomic_json(destination/'al_stage.json',stage)
    baseline={'iter':0,'stage_id':stage['stage_id'],'seed':int(seed),'pool_digest':pool_hash,
              'hv_policy_version':l.HV_POLICY_VERSION,'n_samples':len(pool),'true_hv':float(hv),
              'attempted_queries':0,'accepted_al_queries':0}
    l.write_hv_history(str(cfg.workspace.hv_history_csv),[baseline])
    atomic_json(cfg.workspace.checkpoint_meta_json,{
        'completed_iters':0,'in_progress_iter':None,'pool_samples':len(pool),'test_samples':len(fixed),
        'total_attempts':0,'total_success':0,'failed_points':0,'stage_id':stage['stage_id'],
        'pool_digest':pool_hash,'hv_policy_version':l.HV_POLICY_VERSION,
        'performance_data_schema_version':l.PERFORMANCE_DATA_SCHEMA_VERSION,
        'total_pressure_definition':l.TOTAL_PRESSURE_DEFINITION})
    cfg.save(destination/'stage_config.json')
    export_true_hv_curve(cfg.workspace.hv_history_csv,destination)
    return cfg
