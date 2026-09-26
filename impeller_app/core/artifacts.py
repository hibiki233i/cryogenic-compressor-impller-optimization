"""Content identities for model bundles and derived optimization artifacts."""
from pathlib import Path
import hashlib
import json
import numpy as np
from cfx_convergence import atomic_json, digest
from design_variables import PERFORMANCE_DATA_SCHEMA_VERSION, TOTAL_PRESSURE_DEFINITION

def data_digest(frame, features, targets):
    values = frame[list(features)+list(targets)].to_numpy(dtype='<f8')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite training data')
    rows = sorted(tuple(row) for row in np.round(values, 12))
    return hashlib.sha256(np.asarray(rows,dtype='<f8').tobytes()).hexdigest()

def stage_info(checkpoint_path):
    path = Path(checkpoint_path).parent/'al_stage.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'stage_id':'legacy','seed':42}

def write_model_manifest(model, scaler_x, scaler_y, frame, features, targets, stage_id='legacy'):
    model = Path(model)
    payload = {'bundle_version':1, 'stage_id':stage_id,
        'performance_data_schema_version':PERFORMANCE_DATA_SCHEMA_VERSION,
        'total_pressure_definition':TOTAL_PRESSURE_DEFINITION,
        'raw_features':list(features),'targets':list(targets),
        'data_digest':data_digest(frame,features,targets),
        'artifacts':{str(Path(p).resolve()):digest(p) for p in (model,scaler_x,scaler_y)}}
    atomic_json(model.parent/'surrogate_model_manifest.json',payload)
    return payload

def validate_model_bundle(model, scaler_x, scaler_y, frame, features, targets):
    path = Path(model).parent/'surrogate_model_manifest.json'
    if not path.exists():
        raise ValueError('Model bundle manifest missing; retrain/publish a bound model and scalers before prediction')
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta.get('raw_features') != list(features) or meta.get('targets') != list(targets):
        raise ValueError('Model feature/target definitions do not match this workspace')
    if meta.get('performance_data_schema_version') != PERFORMANCE_DATA_SCHEMA_VERSION:
        raise ValueError('Model performance schema mismatch')
    for file in (model,scaler_x,scaler_y):
        file = Path(file)
        expected = meta.get('artifacts',{}).get(str(file.resolve()),meta.get('files',{}).get(file.name))
        if expected is None or not file.exists() or digest(file) != expected:
            raise ValueError('Model/scaler content mismatch: '+str(file))
    if 'data_digest' in meta:
        if meta['data_digest'] != data_digest(frame,features,targets):
            raise ValueError('Training data changed after model publication; retrain before using predictions')
    else:
        # Migration for the previously activated, hash-bound retraining bundle.
        for name in ('al_training_pool_checkpoint.csv','fixed_test_set.csv'):
            file = path.parent/name
            if not file.exists() or digest(file) != meta.get('files',{}).get(name):
                raise ValueError('Published dataset hash mismatch: '+name)
    return meta
