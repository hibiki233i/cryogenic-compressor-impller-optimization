#!/usr/bin/env python3
"""Read-only surrogate audit and prospective local CFD validation. No CFD or training."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from geometry_constraints import geometry_rule_violations

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [('eff', 100., 'percentage points'), ('pr', 1., 'dimensionless'), ('mf', 1., 'g/s')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def fresh_dir(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f'Refusing to overwrite an existing analysis/protocol: {path}')
    path.mkdir(parents=True)
    return path


def require_finite(frame, columns):
    values = frame[columns].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite values in {columns}')
    return values


def unique_new_mask(candidates, history, tolerance=1e-8):
    if not len(history):
        return np.ones(len(candidates), bool)
    return cdist(candidates, history).min(axis=1) > tolerance


def metrics(frame, group='all'):
    rows = []
    for suffix, factor, unit in OUTPUTS:
        y = require_finite(frame, ['true_' + suffix]).ravel() * factor
        pred = require_finite(frame, ['pred_' + suffix]).ravel() * factor
        err = pred - y
        denominator = np.sum((y-y.mean())**2) if len(y) else 0
        rho = float(spearmanr(y, pred).statistic) if len(y)>2 and np.ptp(y)>0 and np.ptp(pred)>0 else np.nan
        rows.append(dict(group=group, output=suffix, unit=unit, n=len(y),
            rmse=np.sqrt(np.mean(err**2)) if len(y) else np.nan,
            mae=np.mean(abs(err)) if len(y) else np.nan,
            bias=np.mean(err) if len(y) else np.nan,
            r2=1-np.sum(err**2)/denominator if len(y)>1 and denominator>0 else np.nan,
            spearman=rho, p95_abs_error=np.percentile(abs(err),95) if len(y) else np.nan,
            true_min=y.min() if len(y) else np.nan, true_max=y.max() if len(y) else np.nan))
    return pd.DataFrame(rows)


def join_results(candidates, results):
    allowed = {'candidate_id','status','true_eff','true_pr','true_mf','failure_reason','completed_at_utc'}
    if set(results)-allowed:
        raise ValueError(f'Unexpected result columns (predictions/geometry cannot be edited): {set(results)-allowed}')
    if results.candidate_id.isna().any() or results.candidate_id.duplicated().any():
        raise ValueError('Missing or duplicated candidate_id')
    if set(results.candidate_id)-set(candidates.candidate_id):
        raise ValueError('Unknown candidate_id')
    for col in ['true_eff','true_pr','true_mf']:
        if col not in results:
            results = results.assign(**{col:np.nan})
    joined = candidates.merge(results, on='candidate_id', how='left', validate='one_to_one')
    joined['status'] = joined['status'].fillna('pending')
    if not joined.status.isin(['pending','success','failed']).all():
        raise ValueError('status must be pending, success or failed')
    success = joined.status.eq('success')
    values = require_finite(joined.loc[success], ['true_eff','true_pr','true_mf'])
    if len(values) and (np.any((values[:,0]<0)|(values[:,0]>1)) or np.any(values[:,1]<=0) or np.any(values[:,2]<0)):
        raise ValueError('Use efficiency fraction [0,1], positive pressure ratio and nonnegative flow in g/s')
    if joined.loc[~success,['true_eff','true_pr','true_mf']].notna().any().any():
        raise ValueError('Non-success rows must not contain performance results; update their status')
    if 'failure_reason' not in joined:
        joined['failure_reason'] = ''
    if joined.loc[joined.status.eq('failed'),'failure_reason'].astype('string').fillna('').str.strip().eq('').any():
        raise ValueError('Every failed case needs failure_reason')
    return joined


def load_predictor(model_path, x_path, y_path):
    # No import of the orchestration module: it has CFD/runtime dependencies.
    import torch
    from torch import nn
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    expected = {f'net.{i}.{kind}' for i in (0,3,6,9) for kind in ('weight','bias')}
    if set(state) != expected:
        raise ValueError('Unsupported checkpoint; expected four-layer PerformanceSurrogate state_dict')
    dims = [int(state['net.0.weight'].shape[1])] + [int(state[f'net.{i}.weight'].shape[0]) for i in (0,3,6,9)]
    layers = []
    for k in range(4):
        layers.append(nn.Linear(dims[k], dims[k+1]))
        if k<3:
            layers.extend([nn.ReLU(), nn.Dropout((.12,.12,.08)[k])])
    model = nn.Module()
    model.net = nn.Sequential(*layers)
    model.load_state_dict(state, strict=True)
    model.eval()
    sx, sy = joblib.load(x_path), joblib.load(y_path)
    names = list(sx.variable_names)
    if sx.n_features_out_ != dims[0] or sy.n_features_in_ != 3 or dims[-1]!=3:
        raise ValueError('Checkpoint/scaler dimensions disagree')
    def predict(frame):
        X = require_finite(frame, names)
        with torch.no_grad():
            out = model.net(torch.tensor(sx.transform(X), dtype=torch.float32)).numpy()
        return sy.inverse_transform(out)
    return predict, sx, names, dims


def check_archive(predict, data, iteration):
    p = pd.read_csv(data/'fixed_test_predictions_history.csv')
    p = p[p['iter'].eq(iteration)].copy()
    if p.empty:
        raise ValueError(f'No fixed test predictions for iteration {iteration}')
    calculated = predict(p)
    archived = require_finite(p, ['pred_eff','pred_pr','pred_mf'])
    delta = np.max(abs(calculated-archived),axis=0)
    return {'iteration':iteration, 'n':len(p), 'max_absolute_difference':delta.tolist(),
            'matches':bool(np.all(delta<=1e-5)), 'tolerance':1e-5}


def plot_parity(frame, output, title):
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'local-validation-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(11.5,3.6),layout='constrained')
    for ax,(suffix,factor,unit) in zip(axes,OUTPUTS):
        y=frame['true_'+suffix]*factor; p=frame['pred_'+suffix]*factor
        ax.scatter(y,p,s=28,color='#16697a',edgecolors='white',linewidth=.4)
        lo=min(y.min(),p.min()); hi=max(y.max(),p.max()); pad=max((hi-lo)*.08,1e-5)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='.45',lw=1)
        value_unit = '%' if suffix == 'eff' else unit
        display_name = {'eff':'Isentropic efficiency','pr':'Total pressure ratio','mf':'Mass flow rate'}[suffix]
        ax.set(xlabel=f'CFD ({value_unit})',ylabel=f'Prediction ({value_unit})',title=display_name,
               xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad))
        ax.grid(alpha=.18)
    fig.suptitle(title)
    fig.savefig(output.with_suffix('.png'),dpi=220)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def report(path, title, table, notes):
    # No optional tabulate dependency.
    columns=['group','output','n','rmse','mae','bias','r2']
    lines=[f'# {title}', '', *notes, '', '| '+' | '.join(columns)+' |',
           '| '+' | '.join(['---']*len(columns))+' |']
    for _,r in table.iterrows():
        lines.append('| '+' | '.join(f'{r[c]:.6g}' if isinstance(r[c],float) else str(r[c]) for c in columns)+' |')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def online(args):
    d=args.data
    sx=joblib.load(args.scaler or d/'scaler_X.pkl'); names=list(sx.variable_names)
    front=pd.read_csv(args.front or d/'pareto_front_points.csv')
    q=pd.read_csv(d/'al_query_validation.csv')
    if q.run_id.duplicated().any():
        raise ValueError('Duplicate run_id in query history')
    q=q[q['iter'].between(args.end-args.window+1,args.end)].copy()
    require_finite(q,names)
    q['distance_to_front']=cdist(sx.transform(q[names].to_numpy(float)),sx.transform(front[names].to_numpy(float))).min(axis=1)
    success=q.status.eq('success')
    require_finite(q.loc[success],['true_eff','true_pr','true_mf','pred_eff','pred_pr','pred_mf'])
    if not (pd.to_datetime(q.loc[success,'completed_at_utc'],utc=True)>pd.to_datetime(q.loc[success,'submitted_at_utc'],utc=True)).all():
        raise ValueError('Submission/completion timestamps are inconsistent')
    rows=[metrics(q.loc[success],'window_all_success')]; counts=[]
    for radius in args.radii:
        inside=q.distance_to_front.le(radius)
        rows.append(metrics(q.loc[inside&success],f'local_r={radius:g}'))
        counts.append(dict(radius=radius,selected=int(inside.sum()),success=int((inside&success).sum()),
                           failed=int((inside&q.status.eq('failed')).sum()),pending=int((inside&~q.status.isin(['success','failed'])).sum())))
    p=pd.read_csv(d/'fixed_test_predictions_history.csv');p=p[p['iter'].eq(args.end)]
    if p.empty: raise ValueError('Requested fixed-test iteration is absent')
    for label,mask in [('fixed_all',np.ones(len(p),bool)),('fixed_below_target',p.true_mf<3.6),('fixed_compliant',p.true_mf>=3.6)]:
        rows.append(metrics(p.loc[mask],label))
    out=fresh_dir(args.out); table=pd.concat(rows,ignore_index=True)
    table.to_csv(out/'metrics.csv',index=False);q.to_csv(out/'window_all_candidates.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'radius_counts.csv',index=False)
    local=q.loc[success&q.distance_to_front.le(args.radius)]
    local.to_csv(out/'local_success_predictions.csv',index=False)
    if len(local):plot_parity(local,out/'online_parity','Archived pre-CFD predictions: local online validation')
    sources=[d/'al_query_validation.csv',d/'fixed_test_predictions_history.csv',args.front or d/'pareto_front_points.csv',args.scaler or d/'scaler_X.pkl']
    write_json(out/'manifest.json',{'kind':'retrospective_prequential','end':args.end,'window':args.window,
        'radii':args.radii,'primary_radius':args.radius,'sources':{str(p.resolve()):sha(p) for p in sources},
        'distance':'Euclidean in archived scaler coordinates including one-hot nBl; no division by dimension',
        'front_caveat':'Uses explicitly supplied archived front; does not reconstruct historical front or assert manuscript design identity'})
    report(out/'README.md','局部在线验证',table,[
        f'第 {args.end-args.window+1}–{args.end} 轮；主半径 {args.radius:g}；主邻域成功点 {len(local)} 个。',
        '效率误差单位为百分点，流量为 g/s。bias = prediction - CFD。',
        '使用逐轮 CFD 前预测；不是冻结最终模型的独立测试。前沿来源见 manifest，不能默认等同论文代表设计。',
        '所有窗口内点（含邻域外和失败点）保存在 window_all_candidates.csv；不同半径的数量见 radius_counts.csv。',
        '没有按真实效率、真实流量或误差排除局部成功点；只按输入距离限定邻域。'])
    print(f'Online report: {out}; primary local n={len(local)}')


def prepare(args):
    d=args.data
    model=args.model or d/'best_regressor.pth'; xp=args.scaler or d/'scaler_X.pkl'; yp=args.y_scaler or d/'scaler_Y.pkl'
    predict,sx,names,dims=load_predictor(model,xp,yp)
    archive=check_archive(predict,d,args.match_iter)
    if not archive['matches']:
        raise ValueError(f'Model/scalers do not reproduce requested archived iteration: {archive}. Select the correct files/iteration explicitly.')
    front_path=args.front or d/'pareto_front_points.csv'
    front=pd.read_csv(front_path)
    require_finite(front,names+['Efficiency','totalpressureratio'])
    specs=json.loads((d/'design_variables.json').read_text(encoding='utf-8'))['variables']
    specs={s['name']:s for s in specs}
    low=np.array([specs[n]['lower'] for n in names]);high=np.array([specs[n]['upper'] for n in names])
    # Include training, test, checkpoint and all attempted AL points, even failed ones.
    history_files=[d/n for n in ('Compressor_Training_Data.csv','al_training_pool_checkpoint.csv','fixed_test_set.csv','al_query_validation.csv')]
    history=pd.concat([pd.read_csv(p)[names] for p in history_files],ignore_index=True)
    history_X=require_finite(history,names)
    history_norm=sx.transform(history_X)
    # Choose both objective extremes and the normalized ideal-distance compromise.
    obj=front[['Efficiency','totalpressureratio']].to_numpy(float)
    norm=(obj-obj.min(axis=0))/np.maximum(np.ptp(obj,axis=0),1e-12)
    idx=list(dict.fromkeys([int(np.argmax(obj[:,0])),int(np.argmax(obj[:,1])),int(np.argmin(np.sum((1-norm)**2,axis=1)))]))
    anchors=front.iloc[idx].copy()
    rng=np.random.default_rng(args.seed)
    candidates=[]; provenance=[]; accepted_norm=[]; rejected={'history_or_duplicate':0,'geometry':0,'radius':0}
    for k in range(args.count):
        ai=k%len(anchors); anchor=anchors.iloc[ai][names].to_numpy(float)
        # Alternate inner and outer shells; accept/reject on inputs only.
        inner=(k//len(anchors))%2==0
        rlo,rhi=(.04,args.radius*.6) if inner else (args.radius*.6,args.radius)
        for attempt in range(20000):
            lo=np.maximum(low,anchor-args.step*(high-low)); hi=np.minimum(high,anchor+args.step*(high-low))
            x=rng.uniform(lo,hi)
            for fixed in ['nBl','P_out']: x[names.index(fixed)]=anchor[names.index(fixed)]
            if np.any(geometry_rule_violations(x,names,low,high)>1e-10):
                rejected['geometry']+=1;continue
            xn=sx.transform(x[None,:])[0]
            distance=float(np.linalg.norm(xn-sx.transform(anchor[None,:])[0]))
            if not rlo<=distance<=rhi:
                rejected['radius']+=1;continue
            if not unique_new_mask(xn[None,:],history_norm)[0] or (accepted_norm and not unique_new_mask(xn[None,:],np.array(accepted_norm))[0]):
                rejected['history_or_duplicate']+=1;continue
            candidates.append(x);accepted_norm.append(xn)
            provenance.append({'candidate_id':f'LV{k+1:03d}','anchor_front_index':int(anchors.iloc[ai]['front_index']) if 'front_index' in anchors else idx[ai],
                               'shell':'inner' if inner else 'outer','distance_to_anchor':distance})
            break
        else:raise ValueError('Could not fill the prespecified neighbourhood; revise input protocol explicitly')
    c=pd.concat([pd.DataFrame(provenance),pd.DataFrame(candidates,columns=names)],axis=1)
    prediction=predict(c)
    for j,s in enumerate(['eff','pr','mf']): c['pred_'+s]=prediction[:,j]
    c['distance_to_front']=cdist(np.array(accepted_norm),sx.transform(front[names].to_numpy(float))).min(axis=1)
    c['nearest_history_distance']=cdist(np.array(accepted_norm),history_norm).min(axis=1)
    out=fresh_dir(args.out);frozen=out/'frozen';frozen.mkdir()
    for src,target in [(model,'best_regressor.pth'),(xp,'scaler_X.pkl'),(yp,'scaler_Y.pkl'),(front_path,'front.csv')]:shutil.copy2(src,frozen/target)
    c.to_csv(out/'candidates_predictions.csv',index=False)
    c[['candidate_id']+names].to_csv(out/'cfd_inputs.csv',index=False)
    anchors.to_csv(out/'anchors.csv',index=False)
    pd.DataFrame({'candidate_id':c.candidate_id,'status':'pending','true_eff':np.nan,'true_pr':np.nan,'true_mf':np.nan,
                  'failure_reason':'','completed_at_utc':''}).to_csv(out/'cfd_results_template.csv',index=False)
    import torch,scipy,sklearn
    protocol={'kind':'prospective_frozen_local_validation','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'count':args.count,'seed':args.seed,'radius':args.radius,'max_step_fraction':args.step,'architecture':dims,
        'archive_check':archive,'variable_names':names,'rejections_before_freezing':rejected,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'sampling':'bounded uniform proposal, hard geometry rules, same nBl and P_out as anchor; inner/outer shells; no performance filtering',
        'anchor_policy':'efficiency extreme, pressure-ratio extreme, normalized ideal-distance compromise from supplied front; NOT asserted to be manuscript representative',
        'analysis':'all successful cases; no error/output exclusions; failed/pending counts separately; deterministic dropout-off predictions; bias=prediction-CFD',
        'flow_threshold_g_s':3.6,
        'scope':'local conditional on supplied front, same blade-count categories and operating pressure as anchors; no global optimality guarantee',
        'source_hashes':{str(p.resolve()):sha(p) for p in history_files+[model,xp,yp,front_path,d/'design_variables.json']},
        'locked_files':{str(p.relative_to(out)):sha(p) for p in [out/'candidates_predictions.csv',out/'cfd_inputs.csv',out/'anchors.csv',*frozen.iterdir()]}}
    write_json(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    (out/'README.md').write_text('''# 冻结模型局部 CFD 验证包

cfd_inputs.csv 是送 CFD 的 14 个原始输入及唯一编号；candidate_id 必须贯穿运行记录。
复制 cfd_results_template.csv 为 cfd_results.csv 并填结果，禁止修改候选或预测文件。
true_eff 填 0–1 小数（0.81 表示 81%）；true_pr 为总压比；true_mf 单位为 g/s。
成功填 success；失败填 failed 并填写 failure_reason；未完成保留 pending。
所有点使用与原数据库一致的 CFD 设置和性能提取方式。P_out 和 nBl 已固定至对应锚点。
此包没有运行 CFD，也没有训练模型。几何规则通过不等于 CFturbo 或 CFD 一定成功。

冻结文件校验可检测意外更改，不是外部时间戳或第三方预注册。
协议冻结后不应按预测、真实结果或误差更换样本；所有失败和未完成点必须保留。
在新验证完成前不要把新结果加入该冻结模型的训练数据。
模型、变换器、前沿及已知历史数据的哈希和版本写在 protocol.json。
这里使用存档前沿，未自动确认其代表点就是论文中的最终几何。
''',encoding='utf-8')
    print(f'Frozen package: {out}; n={len(c)}; archive iteration {args.match_iter} reproduced')


def evaluate(args):
    package=args.package
    if sha(package/'protocol.json')!=(package/'protocol.sha256').read_text().strip():raise ValueError('Protocol changed')
    protocol=json.loads((package/'protocol.json').read_text())
    for name,digest in protocol['locked_files'].items():
        if sha(package/name)!=digest:raise ValueError(f'Frozen file changed: {name}')
    c=pd.read_csv(package/'candidates_predictions.csv');r=pd.read_csv(args.results)
    joined=join_results(c,r)
    successful=joined[joined.status.eq('success')]
    table=metrics(successful,'frozen_all_success')
    counts=joined.status.value_counts().to_dict()
    state='complete' if counts.get('pending',0)==0 else 'incomplete'
    threshold=protocol['flow_threshold_g_s']
    flow={'true_positive':int(((successful.pred_mf>=threshold)&(successful.true_mf>=threshold)).sum()),
          'false_positive':int(((successful.pred_mf>=threshold)&(successful.true_mf<threshold)).sum()),
          'true_negative':int(((successful.pred_mf<threshold)&(successful.true_mf<threshold)).sum()),
          'false_negative':int(((successful.pred_mf<threshold)&(successful.true_mf>=threshold)).sum())}
    out=fresh_dir(args.out)
    joined.to_csv(out/'all_candidates_results.csv',index=False);table.to_csv(out/'metrics.csv',index=False)
    write_json(out/'summary.json',{'completion':state,'status_counts':counts,'flow_confusion_success_only':flow,
        'results_sha256':sha(args.results),'protocol_sha256':sha(package/'protocol.json'),
        'note':'Failed CFD cases are not assumed flow-infeasible; correlations/r2 unavailable for constant or insufficient outcomes.'})
    report(out/'README.md','冻结模型局部验证结果',table,[f'状态：{state}；样本状态计数：{counts}。',
        '包含全部已成功的预定样本，未按误差/效率/流量过滤。失败点仅报告失败，不虚构性能值。',
        '若仍有 pending，这只是阶段性汇总，不能作为已完成的独立验证。',
        '效率误差单位为百分点；流量 g/s；正偏差表示高估；R²/秩相关不可计算时留空。'])
    if len(successful):plot_parity(successful,out/'frozen_parity','Frozen model: independent local CFD validation')
    print(f'Evaluation: {out}; {state}; {counts}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    a=commands.add_parser('online');a.set_defaults(func=online)
    b=commands.add_parser('prepare');b.set_defaults(func=prepare)
    for p in [a,b]:
        p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--out',type=Path,required=True)
        p.add_argument('--front',type=Path);p.add_argument('--scaler',type=Path)
        p.add_argument('--radius',type=float,default=.25)
    a.add_argument('--end',type=int,default=45);a.add_argument('--window',type=int,default=5)
    a.add_argument('--radii',type=float,nargs='+',default=[.20,.25,.30])
    b.add_argument('--model',type=Path);b.add_argument('--y-scaler',type=Path)
    b.add_argument('--match-iter',type=int,default=45);b.add_argument('--count',type=int,default=24)
    b.add_argument('--step',type=float,default=.08);b.add_argument('--seed',type=int,default=20260910)
    e=commands.add_parser('evaluate');e.set_defaults(func=evaluate)
    e.add_argument('--package',type=Path,required=True);e.add_argument('--results',type=Path,required=True);e.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if hasattr(args,'radius') and args.radius<=.07:parser.error('radius must exceed .07 for the predefined shells')
    if hasattr(args,'count') and (args.count<1 or not 0<args.step<=1):parser.error('count must be positive and step in (0,1]')
    if hasattr(args,'window') and (args.window<1 or args.end<1 or any(r<=0 for r in args.radii)):parser.error('Invalid window/iteration/radii')
    args.func(args)


if __name__=='__main__':
    main()
