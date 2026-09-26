# BOUNDYR

A Windows-oriented impeller optimization workbench built around:

- active learning
- neural-network surrogate prediction
- NSGA-II multi-objective optimization
- Pareto-front querying and case export
- a PySide6 desktop GUI for workflow orchestration

## What Is Included

- `impeller_app/`
  Desktop application package with config, runner, core services, and GUI.
- `NN_NSGA2_ActiveLearning_refactored.py`
  Legacy active-learning and optimization engine.
- `DOE.py`
  Legacy DOE workflow.
- `cfx_runner.py`
  CFX execution pipeline wrapper.
- `pareto_front_query.py`
  Pareto-front extraction and inverse query utilities.
- `pareto_export_cft_cases.py`
  Pareto case export utilities.

## GUI

Install dependencies:

```bash
python3 -m pip install -r requirements-gui.txt
```

Launch the desktop app:

```bash
python3 -m impeller_app
```

On Windows, you can also use:

```bat
launch_gui.bat
```

## External Dependencies

The GUI does not bundle ANSYS CFX, CFturbo, or PowerShell. The target Windows machine still needs:

- PowerShell 7
- ANSYS CFX
- the geometry script `Run-GeometryMeshing.ps1`
- valid template files and accessible working directories

Configure these paths in the GUI `Environment` section before running DOE or active-learning tasks.
The desktop GUI uses left-side workflow navigation, scrollable forms, collapsible DOE engineering
thresholds, and a resizable activity log. The fixed outlet static pressure remains visible in the
DOE section. The status bar shows whether a task is running or stopping.
The desktop app now auto-saves path and runtime edits and reloads them on the next launch.
On Windows, the default config file is `%APPDATA%\\BOUNDYR\\impeller-app-config.json`.
Default engineering parameters, including the fixed NSGA-II operating point, flow/efficiency
thresholds, and geometry screening thresholds, can be edited in the GUI or JSON config file.

## Optimization Problem Definition

- DOE samples only the 13 geometry variables and pins `P_out` to the configured optimization
  outlet static pressure (default `12 Pa`). The same hard geometry rules used by active learning
  are applied before a DOE case is launched.
- `P_out` is an operating-condition input, not a geometry design variable.
- The GUI therefore exposes one shared fixed `P_out` value for DOE and active
  learning; it is no longer shown as a lower/upper design range.
- NSGA-II optimizes the 13 geometry variables at a fixed outlet static pressure. The default
  design point is `P_out = 12 Pa`.
- Total pressure ratio is the mass-flow-averaged stationary-frame total-to-total pressure ratio.
- Total pressure ratio is maximized without a minimum or maximum pressure-ratio constraint.
- Observed-data Pareto extraction only uses samples inside the configured pressure band around
  the fixed design point; it does not mix the full operating map into a geometry Pareto front.

The stationary-frame pressure-ratio definition is performance-data schema version 2. Legacy
`CFX_Results.txt`, training CSV, fixed-test, and active-learning checkpoint data are rejected
unless they carry matching metadata. Use DOE recovery to re-run CFX-Post from existing `.res`
files, then retrain the surrogate and regenerate the Pareto front.

## Surrogate Validation Records

Each active-learning round writes review-ready validation data into the configured workspace:

- `surrogate_validation_history.csv`: fixed-test RMSE, MAE, R² and K-fold mean/std metrics.
- `surrogate_cv_fold_history.csv`: outer-fold RMSE, MAE and R². Each outer training
  partition has a separate inner early-stopping holdout; scalers and density weights
  use only the gradient-training partition. The outer fold is evaluated once.
- `fixed_test_predictions_history.csv`: fixed-test CFD truth and surrogate prediction pairs.
- `al_query_validation.csv`: pre-CFD prediction, MC-Dropout uncertainty, CFD truth, residuals,
  and one-/two-sigma coverage for every queried point.

Surrogate inputs one-hot encode `nBl`, while train/validation and K-fold splits prefer
`nBl`-aware strata. Tempered inverse-density and inverse-frequency category weights keep dense
active-learning clusters from dominating. Efficiency and pressure-ratio losses emphasize
flow-feasible successful CFD points; the mass-flow head uses every successful CFD point.

Each acquisition round creates one MC-Dropout distribution for the complete candidate pool.
Flow feasibility is applied inside each joint draw as
`mean(HVI(Efficiency_s, PressureRatio_s) * (MassFlow_s >= 3.6 g/s))`, preserving
objective/flow dependence in the shared network. Solver and geometry risk weights
are then applied. The same distribution is reused for screening, ranking, selection,
and query logging. The batch quota adapts to validation
reliability: reliable models may select three meaningful-EHVI points plus one local uncertainty
point; ordinary models reduce EHVI use; failed models only perform trust-region space filling.
Zero or negligible EHVI points are never selected merely to fill a quota.
Within the general tier, the two-point quota requires NRMSE, absolute bias,
Brier score, and ECE all to be in the better half of their configured usable
range; a weak dimension lowers the quota to one.

Coverage is skipped when the model is failed or no candidate passes the requested
solver, geometry, and flow gates. Blade-count preference never overrides those gates.
For general/reliable models, the existing 7:3 safe/boundary cycle is retained. Up to
15% of candidates now perturb successful CFD designs within 0.20 g/s of the true
flow boundary, at the fixed operating point and the same blade count. Boundary
identification prefers these observed neighbourhoods and requires normalized distance
to a successful training sample of at most 0.25. General-model local exploration also
checks the solver and geometry probabilities separately. Failed models select only
trust-region space-filling points.

These are configurable heuristics in `ALConfig`, not a guarantee of CFD success.
The flow 10th percentile is an uncalibrated MC estimate and is logged as such.
New query rows carry `acquisition_policy_version=3` and
`ehvi_estimator=joint_mc_flow_constrained`; new cross-validation rows carry
`cv_protocol=outer_test_inner_early_stopping_v2`. Historical scores should be
compared with care because the estimators and validation protocol changed.

Training/validation splits now precede fitting X/Y scalers in all training entry
points. To retrain and evaluate a checkpoint for offline review without replacing
the configured model or running CFD:

```bash
python final_checkpoint_evaluation.py --output-dir _diagnostics/checkpoint_review
```

The output includes a separate model/scaler set, fixed-test metrics, and nested
early-stopping validation metrics. `train_samples` counts gradient-training samples;
`validation_samples` counts the separate early-stopping holdout, not extra gradient
training data.

## Failure-Aware Active Learning

DOE and active-learning run outcomes are persisted in `failure_records.csv`. A failure caused by
software, setup, licensing, CFX-Pre, or CFX-Post is retained for audit but is not treated as a
physically infeasible design. Deterministic blockage, invalid-flow, and explicitly detected
fatal-overflow outcomes are confirmed immediately; retry-based confirmation remains available for
repeatable geometry or mesh-generation failures.

- Confirmed CFturbo/TurboGrid generation failures train the geometry-safety classifier with 12
  learned inputs: fixed `P_out` and categorical blade count `nBl` are deliberately excluded.
- Confirmed fatal-overflow, blockage, and invalid-flow outcomes train the operating-feasibility
  classifier with 12 learned inputs. Fixed `P_out` and `nBl` are excluded to prevent stage leakage
  and self-reinforcing blade-count bias.
- A generic CFX solver exit or divergence without an explicit fatal-overflow signature remains an
  audit record and does not train either design-failure classifier.
- Converged near-boundary samples remain in the regression data with `is_boundary`; they are not
  mixed with hard execution failures.
- The legacy untyped `failed_points.npy` file is ignored by the new classifiers and candidate
  filter. The typed CSV is rebuilt each round so a point that later succeeds is not permanently
  blacklisted.

## Notes

- The project is currently structured to support Windows-based engineering workflows.
- Existing legacy research scripts are preserved and wrapped by the desktop app rather than fully replaced.
- DOE recovery, active-learning resume, Pareto querying, and case export are all available through the GUI layer.
- The GUI maintains surrogate-input bounds in `design_variables.json`. Geometry inputs and
  operating-condition inputs have distinct roles.
- Flow, efficiency, and geometry feasibility thresholds are configurable engineering parameters.
- Environment validation can create an empty training CSV automatically when the configured file does not yet exist.
# 2026-09：清理样本后的主动学习与收敛验收

## 仅 DOE 的 NSGA-II 对照基准

GUI 的“仅 DOE 的 NSGA-II 基准”和命令行 `--nsga2-only` 默认仅用清理后的
DOE 训练数据，排除固定测试集；不读取 AL 训练池，也不使用 AL 失败标签训练分类器。
`--nsga2-use-pool-checkpoint` 仅作为显式选择 DOE＋AL 的兼容选项。
对照模型和 scaler 独立保存在输出目录下的 `nsga2_doe_model/`，不会覆盖 AL 模型。
主动学习自身继续累积和使用 AL 样本。

GUI 正常完成或因 HV 停滞结束后，自动用系统图片查看器打开保存的 `hv_convergence.png`。
论文用真实 HV 图仍保存为 `hv_true_curve.png/.svg/.pdf`；无交互运行只保存文件。
100% wall 终止会记录 PID 并确认退出；持续追踪本次启动的子进程及本次运行目录内
新启动的 CFX 求解进程，清理脱离父进程树的进程，不按名称终止其他工程的求解任务。

权重与两个 scaler 均保存到配置的完整路径，`surrogate_model_manifest.json`
记录它们的内容散列、特征/目标顺序及训练数据身份。Pareto 逆向查询加载配套 scaler，
不再根据整个查询数据重新拟合 scaler。旧的或数据已变化的 Pareto 缓存不能直接导出；
先重新计算真实前沿及工程排序。NSGA-II 输出摘要记录模型、数据、阶段和随机种子。

## 为论文准备独立的 AL 实验

初始化只复制数据、写配置和真实 HV 第 0 轮，不启动几何、网格或 CFD，也不切换当前配置：

```powershell
python prepare_al_stage.py --name paper_clean_doe_s42 --seed-mode doe --seed 42
```

默认读取已保存的 GUI 配置，也可用 `--config <源配置路径>`。
新阶段位于源工作区 `al_stages/paper_clean_doe_s42/`，目标目录必须不存在。

两种模式的实验含义不同：

| 模式 | 初始训练数据 | HV 曲线含义 |
|---|---|---|
| `doe`（默认） | 清理 DOE，排除固定测试集 | 从 DOE 起步重新执行 AL；不预先引入历史 AL 标签或模型 |
| `combined` | 当前有效 DOE＋AL 训练池 | 以已有 AL 成果为初始前沿追加采样，属于 warm start |

当前清理数据对应 DOE 270、固定测试 39；fresh 阶段初始训练池为 231。
combined 阶段使用当前 406 条训练池。这些是此次数据的数量，不是代码固定值。
fresh 模式仅继承 DOE 来源的失败审计记录，不继承旧 AL 失败分类器、查询误差或 HV 停滞记录。
两个模式都从新阶段第 0 轮开始，使用独立工作目录，原工作区原样保留。

以后决定实际运行时，在同一 PowerShell 会话指定新配置，再启动 GUI：

```powershell
$env:IMPELLER_APP_CONFIG = '<新阶段目录>\stage_config.json'
python -m impeller_app
```

然后由用户在 GUI 明确启动主动学习。也可以在 GUI 中加载该配置文件。
旧工作区带有数据清理标记却仍使用旧阶段历史时，AL 会拒绝直接续跑并提示初始化新阶段。
新阶段恢复时检查训练池和固定测试集身份，防止在相同曲线中静默更换数据。

每轮使用阶段种子＋轮次控制候选与 NSGA-II 随机性；`al_stage.json`
保存初始数据散列、配置、算法版本、软件版本、HV 参考点及实验类型。
`hv_history.csv` 保存每轮真实 CFD HV、代理 HV、有效样本数、查询次数及模型身份。
独立论文导出 `hv_true_curve.csv/.png/.svg/.pdf` 仅包含真实 CFD HV，不平滑、不拼接旧阶段、
不把预测提升当作 CFD 提升；第 0 轮是实际初始前沿。固定测试样本不进入 HV 训练池。
每个完整轮次后自动更新该曲线，保留现有 `hv_convergence.png` 诊断图。
实际收益为零时曲线可以出现平台，仍按真实数据保留。39 条测试数据是历史保留集，不是新采集盲测。

## CFX 残差验收与 RES 自动续跑

所有经过 `cfx_runner.run_cfx_pipeline` 的 DOE、AL 和恢复流程，都在后处理前检查
**最新 OUT 的最后一轮 RMS Res**，不使用 Max Res 或前一轮低残差代替：

- 默认要求 U/V/W-Mom、P-Mass、H-Energy、K-TurbKE、O-TurbFreq 齐全、有限，且全部严格小于 `1e-4`。
- 等于 `1e-4` 不合格；OUT 未正常结束、缺少残差或缺少同名 RES 时不作为成功样本。
- 初次求解结束仍未达标时，从该 OUT 对应的 RES 使用 `-def <res>` 自动续跑，无需被清理的 DEF。
- 续跑每批最多 500 次，额外总预算最多 1500 次；每批完成都重新检查，达标立即进入 Post。
- 续跑 CCL 使用 OUT 中实际 FLOW 名称，RMS 目标设为验收阈值的十分之一，避免求解器在验收边界提前结束。
- 每次启动使用独立输出名称，并只监控/终止本次求解进程树。
- 进口或出口任一边界连续三个迭代明确报告 faces/area 均为 100% 时，作为 blockage 终止。
- 预算保存在运行目录 `cfx_convergence_state.json`，每批启动前预留预算。取消/异常不会重获 1500 次；
  未完整结束的 pending 批次需人工检查，程序不会不明原因继续追加预算。
- 之前 `DOE_resume_*.out/.ccl` 记录的续算预算会迁移计入上限；已有 `DOE_INVALID.json` 的
  人工确认 wall 标记也会被尊重。只有标记之后的新成功收敛结果才可解除废弃状态。
- 1500 次预算耗尽仍未达标，返回 `residual_unconverged`，保留 OUT/RES 和审计信息，
  旧性能 TXT 转存到 `rejected_results/`，写 `CFX_INVALID.json`，该点不加入样本库且不执行重复失败复核。
  这是数值不合格，不作为几何/物理不可行的分类器负标签。
- 成功后从最终 RES 重新提取结果，metadata 绑定最终 OUT 散列及 RES 文件身份。
  缓存 TXT 即便 schema 正确，也必须通过对应的收敛与新鲜度检查才能复用。

`RuntimeSettings` / 配置 JSON 新增 `cfx_residual_threshold`（默认 `0.0001`）、
`cfx_max_extra_iterations`（默认 `1500`，不得超过 `1500`）、`cfx_restart_chunk`（默认 `500`）。
这些设置经应用服务传到 DOE 和 AL 求解入口，GUI 保存时保留它们。
恢复旧运行现在可能触发 RES 续跑，不再等同于只执行 Post。

## 离线回归验证

```powershell
python -m unittest tests.test_convergence_policy tests.test_al_stages -v
python -m unittest discover -s tests -v
```

测试使用临时工作区和模拟求解器，覆盖最终残差、严格门槛、OUT/RES 配对、wall、预算耗尽、
取消恢复、旧 TXT 失效、成功结果来源、新阶段隔离、固定测试无泄漏、保存路径及过期导出拒绝。
不需要 ANSYS 许可证。真实求解只能由用户之后明确启动。
