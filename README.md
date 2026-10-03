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

### BOUNDYR desktop appearance

The desktop uses a dark engineering-console palette with teal primary actions, a persistent workflow sidebar, and a page heading that follows navigation. Ctrl+1 through Ctrl+6 switch between the six workflow pages; Ctrl+7 opens live results. Forms scroll and wrap long labels on smaller windows; engineering thresholds and task activity remain collapsible. Task status uses both text and color. Chinese and English navigation, headings, and hints update together. This visual refresh does not change solver execution, performance schemas, fixed outlet pressure, or model compatibility.

界面采用深色工程工作台与青绿色主操作按钮，保留六步工作流，支持 Ctrl+1 至 Ctrl+6 切换工作流页面，Ctrl+7 打开实时结果。小窗口可滚动表单，工程阈值与任务动态可折叠；任务状态同时以文字和颜色表达。此次界面调整无需重新训练模型或生成 Pareto 数据。

### Live results / 实时结果

The **Live results** page (`Ctrl+7`) shows task messages and provided progress immediately. Starting a background task opens this page; DOE selects the DOE data source and active learning selects the training pool. A background reader checks saved files every 3 seconds, with manual refresh and pause controls. Monitoring uses the loaded configuration, updated when a task saves its configuration; editing an input alone does not redirect a running monitor.

The page displays accepted record counts, highest recorded efficiency and total-to-total pressure ratio, a CFD scatter plot, and the last 200 file rows in reverse file order. DOE and AL-pool data are separate selectable sources, never silently merged. Counts include boundary samples and do not imply engineering feasibility. Nonfinite values, invalid boundary flags, and samples outside the configured fixed `P_out` pressure band are excluded. Duplicate geometry/operating-point rows keep the last record. Summary values use the complete filtered dataset; scatter plots render at most 5,000 evenly selected rows. Table order is file order, not solver completion time.

HV curves distinguish verified CFD HV from surrogate HV. They preserve round zero and missing-value gaps and show only iterations committed in checkpoint metadata. History, checkpoint and `al_stage.json` must agree on stage/policy, current performance schema and recorded operating condition. Legacy histories missing this provenance are reported as unavailable rather than silently attributed to the current stage. No solver or model is run to fill missing values. File sources, modification times and check time are shown; hover over a warning for the underlying diagnostic. Missing, malformed, changing, oversized (>64 MiB), or incompatible files clear the affected data and are retried on later refreshes. Other valid data sources remain visible.

实时结果页每 3 秒在后台检查已保存的数据，支持手动刷新及暂停自动刷新。任务日志与已有的进度回调即时显示；单个 CFD 算例尚未完成并写入结果表时，不显示推算的中间性能，HV 则在迭代完成并提交 checkpoint 后更新。本功能不读取 CFX 内部每一步残差，也不启动额外求解。

DOE 与主动学习训练池分别展示；效率—总压比图为已记录 CFD 样本散点，不代表经过工程筛选的 Pareto 前沿。真实 HV 与代理预测 HV 分开标注。缺少阶段来源信息的历史 HV 会显示不可用提示，需要指向已有完整阶段配置与产物，而不是由界面补写元数据。监视当前加载的配置，并随任务启动时保存的配置更新；仅编辑输入框不会把正在监视的任务切换到另一个目录。此功能只读结果文件，不改变训练、求解、采集协议或模型兼容性。

### Native data analysis and case browser / 原生数据分析与算例浏览

The PySide6 desktop now includes **Data analysis (Ctrl+8)** and **Case browser (Ctrl+9)**. Both use native Qt tables and QtCharts; no embedded website, external dashboard or model rerun is required. Visible inspection pages refresh in a background worker every 15 seconds. The shared auto-refresh switch pauses polling; manual refresh remains available. Switching configurations or cases discards late responses from previous reads.

**Open results config…** loads an existing application or stage JSON for read-only inspection in these two pages. Relative workspace roots resolve against the selected configuration file. This does not change the task runner configuration, the live task monitor, saved engineering settings or training data. **View current workspace** returns to the currently loaded task configuration. This is useful when a historical stage has its own training pool, audit files and case directory.

Data analysis provides:

- DOE/pool selection, boundary-flag and categorical blade-count filtering; variable/target scatter plots for efficiency, stationary-frame mass-flow-averaged total-to-total pressure ratio, mass flow and power.
- Pearson correlations for continuous geometry variables only. `nBl` is summarized by group (count, mean, sample standard deviation, min/max); `P_out` remains a fixed operating-condition filter, not a design variable. Correlation is descriptive and does not replace the existing Sobol workflow.
- Separate online pre-CFD and fixed-test prediction views, with latest/specific/all-iteration selection, parity plots, signed errors (`prediction - observation`), original-unit uncertainty where recorded, and MAE/RMSE/bias/R². R² remains unavailable for insufficient or constant truth. Power has no surrogate prediction column and is not fabricated.
- Historical prediction truth is matched by design/operating-point values against schema-v2 observations. Online records must be successful (and `result_valid` when that field exists). Named stages require matching stage tags; legacy untagged records are only numerically verified in a legacy workspace, not attributed to a named stage. Unverified records remain inspectable in raw audit tables but do not enter prediction statistics. These views never rerun or fit a model.
- CV fold histories separated by recorded protocol and any recorded stage; old untagged protocol/stage records are explicitly historical audits rather than current-condition evaluations. Candidate source distributions, raw query records and validation summaries are available under Iteration audit.
- Wheel zoom, rectangle zoom, double-click reset and hover values. Clicking a linked online prediction opens its case. Charts are bounded to 5,000 displayed points; statistics and table exports use all eligible rows. Tables support text search, numeric sorting and atomic export of the currently filtered/sorted rows; monitored source CSVs cannot be overwritten by the export action. Inspection exports are review tables, not automatically schema-tagged training artifacts.

Case browser provides:

- Separate DOE and active-learning inventories, including audit-only records whose directories are missing, and unregistered directories. Source/status/keyword filters do not treat unknown cases as failures. Identity includes source and directory as well as run ID. A same-named case from another stage is not linked to the current audit or prediction point.
- Input parameters, recorded outcome/attempt count/failure reason, structured metadata and current CFX result validity. Recorded success is distinct from current valid-result evidence. The existing CFX definition/convergence checker is reused; no recovery, postprocessing or solver command is executed. Historical fractional LHS `nBl` inputs are retained verbatim while full-impeller flow and power use the same `int(round(nBl))` executed count as the runner.
- File inventory, text/log/JSON preview, bounded raster-image preview and an explicit open-folder button. Text previews read at most the last 256 KiB / 800 lines; images are limited to 8 MiB / 32 million source pixels. Binary solver files and scripts are never executed by preview. File traversal stays within the selected case; directory and file inventories are bounded to 10,000 cases per source and 2,000 files/five directory levels per case.
- Missing, malformed, changed or incompatible data produce explicit diagnostics. Software/license/numerical failures remain audit outcomes and are not reclassified as physical infeasibility. Successful reruns and later cancellations are not overridden by older failure records.

新增页面：**数据分析（Ctrl+8）**、**算例浏览（Ctrl+9）**，均采用 PySide6 原生表格、标签页和 QtCharts。数据分析可以在界面内筛选变量/工况数据、查看分类叶片数统计、对比历史预测与真实值、查看误差/不确定度及不同协议下的交叉验证记录。算例浏览支持按来源、状态和关键词检索，核对输入参数和结果证据，查看文件、日志、JSON 和图片，并从在线预测点跳转到对应算例。

“打开结果配置…”仅切换这两个页面的只读查看工程，不改变实际运行配置或实时任务页。页面显示期间可每 15 秒后台刷新，也可关闭自动刷新；切换工程/算例后不会接收旧读取任务的迟到结果。跨阶段同名算例、来源不明的真实值、缺失数据不会被自动拼接或补造。此次增加的是离线读取、统计与界面操作能力，不修改采集/CV 协议、性能 schema、模型或 scaler，已有模型无需重新训练。
