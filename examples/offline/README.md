# 离线复现：解析合成数据 → 代理训练 → NSGA-II → Pareto 查询

在仓库根目录执行（macOS / Linux，建议 Python 3.13 或 3.14）：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gui.txt
python -m impeller_app.reproduce --config examples/offline/config.json
python -m unittest discover -s tests -v
```

输出默认写入 `offline-output/`，目录必须为空；再次运行请指定新的目录，例如
`--output-dir offline-output/repeat`。配置路径和输出路径均相对于命令启动目录解析。
运行时忽略 `IMPELLER_*` 环境变量，固定 CPU 单线程、PyTorch 确定性算法及随机种子，
生成独立的设计变量配置、训练数据、固定测试集与模型包，不读取作者的训练集或旧权重。
安装依赖的网络耗时不计入运行时限；复现本身应在 10 分钟内完成。

`generate.py` 是解析数据生成函数，使用固定种子 LHS，按变量精度取整，再按共享几何硬规则拒绝采样，
补足 384 行。已提交的 `Compressor_Training_Data.csv` 与 `.csv.meta.json` 是该配置的示例快照；
命令每次从函数重新生成同一数据。所有数值都是**合成示例，不是 CFD 结果或真实设计性能证据**。
schema-v2 元数据表明字段语义，不能证明数据来自 CFD。真实数据和 Windows 工程机 DOE / 主动学习不在此入口中运行。

令 `z_j = (x_j-lower_j)/(upper_j-lower_j)`，按下表变量顺序从 0 编号，
`L = mean((z_1...z_10 - 0.5)^2)`，解析输出为：

```text
Efficiency          = 0.68 + 0.10*z_0 - 0.025*L + 0.005*z_12
PressureRatio       = 1.05 + 0.20*z_4
Power               = 80 + 25*z_4 + 8*z_11
MassFlow            = 3.40 + 0.90*z_5 + 0.30*z_11
totalpressureratio  = 1.35 + 0.55*(1-z_0) + 0.15*z_4
is_boundary         = int(MassFlow < 3.6)
```

## 变量与工况

代码来源：`design_variables.DEFAULT_VARIABLE_SPECS`。范围含端点，`nBl` 取整为 9、10、11、12；
其余变量按 `decimals` 保留精度。尺寸沿用仓库/CFturbo 模板数值单位；离线计算不做尺寸单位转换。
实际模板未入库，真实求解前需核对模板尺寸单位。角度输入为度，脚本将相关角度转换为弧度写入模板。

| 序号 | 字段 | 范围 | 小数位 |
|---|---|---|---|
| 0 | d1s | 0.34–0.42 | 5 |
| 1 | dH | 0.044–0.056 | 5 |
| 2 | beta1hb | 70–84 | 3 |
| 3 | beta1sb | 20–35 | 3 |
| 4 | d2 | 0.45–0.53 | 5 |
| 5 | b2 | 0.044–0.056 | 5 |
| 6 | beta2hb | 40–54 | 3 |
| 7 | beta2sb | 40–55 | 3 |
| 8 | Lz | 0.185–0.215 | 5 |
| 9 | t | 0.0015–0.0035 | 6 |
| 10 | TipClear | 0.0007–0.0015 | 6 |
| 11 | nBl | 9–12，整数 | 0 |
| 12 | rake_te_s | -25–-15 | 3 |

`P_out = 12 Pa` 是出口静压工况，作为第 14 个代理输入，但不是优化设计变量。
元数据中的历史 8–13 Pa 范围不限制固定工况值；观测数据查询只接受距离设定值 ≤0.25 Pa 的样本。
本示例所有行恰为 12 Pa。

## 目标、约束和排序

物理目标是同时最大化 `Efficiency` 和 `totalpressureratio`，约束 `MassFlow >= 3.6 g/s`。
`PressureRatio` 是静压比，不作为优化目标；总压比没有上下限约束。模型只预测效率、总压比和质量流量，
不预测 `Power`。

现有 `CompressorMOOProblem` 实际送给 pymoo 的最小化目标包含风险罚项：

```text
F1 = -clip(mean_eff, 0.45, 0.84) + 0.12*u + 0.035*overlap + 0.20*(1-p_geom)
F2 = -mean_totalpr              + 0.12*u + 0.035*overlap + 0.20*(1-p_geom)
```

`u` 为前两个输出归一化 MC-Dropout 标准差的均值；`overlap` 为非负角度差：
`max(0, rake_min[nBl]-rake_te_s)`，9/10/11/12 叶片的 `rake_min` 分别为 -18/-19/-20/-21 度。
这是软罚项，不是几何硬约束。每次候选评估采样 20 次 Dropout，前沿重新提取采样 50 次，
最终 CSV 的预测均值/不确定度使用 `CFG.mc_samples`。多次重采样的均值会有区别，导出行不表示 CFD 可行性验收。

NSGA-II 硬约束（违反量 ≤0）：

- 预测质量流量 ≥3.6 g/s，预测效率在 [0.60, 0.84]。
- 求解可行概率 ≥0.65，几何安全概率 ≥0.55。本例没有失败样本，两者默认 1，不训练失败分类器。
- `d2-d1s >= 0.070`；`beta1hb-beta1sb <= 52` 度；`abs(beta2hb-beta2sb) <= 13.5` 度。
- `b2 >= lower_b2`，`t <= upper_t`，`TipClear`、`Lz`、`rake_te_s` 在各自范围内。
  所有设计变量还受 NSGA-II 上下界限制。

解析 DOE 数据查询使用现有 `pareto_front_query`：工况带、同一几何规则、效率 ≥0.45、流量 ≥3.6，
然后对效率/总压比提取非支配前沿。已成功的数据不因失败分类器被否决。工程排序分数是
`0.32*eff_margin + 0.28*flow_margin + 0.22*knee_score + 0.18*p_geom
- 0.10*nearest_success_dist - 0.06*stability_penalty`；各项先除以前沿中的最大值，
分母最小为 `1e-8`。此排名只用于推荐，不能替代物理双目标前沿。

## 输出定义

| 文件/字段 | 含义 |
|---|---|
| Compressor_Training_Data.csv | 13 个几何输入、P_out 和解析输出；全叶轮口径 |
| Efficiency | 无量纲等熵压缩效率；本例为解析函数值 |
| PressureRatio | 出口/进口质量流量平均**静压**之比，无量纲 |
| totalpressureratio | 静止系质量流量平均出口总压 / 进口总压，schema-v2，无量纲 |
| Power | 轴流体功率 W；真实链路按静止系总焓升×流量计算，本例为解析值 |
| MassFlow | 全叶轮质量流量 g/s；本例已按全叶轮口径生成，无需再乘 nBl |
| is_boundary | 流量低于 3.6 的保留边界样本标志；不等于几何失败 |
| fixed_test_set.csv | 15% 固定留出集；不参与梯度训练、早停选择和 scaler 拟合 |
| nsga2_doe_model/ | 新训练权重、scaler_X/Y 和绑定数据/模型散列的 manifest |
| nsga2_surrogate_pareto.csv | 代理 NSGA-II 结果：front_index 从 1 开始；几何输入、P_out、pred_Efficiency、pred_totalpressureratio、pred_MassFlow、uncertainty_norm、surrogate_hv |
| uncertainty_norm | 前两个输出归一化 Dropout 标准差均值，无量纲，不是原单位置信区间 |
| surrogate_hv | 以效率 0.6、总压比 1.0 为参考的预测双目标超体积，整列相同，不是 CFD HV |
| pareto_front_points.csv | 解析 DOE 可行非支配前沿；front_index 从 0 开始，与代理表索引无关联 |
| pareto_engineering_ranked.csv | 上述解析前沿按 engineering_score 降序；engineering_rank 从 1 开始 |
| pred_geom_safe_prob | 几何安全概率，本例无失败分类器时为 1 |
| nearest_success_dist | 到其他解析样本的归一化距离，反映数据支持 |
| knee_score | 前沿到端点连线的归一化距离 |
| stability_penalty | 邻近样本输出离散度罚项 |
| flow_margin / eff_margin | 相对 3.6 g/s / 0.6 的非负裕量 |
| pareto_selected_point.json | 查询工程排名第一的解析行；沿用 API 的 true_outputs 字段名，此处内容是合成解析值 |
| summary.json | 数据来源、环境版本、配置、推荐行、误差、HV、模型身份与运行耗时 |
| *.csv.meta.json | schema_version=2 与 stationary_frame_mass_flow_averaged_total_to_total 定义 |

训练池还会分出早停验证集；scaler 和样本密度/nBl 权重只拟合梯度训练分区。
数据种子和 NSGA-II 种子由配置指定（42），固定测试/验证划分种子沿用现有实现（42），训练种子沿用 1000。

## 预期数值与容差

参考环境：macOS arm64，Python 3.14.3，NumPy 2.4.2、Pandas 3.0.1、SciPy 1.17.1、
scikit-learn 1.8.0、PyTorch 2.11.0、pymoo 0.6.1.6。配置为 seed=42、384 行、
240 epoch、patience=40、population=48、32 代。主流程约 1.4 秒；冷导入与设备性能会增加总耗时。

| summary.json 路径 | 参考值 | 验收绝对容差 |
|---|---|---|
| samples / nsga2.train_samples / nsga2.test_samples | 384 / 326 / 58 | 精确 |
| analytic_front_size | 50 | 精确（同一解析数据） |
| analytic_max_efficiency | 0.7807738769 | 1e-8 |
| analytic_max_total_pressure_ratio | 2.01885 | 1e-8 |
| selection.engineering_score | 0.8300447961 | 1e-6 |
| selection.true_outputs.Efficiency | 0.7486677356 | 1e-8 |
| selection.true_outputs.totalpressureratio | 1.6819375 | 1e-8 |
| selection.true_outputs.MassFlow | 4.49925 g/s | 1e-8 |
| nsga2.fixed_test_rmse_eff | 0.00384668 | 0.001 |
| nsga2.fixed_test_rmse_pr | 0.02401822 | 0.01 |
| nsga2.fixed_test_rmse_mf | 0.04061173 g/s | 0.02 |
| nsga2.surrogate_hv | 0.15848537 | 0.03 |
| nsga2.front_size | 30（参考环境） | 跨版本要求 1–48；集合基数不作逐行相等保证 |
| nsga2.epochs | 240（参考环境） | 1–240（早停） |

同一依赖栈连续两次复现的 CSV 数值应在 `atol=1e-7, rtol=0` 内一致；
跨平台/依赖版本用上表比较指标。模型序列化散列、绝对路径、耗时不作跨机器相等要求。
解析示例数据也依赖 SciPy LHS 实现；更换采样算法/重大版本需重新建立参考基线，
`summary.json.environment` 记录实际版本供核对。

干净 macOS Python 3.14.3 虚拟环境也已验证：NumPy 2.5.4、Pandas 3.0.6、SciPy 1.18.1、
scikit-learn 1.9.1、PyTorch 2.14.1、pymoo 0.6.2、PySide6 6.12.0；96 个测试通过、0 跳过。
该环境默认复现约 11 秒，代理前沿为 31 行、HV 为 0.15713864，仍满足上表容差。

GUI 测试需要 PySide6。Linux 无桌面环境时使用 `QT_QPA_PLATFORM=offscreen`；
Debian slim 可先安装 `libegl1 libgl1 libxkbcommon0 libdbus-1-3 libglib2.0-0 libfontconfig1`。
用例因缺依赖跳过时应单列报告，不能把跳过当作该能力验证通过。

## 数值回归与 CI

`python -m unittest tests.test_reproduce -v` 运行真实的离线训练/优化，
比对入库的 384 行合成数据、50 行解析前沿、推荐点的三项输出、模型误差、HV 和代理目标极值。
代理最大效率参考为 `0.775687 ± 0.01`，最大总压比为 `2.040334 ± 0.05`；
这些模型容差允许 CPU/依赖版本带来的小幅差异，解析输出仍使用上表的严格容差。
代理行数须与摘要一致且介于 1 和种群大小 48 之间：离散前沿成员在不同依赖栈下会改变，
因此结合目标极值与 HV 检查前沿，而不强制 30 行逐项相等。

现有 CI 增加 Ubuntu Python 3.11，保留其余平台和版本，并在 Python 3.11 上做语法检查。
手工验证测试敏感性时，可临时把 `config.json` 中的 `seed` 从 42 改为 43，
运行 `python -m unittest tests.test_reproduce.OfflineReproductionTests.test_documented_default_metrics -v`；
应返回非零，恢复 42 后再次运行应通过。参考数据不会随运行自动更新。
