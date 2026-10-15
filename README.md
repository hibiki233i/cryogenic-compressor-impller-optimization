# BOUNDYR：低温压缩机叶轮优化工作台

BOUNDYR 用来给低温压缩机的叶轮找到更好的几何形状。它的思路是：先在 CFD（流体仿真）里算一批样本，
用这些样本训练一个"快速预测模型"，再让优化算法在模型上大范围搜索，最后挑出少数最有希望的设计，
回到 CFD 里验证，并把验证结果补充给模型，继续下一轮。整个过程有一个桌面界面（PySide6）串起来，
不必手动敲脚本。

## 在整条链路中的位置

本仓库是三个仓库中的第一步：

```
叶轮优化（本仓库） ──→ 叶片优化 ──→ 性能评估 / 特性图（workingmap 仓库）
```

- 本仓库输出"推荐的叶轮设计"（Pareto 前沿表，即在效率和压比之间无法再两全的最优设计集合）。
- [workingmap](https://github.com/hibiki233i/workingmap) 读取其中选定的一个设计，扫描不同转速和背压，画出压气机特性图。
  对接方法见 workingmap 仓库的 `docs/from_optimization.md`。

## 两种使用方式，先分清

| | 离线快速开始 | 真实 CFD 运行 |
|---|---|---|
| 适用系统 | macOS / Linux / Windows 都行 | **只能在 Windows 上**，且装有商业软件 |
| 需要 CFD 软件吗 | 不需要 | 需要 ANSYS CFX、CFturbo 等 |
| 数据 | 程序自带的合成数据 | 你自己的 CFD 结果 |
| 用途 | 确认环境装好、整条流程能跑通；验证结果可复现 | 真正的设计优化 |

离线数据是用公式编出来的"示例"，**不是 CFD 结果，不能当作设计性能的依据**。

## 环境要求

### Python

- Python 3.11 或更新（自动测试覆盖 3.11 到 3.14；作者在 macOS 上用 3.14.3 验证过）。
- 依赖写在 `requirements-gui.txt`，包括界面库 PySide6，以及 numpy、pandas、scipy、scikit-learn、
  matplotlib、joblib、psutil、pymoo、torch、SALib。其中 torch 体积较大，安装需要一些时间。
  即使只做离线快速开始，也需要装全这些依赖。

### 外部软件（只有真实 CFD 运行才需要）

| 软件 | 已验证的版本 | 说明 |
|---|---|---|
| ANSYS CFX | 2025 R1（v251） | 求解与后处理 |
| CFturbo | 2025.2.2 | 生成叶轮几何 |
| TurboGrid | 版本待确认 | 划分网格 |
| PowerShell | 7 | 驱动几何和网格脚本 |
| Windows | — | 以上软件的运行平台 |

### 需要自备的模板（未随仓库提供）

真实运行前，需要准备这些文件，并在界面"环境"页填好路径（`windows-app-config.template.json` 是一份路径清单示例）：

- CFX 模板（`.cfx`）
- 结果提取脚本模板（`.cse`；仓库里的 `cfx_post/Extract_Results.cse` 可作参考）
- CFturbo 基础工程（`.cft`）和批处理模板（`.cft-batch`）
- 工作目录：保存样本、运行结果和训练数据的位置

## 离线快速开始

在仓库根目录执行。macOS / Linux：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gui.txt
python -m impeller_app.reproduce --config examples/offline/config.json
python -m unittest discover -s tests -v
```

Windows（PowerShell）：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gui.txt
python -m impeller_app.reproduce --config examples/offline/config.json
python -m unittest discover -s tests -v
```

如果 PowerShell 提示禁止运行脚本，可先执行 `Set-ExecutionPolicy -Scope Process RemoteSigned`（仅对当前窗口有效）。

其中 `reproduce` 这一步是"离线复现"：用固定随机种子生成 384 个合成样本，训练预测模型，再做一次完整的多目标优化，
最后在结果里挑出推荐设计。通常几秒到十几秒完成。结果写入 `offline-output/`（目录必须为空，
重复运行请用 `--output-dir` 指定新目录）。

最后一步运行全部自动测试，不需要任何 CFD 软件。在没有桌面环境的 Linux 上，请先设置环境变量
`QT_QPA_PLATFORM=offscreen`。示例的详细说明、参考数值和允许误差见 [离线复现指南](examples/offline/README.md)。

## 真实 CFD 运行（Windows）

1. 安装上面列出的外部软件，并准备好自备的模板。
2. 在 Windows 上运行 `launch_gui.bat`（它会自动找 Python、安装依赖并打开界面），
   或手动执行 `python -m impeller_app`。
3. 在"环境"页填写各软件和模板的路径，点击检查。设置会自动保存，下次启动自动读取
   （保存位置 `%APPDATA%\BOUNDYR\impeller-app-config.json`）。
4. 按界面左侧的工作流依次进行：初始采样（DOE）→ 主动学习 → 查看 Pareto 前沿 → 敏感性分析 → 导出算例。

每一步怎么做、界面上各页的用途见 [使用指南](docs/user_guide.md)。

## 仓库里有什么

| 位置 | 作用 |
|---|---|
| `impeller_app/` | 桌面程序本体：配置、任务调度、核心算法服务、界面 |
| `NN_NSGA2_ActiveLearning_refactored.py` | 优化引擎：模型训练、主动学习、多目标搜索（早期脚本，界面在其外面包了一层） |
| `DOE.py` | 初始采样流程（早期脚本） |
| `cfx_runner.py`、`cfx_convergence.py` | 调用 CFX 求解、检查收敛、必要时续算 |
| `pareto_front_query.py`、`pareto_export_cft_cases.py` | 在 Pareto 前沿上查询设计点，并导出成 CFturbo 算例 |
| `prepare_al_stage.py` | 为独立的主动学习实验建立新阶段 |
| `final_checkpoint_evaluation.py` | 不跑 CFD，只重新训练并评估模型 |
| `Run-GeometryMeshing.ps1` | 几何与网格生成脚本（PowerShell） |
| `cfx_post/` | CFX 后处理提取脚本 |
| `examples/offline/` | 离线复现示例与参考数据 |
| `tests/` | 自动测试 |
| `docs/` | 详细文档（见下） |
| `.github/workflows/ci.yml` | 每次提交自动运行的检查 |

## 文档索引

- [使用指南](docs/user_guide.md)：桌面程序怎么用，每一页是做什么的
- [优化问题说明](docs/optimization_problem.md)：优化什么、限制是什么、数据怎么约定
- [主动学习说明](docs/active_learning.md)：模型如何挑选下一批要算的设计，如何记录和检验模型的可靠性
- [CFX 收敛检查与续算](docs/cfx_convergence.md)：怎样判断一个算例"算完了"
- [离线复现指南](examples/offline/README.md)：示例数据、参考结果与验收误差
- [更新记录](docs/changelog.md)：按时间整理的功能变化

## 测试与自动检查

- 本地：`python -m unittest discover -s tests -v`，使用临时目录和模拟的求解器，不需要 ANSYS 许可证。
- 自动检查（GitHub Actions）：每次提交和合并请求会做语法检查，并在 Ubuntu（Python 3.11 至 3.14）
  和 Windows（Python 3.12）上运行全部测试与离线复现。

## 注意事项

- 界面和求解链路面向 Windows 工程机；macOS / Linux 适合开发、阅读结果和运行离线测试。
- 模板文件、训练数据和历史模型权重不随仓库提供，需要自行准备。
- 仓库里的 `best_regressor.pth`、`failed_points.npy` 是早期遗留文件，离线示例不使用它们。
