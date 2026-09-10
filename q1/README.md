# 第一问：Notebook与脚本

**推荐先打开Q1_microgrid.ipynb。**它包含中文说明、可编辑参数、完整模型代码、数据表、约束检查和三张图，已经保存执行输出。

Notebook内置全部144条输入数据，单独下载该文件也能运行。使用VS Code打开后，右上角选择`cumcm2026`对应的Python内核，点击`Run All`，或按`Shift+Enter`逐格运行。不要只修改参数而保留后续旧结果，修改后应重新运行后续所有单元格。

如果找不到内核，在已激活的环境中执行`python -m pip install ipykernel`，然后重新选择内核。Notebook结果写入当前工作目录的`q1_notebook_results`，与下面脚本的`results`分开。

Notebook的所有Python单元格已顺序执行，保存了真实表格与图像输出；制作环境未通过Jupyter内核执行协议做额外测试。文件内元数据记录了这一验证范围。

以下步骤是保留的脚本运行方式，便于后续批量实验。

本包包含已处理的第一问输入、求解和校验脚本，以及已运行产生的结果。可独立使用，不需要重新下载预处理包。

## 现在做这几步

1. 将压缩包解压，把整个`CUMCM2026_C_Q1`文件夹放到`E:\桌面\CUMCM_2026`下。
2. 在PowerShell中执行：

```powershell
conda activate cumcm2026
cd "E:\桌面\CUMCM_2026\CUMCM2026_C_Q1"
python run_q1.py
python check_q1.py
```

如果提示缺少模块，再运行 `python -m pip install -r requirements.txt`，然后重试。无需GPU或Torch。

正常输出应包括：

```text
Q1 SOLVED: PASS
Cost: 35126.9486 yuan
Grid purchase: 59482.6990 kWh
Savings vs no storage: 26.8981%
Physical checks: PASS
Independent checks: PASS (26 checks)
```

运行只重建本包results中的对应结果，不修改输入，不接触此前预处理包。代码按脚本自身位置定位文件，即使从其他目录运行绝对脚本路径也可以。

## 接着查看哪些文件

| 文件 | 用途 |
|---|---|
| results/第一问结果说明.md | 先看总费用、节省比例与限制 |
| results/dispatch_10min.csv | 144时段的完整购电、充放电、储能和费用 |
| results/paper_table1.csv | 题面指定6个时段的购电量 |
| results/paper_table2.csv | 每4小时充放电汇总及边界储能 |
| results/storage_states.csv | 145个边界时刻的储能电量 |
| results/baseline_no_storage.csv | 无储能对照，仍利用光伏满足负载 |
| results/figures/01_load_pv.png | 负载与光伏 |
| results/figures/02_dispatch.png | 购电与充放电，放电用负方向表示 |
| results/figures/03_storage_price.png | 储能电量与电价 |
| results/summary.json | 未舍入汇总、假设、求解与物理校验信息 |
| results/independent_checks.json | 独立按输入重算的验证记录 |

CSV保留完整计算精度，展示或论文填表时再四舍五入。

## 模型与代码的对应

所有决策以kWh计量，144个时段，每段1/6小时。输入功率乘1/6得到区间电量近似。

- 目标：`sum(price[t] * grid[t])`最小。
- 电量平衡：`grid + PV + discharge = load + charge + curtailment`。
- 储能更新：`E_next = E + 0.9*charge - discharge/0.9`。
- 储能范围：1200至10800kWh。初始与终止储能都为6000kWh。
- 充电/放电单时段上限：5000/6kWh。
- 不售电；允许弃光；储能不允许同一时段同时充放电。

`run_q1.py`中的`solve_dispatch()`构造并求解模型，`check_solution()`校验，`make_figures()`绘图。
`check_q1.py`从输入和输出文件独立重算电量与费用，不调用优化器。

默认先求线性规划松弛。若得到的最优解满足充放电互斥，该解就同时达到严格MILP模型最优值。否则脚本自动增加二元变量并求解MILP。也可以手动运行 `python run_q1.py --solver milp`。

参数集中在parameters.json。修改参数后必须重新求解和校验。`interval_minutes=10`是当前输入粒度，不能只改这个参数就改变输入的时间分辨率。

## 本次计算的假设仍需写入论文

1. 暂把采样时刻t的数据看作此前10分钟区间的平均功率；这不是题面明确确认的采样含义。
2. 充电效率和放电效率分别为90%。
3. 日初、日末电量均取6000kWh。
4. 允许弃光，不允许售电；未自行增加退化成本、自放电或购电功率限制。

原官方模板时间标签存在歧义。本包使用明确的物理时间标签，输出的是计算与论文核对结果，**未生成官方result1.xlsx**。应在确认时间口径后统一导出，不能把CSV按行号直接填入原模板。

同一4小时汇总内充放电都非零，不表示同时充放电；它们可能发生在不同10分钟时段。不同求解器版本可能给出等价最优安排，核对时优先检查总费用和物理约束，而非要求每个决策逐格一致。

## 团队当前交付

编程手：本地运行通过，读懂三张图，能解释每项约束的数值检查。

建模手：确认上述假设与变量单位，核对约束和最优性解释。

论文手：先使用结果表和图搭建第一问的模型、结果与验证部分，并明确本次假设。模板时间问题解决后再统一最终表述与导出。
