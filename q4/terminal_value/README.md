# Q4-2 期末储能价值实验

独立实验，不修改已有提交程序或结果。研究对象是：用决策时已知的未来预测，学习期末储能的边际价值，以凹二次奖励指导当天购电。

- 输入：当天 Q2 的0.85分位净负荷、前一周同日电价、未来六天的周复制供需与价格预测。
- 标签：未来六天优化对七个起始电量计算的预计节费，约束未来总末端回到6000 kWh。
- 输出：线性单位价值，或二次奖励在低、高电量端的两个边际价值。
- 执行：固定计划购电，富余先充、缺口先放，不足再紧急购电；各策略采用相同规则。

## 使用

打开 [实验 Notebook](Q4_2_期末储能价值实验.ipynb)，或从项目根目录运行以下命令。首次安装隔离求解器：

```sh
/Users/jinyu/miniconda3/envs/mcm/bin/python -m pip install --target .tmp/q4_terminal_value/deps --no-deps clarabel==0.11.1 osqp==1.1.3
```

完整复现：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/jinyu/miniconda3/envs/mcm/bin/python q4/terminal_value/experiment.py --stage all
PYTHONDONTWRITEBYTECODE=1 /Users/jinyu/miniconda3/envs/mcm/bin/python -m unittest discover -s q4/terminal_value -p 'test_*.py' -v
```

缓存、图表、逐时记录和模型文件全部写入 `.tmp/q4_terminal_value/`。代码与数据指纹变化时缓存失效，普通重跑复用已有计算。`REPORT.md` 是实验产生的正式中文报告。

`--stage` 支持 `labels`、`diagnostics`、`validate`、`evaluate`、`explore`、`report`、`all`。除标签与敏感性阶段外，其余阶段会检查是否已完成选参；分阶段使用时先完成其依赖。

## 文件职责

- `core.py`：因果数据、标签、特征、学习模型、LP/QP 日计划与实际执行。
- `experiment.py`：分期选参、冻结评价、全年探索、敏感性检查和报告。
- `test_terminal_value.py`：经济含义、因果性、优化等价性和执行规则的回归测试。

公开接口以 kWh 为能量单位，内部求解缩放为 MWh。状态更新为 $$E_{t+1}=E_t+0.9c_t-d_t/0.9$$。低于1200 kWh的电量不计入可用储能奖励。

学习模型按日期严格分期。全年探索使用开发期所选超参数，不可作为独立全年样本外证据。直接估值对照与学习模型信息相同，并非知道真实未来的理想下界。老师只估计确定性正常购电费用，未包含未来紧急风险。
