# 第二问：从 Notebook 开始

1. 解压完整包。在 VS Code 打开 `Q2_microgrid.ipynb`，选择 `cumcm2026` 内核。
2. 看参数单元后从上往下运行。Notebook 内嵌数据，不依赖第一问或原先预处理目录。
3. 先看策略比较和四个指定日期，再结合 `report/Q2_Report.md` 写论文。

已经计算完整 365 天过程，评价 2—12 月 334 天。主策略是 28 日历史情景日前优化与实时储能执行；七日均值、无储能是比较策略。

## 三个需要理解的口径

- 普通购电按 00:00 确定的全部计划量付费，应急额外按 5 倍价付费。
- 实际电池状态跨日连续。6000 kWh 是计划末态目标，实际末态不强制回到 6000。
- 模板时间列存在一列错位，本版本用 `00:00-00:10` 至 `23:50-24:00`。`result2_时间口径暂定.xlsx` 含明确说明，正式提交前需要确认题目解释。

## 脚本运行

```bash
python run_q2.py
python check_q2.py
```

如需从原始附件重新提取：

```bash
python prepare_inputs.py
python run_q2.py
python check_q2.py
python build_report.py
```

默认 Python 依赖见 `requirements.txt`，在自己的环境缺什么再安装。制作环境完成了全部 Python 单元的顺序执行，没有测试 Jupyter 内核协议。

Notebook 写入 `q2_notebook_results/`；脚本写入 `results/`。两个目录的默认结果完全一致。修改参数重新运行后会更新 CSV/JSON/PNG，原 Excel 和 Markdown 报告不会自动更新。Excel 导出脚本 `export_workbook.mjs` 需要 `@oai/artifact-tool` Node 依赖，读取脚本结果目录的 JSON，不参与数值优化。

## 文件

- `input/`：从原始附件重建的负载、光伏和固定价格，无附件 3、4 数据。
- `raw/`：附件 1、2 原样副本。
- `templates/`：原 result2 模板。
- `results/`：主策略和两种对照的全年明细、日报、论文表格与核验。
- `report/`：Markdown 报告及相对链接图片。
- `verification/`：导出和 Notebook 运行核验记录。

这是可复现的完整工作版本。日前模型对有限历史情景目标求优，实时控制为因果启发式，不能宣称整个随机控制问题的全局最优。
