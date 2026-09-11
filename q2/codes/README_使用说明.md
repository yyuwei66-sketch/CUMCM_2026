# P2 最终代码包

本包只保留第二问当前需要补齐的代码，不包含旧 W30/W120 过渡版本。

## 1. p2.py —— 直接上交版

这是第二问最终主程序，已做完整运行测试。

默认目录结构：

q2/
├─ p2.py
├─ p2_joint_search.py
├─ p2_stability.py
├─ p2_probability_calibration.py
└─ input/
   ├─ q2_actual_load_pv.csv
   ├─ q2_fixed_prices.csv
   └─ result2_template.xlsx

直接运行：

python p2.py

默认输出：

- result2.xlsx
- results_final/qstar_development_scan.csv
- results_final/final_strategy_daily.csv
- results_final/final_strategy_slots.csv
- results_final/final_emergency_events.csv
- results_final/verification_summary.json

p2.py 已验证重新得到：

- q* = 0.85
- 334 天
- 48,096 个 10 分钟时段
- 计划购电量 = 22,076,485.272805803 kWh
- 紧急购电量 = 225,320.99023033754 kWh
- 计划费用 = 13,456,454.710735613 元
- 紧急购电费用 = 876,478.6500762145 元
- 总费用 = 14,332,933.36081183 元

并成功生成 result2.xlsx。

## 2. p2_joint_search.py —— 联合搜索 + Profile 复现代码

完整候选空间：

- W ∈ {30,60,90,120}
- K_L ∈ {1,2,3,4,5}
- K_P ∈ {1,2,3,4}
- h_L,h_P ∈ {H0,H1,H2,H3}
- Load calendar ∈ {fri_sat, weekday7, fri_sat+annual1, fri_sat+annual2}
- PV calendar ∈ {annual1, annual2, weekday7+annual1, weekday7+annual2}
- α_L, α_P ∈ {0.1,0.3,1,3,10,30,100}

公共验证区：

2025-05-01 至 2025-08-31

目标：

pooled Net Load RMSE

完整运行：

python p2_joint_search.py

快速环境检查：

python p2_joint_search.py --smoke

smoke 已实际验证得到：

W=60, K_L=3, K_P=2,
h_L=lag1,
h_P=lag1+lag7+mean7,
Load calendar=weekday7,
PV calendar=annual1,
α_L=0.3, α_P=10,
RMSE=392.623240 kW。

## 3. p2_stability.py —— Top 配置跨月稳定性

先运行：

python p2_joint_search.py

再运行：

python p2_stability.py

输出 Top10×各W 的 May/Jun/Jul/Aug 月度 RMSE 和稳定性指标。

smoke 模式对应的 W=60 Top10 已实际运行通过，
rank1 monthly mean RMSE = 392.0820 kW。

## 4. p2_probability_calibration.py —— 共形校准复现代码

运行：

python p2_probability_calibration.py

统一重算：

M ∈ {14,28,42,60}

最新结果：

- M=14 calibration error = 0.0009146341
- M=28 calibration error = 0.0053297200
- M=42 calibration error = 0.0074751581
- M=60 calibration error = 0.0089430894

最终仍选择：

M=14

这已经修正之前 M=60 内嵌表无法完全复现的问题。

Sep-Dec：

- CRPS = 145.847774 kW
- 80% coverage = 80.5442%
- 90% coverage = 90.1639%

## 5. 论文口径

- W=60 表示最大历史窗口长度；启动阶段历史不足60天时使用全部已完成历史。
- 残差场景池最多28个已完成日；启动阶段使用当时可获得的全部残差。
- q*=0.85 在 May-Aug 开发期离线标定后用于全年策略回放。
- Sep-Dec 用于冻结预测模型的样本外评价。
