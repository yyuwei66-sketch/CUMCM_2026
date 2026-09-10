# CUMCM 2026 C题预处理数据包

这份数据包已包含处理后的数据，可直接使用，不需要先重新运行。原始附件和结果模板均保留原样。

## 开始使用

把整个文件夹放到你的项目中，例如 `E:\桌面\CUMCM_2026\CUMCM2026_C_preprocessed`。
在当前文件夹打开终端，激活已有环境：

```powershell
conda activate cumcm2026
python src/verify.py
python src/data_access.py
```

如果提示缺依赖再执行 `python -m pip install -r requirements.txt`。此流程不需要GPU、Torch或CatBoost。
重新生成标准数据：

```powershell
python src/preprocess.py
python src/verify.py
```

脚本将重建processed和diagnostics中的相应派生文件，不会修改raw和templates中的Excel。脚本所在文件夹用于定位数据，因此从其他工作目录运行绝对脚本路径也可以。

## 先读这四条

1. **模板时间标签仍有歧义。**原采样时刻00:10到次日00:00，模板却从00:10–00:20开始。本包暂按“采样值代表前10分钟区间平均功率”生成区间和近似电量。这是模型假设，不是已经确认的官方解释。`sample_time`和来源坐标完整保留，不能按位置直接导出正式模板。
2. **预报按发布版本保留。**同一目标时刻有多次预报是正常情况，不能按目标时刻简单去重，否则早上的决策可能拿到晚上的预报。
3. **实际标签不能作为预测输入。**使用`day_ahead_features.csv`做日前特征，使用单独的标签表做历史训练和事后评价。训练时仍需限制训练标签在训练时已可获得。
4. **没有擅自删除异常值。**光伏零值、负净负载、低电价和真实峰值全部保留。5000kW是储能充放电功率限制，不是负载/光伏上限。

## 文件用途

| 文件 | 内容 | 使用场景 |
|---|---|---|
| processed/q1_typical_day.csv | 144个典型日点，含原功率/电价及近似电量 | 第一问 |
| processed/actuals_10min.csv | 全年52560条实际负载、光伏、电价及净负载 | 回测运行、历史训练、事后评价 |
| processed/pv_forecast_hourly.csv | 原始35040个整点预报，保留发布时间与目标时间 | 第三问、第四问对应第三问 |
| processed/pv_forecast_10min.csv | 210240个按版本独立插值的10分钟预报 | 调度输入；这是派生数据 |
| processed/day_ahead_features.csv | 2—12月48096条日前特征 | 第二、三、四问的历史预测特征 |
| processed/evaluation_labels_DO_NOT_USE_AS_FEATURES.csv | 同期实际负载、光伏、净负载、电价标签 | 仅历史训练标签和事后评价 |
| processed/evaluation_calendar.csv | 1月初始历史、2—12月评估及指定报告日期 | 按日回测 |
| data_dictionary.csv | 所有输出字段、类型、单位和说明 | 查字段 |
| diagnostics/预处理检查报告.md | 处理统计与剩余问题 | 团队对齐 |
| diagnostics/template_time_mapping_UNRESOLVED.csv | 原模板标签与暂定物理区间对照 | 最终导出前核对 |
| diagnostics/review_flags_NOT_FOR_TRAINING.csv | 全年3IQR描述性标记 | 仅人工审阅，不作训练特征或删除依据 |
| manifest.json | 原始文件校验值、输出校验值、假设、运行版本 | 追溯与复现 |
| src/preprocess.py / data_access.py / verify.py | 预处理、按时刻取数、独立验证 | 本地复现 |

CSV使用UTF-8 BOM编码，Excel和pandas均可读取；无额外索引列。计算保留原值和浮点精度，最终结果展示时再四舍五入。

## 时间、单位与预报插值

- 时间按文件本地时钟处理，无时区转换，不使用运行电脑所在地或用户所在地。
- 每日144个记录按来源日期与slot=1..144关联。来源日期的次日00:00仍归前一来源日，不能按`sample_time.dt.date`直接重新分日。
- 暂定区间为`[sample_time-10min, sample_time)`。所有`*_kwh_approx`为对应功率除以6。
- **预报发布时间＋lead_hours＝target_time**。不会把所有预报的“预报1小时”都当作01:00。
- 10分钟预报在同一发布版本内分段线性插值，整点值与原预报完全一致，不做平滑或外推。
- 插值的0小时锚点优先采用发布时间的已观测光伏，假设该测量当时已可用。缺少该时刻观测时用首小时预报作为平坦回退锚点，并标注来源。首个2025-01-01 00:00版本需要此回退。
- 插值点当作区间平均功率来计算电量是另一个数值近似。它不保证与整点曲线梯形积分完全相等，也不增加真实预测精度。
- 年末未来24小时预报会越过2025年；本包全部保留，`target_in_actual_coverage=false`表示缺乏对应实际值，不能把它填成实际0。

## 防止未来数据泄露

日前特征包含日历字段、负载/光伏/电价的前1天、前7天值，以及截至前一天的7天、28天同期均值；它们不包含当天实际值。1月作为初始历史，因此2月1日已具备28天窗口。

`latest_history_sample_time <= origin_time`在全部特征行上成立。这里假设测量无报告延迟。若团队认为00:00时上一个区间的测量尚不可得，需要加可用性延迟后重新设计特征与锚点。

已提供的访问器：

```python
import sys
from pathlib import Path
import pandas as pd

root = Path(r"E:\桌面\CUMCM_2026\CUMCM2026_C_preprocessed")
sys.path.insert(0, str(root / "src"))
from data_access import CData

data = CData(root)
origin = pd.Timestamp("2025-03-20 00:00")
history = data.history(origin)  # 仅当时及此前的实际记录
features = data.day_ahead_features(origin, problem=3)
pv = data.pv_forecast(origin, features["interval_end"])  # 每个目标使用已发布的最新版本
```

访问器不能替代训练流程的时间约束：在日期D训练时，只能用D之前已揭晓的标签拟合模型、缺失值填充器或标准化器。不要把2—12月随机打乱训练，不要先用全年数据拟合标准化器再回测。

第二问按题面使用附件1电价和附件2历史，不使用附件3的未来预报；第三问才使用附件3。`day_ahead_features(problem=4)`会删除第二/三问固定日电价列，避免将其误当作第四问已知的未来实际电价。第四问的电价提前可知性需另行明确。

描述性统计、全年IQR标记、全年日汇总使用了完整实际数据，只适用于检查和事后分析，不可作为早期预测输入。附件1曲线与全年同期均值很接近，也不能因此自行用全年统计拟合一个“历史可知”的预测器。

## 后续建模

第一问可读取q1表开展线性规划。以后按计划器、实际运行模拟器、费用计算器分别实现；预测特征不包含电池SOC，SOC需由模拟器跨日连续维护。

本包没有拟合预测模型、选择训练超参数、执行购电优化或填写正式结果。电池初始状态、充放电效率解释、结算方式、未用电量处理和未来电价信息边界仍需在模型中明确。
