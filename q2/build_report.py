"""Generate the Markdown report directly from computed result tables."""
from pathlib import Path
import hashlib,json,shutil
import pandas as pd

ROOT=Path(__file__).resolve().parent


def table(frame,decimals=4):
    def f(v):
        if isinstance(v,(float,int)) and not isinstance(v,bool):return f'{v:,.{decimals}f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(f,row))+' |' for row in frame.itertuples(index=False,name=None)])


def build():
    r=ROOT/'results';out=ROOT/'report';out.mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    for f in (r/'figures').glob('*.png'):shutil.copyfile(f,out/'figures'/f.name)
    comparison=pd.read_csv(r/'strategy_comparison.csv');by=comparison.set_index('strategy')
    main=by.loc['saa28_battery'];base=by.loc['mean7_battery'];nob=by.loc['saa28_no_battery']
    daily=pd.read_csv(r/'paper_daily_totals.csv');t1=pd.read_csv(r/'paper_table1.csv');t2=pd.read_csv(r/'paper_table2.csv');t3=pd.read_csv(r/'paper_table3.csv')
    checks=json.loads((r/'independent_checks.json').read_text())
    x=pd.read_csv(r/'saa28_battery/dispatch_10min_all_year.csv');ev=x[x.evaluation]
    phys=pd.read_csv(r/'physical_checks.csv');pm=phys[phys.strategy=='saa28_battery'].set_index('check')
    saves=100*(1-main.total_cost_yuan/base.total_cost_yuan)
    names={'mean7_battery':'七日均值 + 储能','saa28_battery':'28 日历史情景 + 储能（主策略）','saa28_no_battery':'28 日历史情景，无储能'}
    comp=comparison[['strategy','planned_cost_yuan','emergency_cost_yuan','total_cost_yuan','emergency_kwh','unused_energy_kwh']].copy()
    comp.strategy=comp.strategy.map(names)
    for c in ['planned_cost_yuan','emergency_cost_yuan','total_cost_yuan']:comp[c]/=10000
    comp.columns=['策略','计划费（万元）','应急费（万元）','总费（万元）','应急电量（kWh）','未利用供电（kWh）']
    dailyshow=daily[['date','planned_grid_kwh','emergency_kwh','planned_cost_yuan','emergency_cost_yuan','total_cost_yuan']].copy()
    dailyshow.columns=['日期','计划购电（kWh）','应急购电（kWh）','计划费（元）','应急费（元）','总费（元）']
    tab1=t1.pivot(index='interval',columns='date',values='planned_grid_kwh').reset_index();tab1.columns.name=None
    tab1=tab1.rename(columns={'interval':'时段'})
    blocks=[]
    for day in daily.date:
        z=t2[t2.date==day][['interval','charge_kwh','discharge_kwh']].copy();z.columns=['时段','实际充电（kWh）','实际放电（kWh）']
        d=daily[daily.date==day].iloc[0]
        blocks.append(f'### {day}\n\n'+table(z)+f'\n\n00:00 实际储电量 **{d.initial_storage_kwh:,.4f} kWh**；24:00 实际储电量 **{d.terminal_storage_kwh:,.4f} kWh**。')
    events=t3[['date','interval','emergency_kwh']].copy()
    events=pd.concat([events,pd.DataFrame([{'date':'2025-06-21','interval':'无','emergency_kwh':0.}])]).sort_values(['date','interval'])
    events.columns=['日期','连续应急时间段','应急电量（kWh）']
    text=rf'''# 第二问报告：基于历史情景的日前购电与实时储能执行

## 1. 计算结论与适用范围

针对第二问，建立“过去数据构造情景—日前购电优化—实时储能执行—实际费用核算”的逐日决策方法。每天 00:00 只使用此前已结束日期的负载和光伏数据，电价固定取附件 1 的分时电价。完整运行覆盖 2025 年 1 月 1 日至 12 月 31 日，1 月用于初始化和预热，题面要求汇总的评价期为 **2 月 1 日至 12 月 31 日，共 334 天、48,096 个十分钟时段**。

默认参数下，主策略评价期实际总费用为 **{main.total_cost_yuan:,.4f} 元**，其中计划费用 **{main.planned_cost_yuan:,.4f} 元**、应急费用 **{main.emergency_cost_yuan:,.4f} 元**。实际应急购电量为 **{main.emergency_kwh:,.4f} kWh**。与同样采用实时储能控制的七日均值基准相比，总费用下降 **{saves:.4f}%**。

这份报告给出可复现的第二问完整工作版本。日前优化针对有限历史情景的近似目标求解，实时控制采用明确的因果规则；两者组合尚未证明为原随机控制问题的全局最优。下文结果均基于所列模型假设，不能解释为未来年度费用保证。

## 2. 数据来源与时间口径

### 2.1 第二问允许使用的数据

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| 144 个分时电价 | 附件 1 的电价列 | 每天已知且相同的普通购电价 |
| 365×144 个负载值 | 附件 2“小区负载” | 日前历史情景、当日实际仿真 |
| 365×144 个光伏值 | 附件 2“光伏发电实际功率” | 日前历史情景、当日实际仿真 |
| 储能参数 | C 题附录 1 | 状态与功率约束 |
| 输出范围和格式 | C 题问题 2、附录 2 及 result2 模板 | 334 天结果、指定日期论文表 |

本次输入直接从原始附件 1、2 重建。负载、光伏无缺失数值，未做填补、截尾、标准化或负净负载清零。保留了全部 52,560 条观测以及完整的日期、时段主键。原先预处理特征表在当前副本中只有 19,999 行，本次不依赖该表。

**附件 3 的官方光伏预报和附件 4 的实际电价不进入第二问。** 也未把附件 1 的典型负载、典型光伏当作全年已知值，未按全年均值构造预测。内嵌 Notebook 的数据表严格只含日期、时段、负载、光伏；价格来自另一张附件 1 提取表。

### 2.2 十分钟时段解释

沿用第一问的暂定口径：样本时刻 $t$ 的功率代表此前十分钟区间的平均功率。时段 $t=1,ldots,144$ 分别为 00:00—00:10，……，23:50—24:00，电量换算为

$$
L_{{d,t}}=rac{{P^L_{{d,t}}}}6,qquad V_{{d,t}}=rac{{P^{{PV}}_{{d,t}}}}6.
$$

这是一项数值积分假设，原始离散功率本身不能唯一确定区间电量。日末 24:00 样本在次日 00:00 可用，忽略数据传输延迟；实时执行也假设可在当期区间内测量、平衡供需。

**原模板的计划购电首列为 0:10—0:20，和上述内部口径存在一列错位。** 因此提供 `result2_时间口径暂定.xlsx`，明确使用 00:00—00:10 至 23:50—24:00 的物理区间，附带“口径说明”。原模板单独保留。竞赛方时间解释确认后再决定是否改标签或重算，不能直接忽略标题、按位置提交。

## 3. 初始化、信息集与跨日状态

### 3.1 可见信息

第 $d$ 天 00:00 可见的信息包括固定分时价格、已结束日期的负载和光伏、当前实际储电量。对当天及以后真实数据的读取只发生在随后逐时执行和事后评价中。

日前情景取历史切片 `[max(0,d-K):d]`，不包含当天。默认主策略 $K=28$，比较基准 $K=7$。窗口是事先固定的建模选择，未按 2—12 月回测结果寻优。保留每日日志中的历史起止日期，便于检查时序。

### 3.2 1 月预热规则

题面给定 1 月 1 日 00:00 电量为 6000 kWh，却没有给定 2024 年历史数据。本版本采用以下明确初始化：

1. 1 月 1 日电池待机，计划普通购电为零，光伏直接供负载，不足部分应急补购。该日作为无历史数据的冷启动。
2. 1 月 2 日起，使用此前已有的最多 7/28 天数据正常制定计划并实时执行。
3. 所有策略保留 1 月明细与费用，但论文的 334 天比较不计入 1 月费用。

该冷启动规则用于确定可复现的历史状态，不是对 1 月最优运营的结论。若取得更早的历史数据，可以替换冷启动；若改用其他 1 月策略，2 月初态和评价结果也应重新计算。

### 3.3 计划末态与实际末态

为限制单日有限时域模型在日末过度耗尽电池，在**日前计划模型**中设置末态目标 $E^p_{{d,144}}=6000$ kWh。第二题没有强制每天始末电量相等，这个目标是本版本额外引入的策略条件，不应写成题面要求。

实际执行时不强制日末回到该目标，且严格满足

$$
E^a_{{d+1,0}}=E^a_{{d,144}}.
$$

主策略在 2 月 1 日实际初态为 **{main.initial_storage_kwh:,.4f} kWh**，12 月 31 日实际末态为 **{main.terminal_storage_kwh:,.4f} kWh**。这些均来自连续递推，不是每天重置的初值。统计费用不计算末端库存残值。

## 4. 日前历史情景模型

### 4.1 净负载情景

净负载定义为 $N_{{d,t}}=L_{{d,t}}-V_{{d,t}}$，可为负。每天取此前最多 28 天各时段的实际净负载作为等概率情景，记作 $N^{{(s)}}_{{d,t}}$，每个情景概率为 $1/S_d$。

情景沿用历史日的完整曲线，但本版本各情景共享同一条计划储能轨迹，因此没有实现随情景分支调整储能的多阶段随机控制。历史分布只是对下一天不确定性的经验近似，不能保证极端天气或季节变化时仍准确。

### 4.2 决策变量

| 符号 | 意义 | 单位 |
| --- | --- | --- |
| $g_t$ | 00:00 确定的普通计划购电量 | kWh |
| $c^p_t,d^p_t$ | 00:00 模型中的计划充、放电量，均按交流侧计 | kWh |
| $E^p_t$ | 日前模型中的储电量 | kWh |
| $h_{{s,t}}$ | 固定该计划充放电轨迹时，情景 $s$ 的辅助应急购电量 | kWh |
| $p_t$ | 附件 1 固定分时电价 | 元/kWh |

### 4.3 目标函数

为在购电时考虑预测不足的高额代价，采用样本平均近似目标：

$$
min sum_{{t=1}}^{{144}}p_tg_t+rac1Ssum_{{s=1}}^Ssum_{{t=1}}^{{144}}5p_t h_{{s,t}}.
$$

第一项是必付的普通购电费，第二项是固定计划储能轨迹下的历史情景平均应急费。没有把未使用的普通购电量退费，也没有把辅助应急量误当成当日实际已经发生的补购。

### 4.4 约束

对每个情景和时段要求

$$
g_t+h_{{s,t}}+d^p_t-c^p_tge N^{{(s)}}_t,qquad g_t,h_{{s,t}}ge0.
$$

不等式中的富余量允许不使用，微网不向外网售电。

储能更新为

$$
E^p_t=E^p_{{t-1}}+eta_c c^p_t-rac{{d^p_t}}{{eta_d}},qquad eta_c=eta_d=0.9.
$$

容量、功率、边界条件为

$$
1200le E^p_tle10800,qquad
0le c^p_tlerac{{5000}}6,qquad
0le d^p_tlerac{{5000}}6,qquad
c^p_td^p_t=0,
$$

$$
E^p_0=E^a_{{d,0}},qquad E^p_{{144}}=6000.
$$

这里各向效率取 90%，对应往返效率 81%。若题目后续澄清“90%”指往返效率，应统一修改两问参数并重算。

### 4.5 求解与最优性边界

当 $S=28$ 时，模型含 4609 个连续变量、144 个储能等式和 4032 个情景供需不等式，另有变量界。使用 SciPy 调用 HiGHS 求解。

先去掉充放电互斥约束求 LP 松弛，再检查是否存在同一时段同时充、放电。若不存在，所得解同时可行于严格模型并达到其松弛下界，即为这个日前模型的最优解；若存在，则加入 144 个二进制充放电模式变量求 MILP。

本次主策略在 1 月 2 日至 12 月 31 日的 **364 次优化均由 LP 完成且满足互斥**，没有触发 MILP 回退。该结论只适用于上述历史情景规划模型，不能推出后续实时控制组合的全年全局最优。

## 5. 实时控制和实际结算

### 5.1 当期供需规则

当天普通计划 $g_t$ 保持不变。观察到当前实际负载、光伏后，计算交流侧净余量

$$r_t=g_t+V_t-L_t.$$

若 $r_tge0$，实际充电量为

$$
c^a_t=minleft(r_t,rac{{5000}}6,rac{{10800-E^a_{{t-1}}}}{{0.9}}ight),qquad d^a_t=h^a_t=0,qquad w_t=r_t-c^a_t.
$$

若 $r_t<0$，实际放电、应急购电量为

$$
d^a_t=minleft(-r_t,rac{{5000}}6,0.9(E^a_{{t-1}}-1200)ight),qquad
h^a_t=-r_t-d^a_t,qquad c^a_t=w_t=0.
$$

实际电量更新为

$$E^a_t=E^a_{{t-1}}+0.9c^a_t-d^a_t/0.9.$$

该规则有三个含义：优先用已购电和光伏供负载；有余量才在限制内充电；缺电先在限制内放电，其余应急补足。它只依赖当前时段与当前电量，未提前读取当天后续观测。日前的 $c^p,d^p,E^p$ 用来优化购电，实际充放电由上述规则决定，因此计划与实际储能曲线可以不同。

本版本属于实时贪心控制：没有显式衡量把电留给未来高电价时段的价值。它清楚、可复现、满足物理约束，但仍有改进空间。进一步优化实时控制时，必须保持普通购电不变且不读取未来真实数据。

### 5.2 费用核算与余电

实际费用统一计算为

$$
F_{{mathrm{{actual}}}}=sum_dsum_t p_tg_{{d,t}}+sum_dsum_t5p_t h^a_{{d,t}}.
$$

普通计划全部付费，应急购电按五倍价付费，不存在从普通费中扣除闲置电量的步骤。供需平衡逐时满足

$$g_t+V_t+d^a_t+h^a_t=L_t+c^a_t+w_t.$$

$w_t$ 称为“未利用供电”，可能包含光伏以及已经付费的购电。题目没有给出两种能源在余电中的分摊规则，所以没有将其全部记为弃光或据此计算光伏利用率。

## 6. 对照策略与评价结果

### 6.1 对照定义

七日均值基准对此前最多 7 天的同一时段净负载取平均，使用同样的电池参数、计划末态目标和实时控制。均值模型无法充分表达五倍应急电价带来的不对称损失。

无储能对照使用与主策略相同的 28 日历史样本，但电池完全不参与。逐时最小化

$$p_tg_t+5p_trac1Ssum_s(N^{{(s)}}_t-g_t)^+.$$

去掉正价格常数后，其经验最优解可选为截断到非负的 80% 分位数。这是无储能、无时段耦合的结论；主策略有储能时仍应整体求解。

三个策略的参数与角色均在回测前设定。这里报告的是同一段历史回测的实际策略比较，不是外部独立年度测试。1 月预热后的储电量和 12 月末库存列在结果文件中；无储能对照电池保持 6000，储能策略的评价边界库存可能不同，费用比较未做库存价值调整。

### 6.2 2—12 月实际费用

{table(comp)}

主策略相对七日均值基准节省 **{base.total_cost_yuan-main.total_cost_yuan:,.4f} 元（{saves:.4f}%）**；相对无储能情景对照节省 **{nob.total_cost_yuan-main.total_cost_yuan:,.4f} 元（{100*(1-main.total_cost_yuan/nob.total_cost_yuan):.4f}%）**。

主策略计划购电量为 **{main.planned_grid_kwh:,.4f} kWh**，比均值基准多买电，但应急量下降 **{100*(1-main.emergency_kwh/base.emergency_kwh):.4f}%**，在五倍应急价格下使总费明显下降。与此同时，未利用供电达到 **{main.unused_energy_kwh:,.4f} kWh**，说明保守购电有明显浪费代价。不能只报告应急下降，而忽略多买电和余电增加。

主策略共有 **{int(main.emergency_days)} 天**出现应急购电，其余日期应急为零。由于储能有功率、容量限制，降低应急风险不等于完全消除应急。

![各策略累计实际费用](figures/01_cumulative_cost.png)

![各策略月度应急购电量](figures/02_monthly_emergency.png)

## 7. 题面指定日期结果

### 7.1 全天购电量与费用

{table(dailyshow)}

“计划购电”与“应急购电”分开列示。如需外网实际总购电量，应将二者相加。表 1 中的全天计划费对应普通计划，论文同时给出应急费和总费以避免混淆。

### 7.2 表 1：六个指定区间的计划购电量

单位：kWh。为便于跨日期对照，下表将题面的六个区间逐行列出，数值对应同一物理区间。

{table(tab1)}

### 7.3 表 2：四小时实际充放电量及日边界电量

四小时内充电量、放电量均非零，代表不同十分钟时段先后充放电，不代表同时充放电。

{chr(10).join(blocks)}

### 7.4 表 3：连续应急购电时段

同一天连续非零的十分钟应急记录合并，电量求和，不跨午夜合并。

{table(events)}

6 月 21 日应急为零，但该日未利用供电为 **{daily.loc[daily.date=='2025-06-21','unused_energy_kwh'].iloc[0]:,.4f} kWh**。这说明“零应急”本身不能作为经济性最优的证据。

![指定日期的购电与实际、计划储能轨迹](figures/03_selected_days.png)

## 8. 核验

三个策略共保留 157,680 条十分钟全年运行记录。每个策略有 14 项全时段物理及结算核验，共 42 项通过；另有 **{checks['passed_count']} 项独立检查通过**，包括：

- 原始数据与输出数值、日期数量、固定价格一致性。
- 实际供需平衡、储电更新、充放电互斥、容量和功率边界。
- 每日末态与次日初态连续，计划初态与当天实际初态一致。
- 普通计划全额收费、应急五倍收费、汇总与明细一致。
- 历史样本严格早于决策日，不使用当天及之后的数据。
- 修改决策日之后的真实负载、光伏，不改变之前的情景与计划。
- 修改未来时段真实数据，不改变实时执行的既有前缀。
- 满电时多余的已购电仍未利用；最低电量时应急准确补足；充电功率正确限幅。
- 连续应急事件合并前后总电量一致。

主策略供需平衡最大残差为 **{pm.loc['energy_balance_kwh','value']:.3e} kWh**，储电更新最大残差为 **{pm.loc['battery_update_kwh','value']:.3e} kWh**，跨时段电量衔接最大残差为 **{pm.loc['continuous_state_kwh','value']:.3e} kWh**。核验容差为 $10^{{-6}}$。

Notebook 的全部代码单元已通过顺序 Python 执行并保存真实表格与图像输出，策略比较结果与脚本输出完全一致。制作环境没有 Jupyter 内核依赖，因此未测试 Jupyter 内核协议；在用户电脑中应选择自己的 Python 内核运行。

## 9. 文件、复现和下一步

| 文件 | 用途 |
| --- | --- |
| `Q2_microgrid.ipynb` | 编程主入口，独立内嵌数据、中文说明与已运行输出 |
| `q2_core.py` | 情景、日前优化、实时执行、汇总和物理检查 |
| `run_q2.py` | 脚本方式完整运行三个策略 |
| `prepare_inputs.py` | 从原始附件 1、2 重新提取输入 |
| `check_q2.py` | 独立核验 |
| `parameters.json` | 默认参数 |
| `results/` | 三个策略全年明细、334 天表格、比较、检查结果和图 |
| `q2_notebook_results/` | Notebook 对应的实际运行输出 |
| `result2_时间口径暂定.xlsx` | 主策略的完整 Excel，附时间口径与费用说明 |
| `templates/result2_original.xlsx` | 原模板，不改动 |
| `report/Q2_Report.md` | 本报告 |

在完整包目录执行 `python run_q2.py` 可重算数值结果，再执行 `python check_q2.py` 核验。Notebook 则直接选择内核后从上往下运行。修改参数后要重新生成依赖结果，已交付 Excel 和报告不会因 Notebook 重跑自动更新。Excel 构建脚本需要 `@oai/artifact-tool` Node 环境，读取 `results/workbook_data.json`；数值计算本身只依赖 Python。

论文建议以“信息可见性—历史情景随机规划—实时执行—比较与局限”的顺序叙述。当前还应明确保留四项改进方向：时间标签的官方确认；充放电效率解释的统一；只用过去数据做按星期或季节条件化情景；引入未来价格价值和不确定性信息的实时储能控制。若做参数选择，应使用时间滚动验证，不能用未来实际数据校准过去计划。
'''
    report=out/'Q2_Report.md';report.write_text(text,encoding='utf-8')
    (out/'report_metadata.json').write_text(json.dumps({'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
       'source':'results/summary.json and full-precision CSV outputs','time_mapping':'PROVISIONAL','model_status':'feasible causal policy, surrogate planner optimal'},ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':print(build())
