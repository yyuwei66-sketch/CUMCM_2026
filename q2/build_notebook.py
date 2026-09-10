"""Build a standalone notebook and execute its Python cells without Jupyter.

The captured outputs are real Python execution outputs. No Jupyter kernel
protocol is exercised in this build environment; metadata records this limit.
"""
from pathlib import Path
import ast,base64,contextlib,gzip,io,json,time,uuid
import pandas as pd

ROOT=Path(__file__).resolve().parent


def build():
    core=(ROOT/'q2_core.py').read_text();tree=ast.parse(core);lines=core.splitlines()
    functions={n.name:'\n'.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body if isinstance(n,ast.FunctionDef)}
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=s.strip()+'\n'))
    def code(s,hidden=False):cells.append(dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={'jupyter':{'source_hidden':True}} if hidden else {},source=s.strip()+'\n',execution_count=None,outputs=[]))
    def funcs(*names):code('\n\n\n'.join(functions[n] for n in names))
    md('''# 第二问：日前购电与实时储能执行

本 Notebook 可独立运行，内嵌从附件 1、2 重建的数据，无需第一问目录。

**先看第 2 节参数，再从上到下运行。** 比较七日均值基准、28 日历史情景模型和无储能对照，保留 1 月预热及 2—12 月的完整结果。

当前按“时刻 t 的功率代表前 10 分钟平均功率”计算，原模板时间列存在错位，提交前需确认。功率乘 1/6 转为 kWh；充电、放电效率各取 90%。

这是一份可复现的第二问完整工作版本。日前模型优化的是历史情景近似目标，实时执行规则为启发式，不宣称未知未来下的全局最优。''')
    md('''## 1. 环境

在 VS Code 中选择 `cumcm2026` 内核。需要 numpy、pandas、scipy、matplotlib、ipykernel。完整项目的原始 Excel 重提取额外需要 openpyxl。

当前文件已保存真实运行输出。制作环境按顺序执行全部 Python 单元，未测试 Jupyter 内核协议。''')
    code('''from pathlib import Path
import json, io, gzip, base64, time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix
try:
    from IPython.display import display, Image
except ImportError:
    if 'display' not in globals():
        display = print
    if 'Image' not in globals():
        Image = lambda filename: Path(filename)
pd.set_option('display.max_columns', 16)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
print('Python numerical stack:', np.__version__, pd.__version__, scipy.__version__)''')
    md('''## 2. 参数与关键假设

- 1 月 1 日初始储电量为 6000 kWh，当天无历史数据：电池待机，光伏供电，不足应急补购。
- 1 月 2 日起使用已结束的历史天运行策略；1 月属于预热，不进入题面要求的 2—12 月费用汇总。
- 每天计划的末端目标为 6000 kWh。这是防止单日模型耗尽储能的策略假设，**不是第二题强制的日循环约束**。
- 实际末端状态不强制等于 6000，直接成为下一天的初始状态。
- 每个 10 分钟内允许根据当期负载和光伏完成平衡，忽略监测、控制延迟。
- 历史窗口取 28 日，基准取 7 日，均事先固定，不根据 2—12 月回测费用选择。

窗口或电池参数修改后需重新运行后续单元；策略代号中的 7/28 是默认值，实际值以 cfg 为准。''')
    from q2_core import DEFAULT
    code('DEFAULT = '+repr(DEFAULT)+'\ncfg = DEFAULT.copy()\nREPORT_DATES = '+repr(['2025-03-20','2025-06-21','2025-09-23','2025-12-21'])+"\nMETHODS = ['mean7_battery', 'saa28_battery', 'saa28_no_battery']\nPRIMARY = 'saa28_battery'\nOUT = Path.cwd() / 'q2_notebook_results'\nOUT.mkdir(exist_ok=True)\ndisplay(pd.DataFrame(list(cfg.items()),columns=['参数','数值']))")
    md('''## 3. 输入数据与完整性

内嵌表只包含日期、时段、负载和光伏。电价单独来自附件 1。**第二问不读取附件 3、4，也不使用此前特征表中的历史实际电价列。**

内嵌数据展开后应有 365×144=52,560 行。保留原始数值，不填补、不削峰、不标准化；负净负载代表光伏富余，予以保留。''')
    actual=base64.b64encode(gzip.compress((ROOT/'input/q2_actual_load_pv.csv').read_bytes(),mtime=0)).decode()
    price=base64.b64encode(gzip.compress((ROOT/'input/q2_fixed_prices.csv').read_bytes(),mtime=0)).decode()
    code("ACTUAL_B64 = '"+actual+"'\nPRICE_B64 = '"+price+"'\nactual = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(ACTUAL_B64))), float_precision='round_trip')\nprice_data = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(PRICE_B64))), float_precision='round_trip')",True)
    code('''assert list(actual.columns) == ['date', 'slot', 'load_kw', 'pv_kw']
assert len(actual) == 52560 and not actual.duplicated(['date','slot']).any()
actual = actual.sort_values(['date','slot']).reset_index(drop=True)
dates = pd.DatetimeIndex(sorted(pd.to_datetime(actual.date.unique())))
assert dates.equals(pd.date_range('2025-01-01','2025-12-31'))
assert np.array_equal(actual.slot, np.tile(np.arange(1,145),365))
load = actual.load_kw.to_numpy().reshape(365,144)
pv = actual.pv_kw.to_numpy().reshape(365,144)
prices = price_data.price_yuan_per_kwh.to_numpy()
assert np.isfinite(load).all() and np.isfinite(pv).all() and (prices > 0).all()
print('天数 / 每天时段数:', load.shape, '；评价时段数:', 334*144)
display(actual.head())''')
    md(r'''## 4. 只用过去构造情景

第 d 天的第 s 个历史情景是此前最多 28 天中的一整天：

$$N^{(s)}_{d,t}=\frac{L_{d-s,t}-P_{d-s,t}}6.$$

不截断负值。七日均值基准先对过去最多 7 天同一时段取平均，作为唯一确定性情景。索引切片 `first:day_index` 明确排除当天及之后的数据。''')
    funcs('history_scenarios')
    code('''feb_index = int(np.flatnonzero(dates == pd.Timestamp('2025-02-01'))[0])
scenarios, first = history_scenarios(load, pv, feb_index, PRIMARY, cfg)
print('2 月 1 日情景形状:', scenarios.shape)
print('可见历史:', dates[first].date(), '至', dates[feb_index-1].date())
display(pd.DataFrame(scenarios[:3,:6], columns=[f'前{i+1}个时段' for i in range(6)]))''')
    md(r'''## 5. 日前随机规划近似模型

变量：计划普通购电 $g_t$、计划充放电 $c_t,d_t$、计划电量 $E_t$，以及固定该充放电计划时每个历史情景的应急量 $h_{s,t}$。

$$\min\sum_t p_tg_t+\frac1S\sum_s\sum_t5p_t h_{s,t}$$

$$g_t+h_{s,t}+d_t-c_t\ge N_{s,t},\qquad h_{s,t},g_t\ge0$$

$$E_{t+1}=E_t+0.9c_t-d_t/0.9,\qquad1200\le E_t\le10800$$

$$0\le c_t,d_t\le5000/6,\qquad c_td_t=0$$

初态取当天实测电量，计划末态目标取 6000。富余供给允许未使用。

先求 LP 松弛，检查充放电互斥；若冲突则用二进制模式变量求 MILP。若 LP 解满足互斥，它就是该日前情景模型的最优解。**情景中的应急量是优化辅助变量，不能当作当天已经发生的应急量。**

无储能对照逐时求解 $g+5\mathbb E[(N-g)^+]$，采用经验 80% 分位数。带储能模型通过时间耦合约束统一求解，不能直接逐时套分位数。''')
    funcs('plan_day')
    md(r'''## 6. 实时执行与实际费用

普通购电 $g_t$ 全天固定。当前净余量为 $r_t=g_t+P_t-L_t$（均为 kWh）。

- $r_t\ge0$：在功率和容量上限内充电，其余记作未利用供电。
- $r_t<0$：在功率和最低电量限制内放电，其余缺口应急补购。

实际电量逐时更新，因此次日初态来自本日末态。这个控制器只读当前时段，不读取未来负载、光伏；计划储能轨迹用于日前购电优化，实际充放电根据当前余缺调整。

$$F_{\rm actual}=\sum_t p_tg_t+\sum_t5p_t h^{\rm actual}_t.$$

计划量没有使用也收费。未利用供电可能同时包含光伏和已购电，不能全部称为弃光。''')
    funcs('execute_day','run_strategy','daily_summary','verify_physics')
    md('''## 7. 先运行到 2 月 1 日，检查衔接

这里从 1 月 1 日开始真实递推，展示 2 月 1 日初态，不能直接把 6000 当作 2 月 1 日初态。''')
    code('''demo, demo_log = run_strategy(load[:32], pv[:32], prices, dates[:32], PRIMARY, cfg, progress=False)
display(daily_summary(demo).tail(2))
display(verify_physics(demo, cfg))''')
    md('''## 8. 完整逐日回测

三个策略均从 1 月 1 日预热，仅用 2—12 月比较。默认参数在制作环境约几十秒完成，耗时依电脑而异。''')
    code('''started = time.perf_counter()
results = {name: run_strategy(load, pv, prices, dates, name, cfg) for name in METHODS}
print(f'运行耗时：{time.perf_counter()-started:.1f} 秒')''')
    funcs('compare_strategies')
    code('''comparison = compare_strategies(results)
display(comparison)
by_name = comparison.set_index('strategy')
savings = 100*(1-by_name.loc[PRIMARY,'total_cost_yuan']/by_name.loc['mean7_battery','total_cost_yuan'])
print(f'主策略相对七日均值基准费用下降：{savings:.4f}%')''')
    md('''## 9. 物理检查与信息时序检查

除了全时段供需平衡、容量和功率限制，还修改决策日及之后的数据，确认日前情景及购电计划不变。完整项目的 `check_q2.py` 另含 68 项独立核验。''')
    code('''physical = pd.concat([verify_physics(value[0],cfg).assign(strategy=name) for name,value in results.items()],ignore_index=True)
assert physical.passed.all()
k = 150
before, _ = history_scenarios(load,pv,k,PRIMARY,cfg)
modified_load, modified_pv = load.copy(), pv.copy()
modified_load[k:] *= 11
modified_pv[k:] = 0
after, _ = history_scenarios(modified_load,modified_pv,k,PRIMARY,cfg)
assert np.array_equal(before,after)
assert np.array_equal(plan_day(before,prices,4321,cfg)['grid'], plan_day(after,prices,4321,cfg)['grid'])
print(f'{len(physical)} 项物理核验通过；未来数据扰动不改变日前计划。')
display(physical.groupby('strategy').passed.agg(['count','all']))''')
    md('''## 10. 论文所需表格

表 1 的 6 个时段按真实区间起点选择；表 2 是实际执行的 4 小时累计充放电量；表 3 把同一天连续应急的 10 分钟时段合并。无应急日期在日报中显示 0。''')
    funcs('clock','emergency_events','make_tables')
    code('''tables = make_tables(results[PRIMARY][0])
display(tables['paper_daily_totals'])
display(tables['paper_table1'])
display(tables['paper_table2'])
display(tables['paper_table3'])''')
    md('''## 11. 结果图

累计费用、月度应急电量、四个指定日期的购电与电池轨迹，均由上面的实际回测结果生成。''')
    funcs('make_figures')
    code('''make_figures(results, OUT/'figures')
for filename in ['01_cumulative_cost.png','02_monthly_emergency.png','03_selected_days.png']:
    display(Image(filename=str(OUT/'figures'/filename)))''')
    md('''## 12. 导出

输出目录为当前工作目录下的 `q2_notebook_results`。包含 10 分钟明细、日前计划日志、日报、4 小时储能表、合并应急表、检查结果和 Excel 数据矩阵。

完整包中的 Excel 和 Markdown 报告对应已交付的默认参数计算。**修改参数重新运行后，CSV/JSON/PNG 会更新，原 Excel 和报告不会自动更新**。Excel 构建脚本 `export_workbook.mjs` 根据 JSON 矩阵导出，需具备其所列 Node 依赖的环境。

原模板首列为 `0:10-0:20`，内部计算首列为 `00:00-00:10`。本次 Excel 按明确物理区间列标题输出，并附说明；正式提交前须确认竞赛方时间口径。''')
    funcs('export_results')
    code('''tables, comparison = export_results(results, OUT, cfg)
physical.to_csv(OUT/'physical_checks.csv',index=False,encoding='utf-8-sig')
print('导出目录：', OUT)
print('2—12 月天数：', len(tables['daily_summary']))
print('4 小时储能记录数：', len(tables['storage_4h']))
print('合并应急事件数：', len(tables['emergency_events']))
print('模板时间映射：暂定，详见说明。')''')
    md('''## 13. 下一步改进方向

先结合 `Q2_Report.md` 把本版本的方法、假设和结果写入论文。若继续改进，可在仅使用当时可见数据的条件下做工作日/周末情景筛选、滚动窗口验证，以及考虑未来时段价值的储能控制。

当前方法仍会多买并浪费部分电量。28 日经验分布不能保证覆盖季节转折和极端天气；实时贪心控制也未优化跨时段储能价值。这些是后续改进点，不能用全年真实数据调好参数后再称为独立测试。''')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python (cumcm2026)',language='python',name='python3'),language_info=dict(name='python',version='3.12'),execution_validation={'method':'sequential Python exec with captured real outputs','jupyter_kernel_protocol_tested':False}),cells=cells)
    dest=ROOT/'Q2_microgrid.ipynb'
    dest.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return dest


def execute(path):
    import os,traceback
    path=Path(path);nb=json.loads(path.read_text());current=[]
    class LocalImage:
        def __init__(self,filename):self.filename=filename
    def display(obj):
        if isinstance(obj,LocalImage):data={'image/png':base64.b64encode(Path(obj.filename).read_bytes()).decode(),'text/plain':str(obj.filename)}
        elif isinstance(obj,(pd.DataFrame,pd.Series)):
            frame=obj.to_frame() if isinstance(obj,pd.Series) else obj
            data={'text/html':frame.to_html(),'text/plain':frame.to_string()}
        else:data={'text/plain':str(obj)}
        current.append(dict(output_type='display_data',data=data,metadata={}))
    namespace={'__name__':'__main__','display':display,'Image':LocalImage}
    count=0;os.chdir(ROOT)
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        count+=1;cell['execution_count']=count;current.clear();stream=io.StringIO();start=time.perf_counter()
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            exec(compile(cell['source'],f'Q2_microgrid.ipynb cell {count}','exec'),namespace)
        cell['outputs']=([dict(output_type='stream',name='stdout',text=stream.getvalue())] if stream.getvalue() else [])+list(current)
        print(f'Executed code cell {count}, {time.perf_counter()-start:.2f}s',flush=True)
    nb['metadata']['execution_validation']['code_cells_executed']=count
    nb['metadata']['execution_validation']['all_succeeded']=True
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    reference=pd.read_csv(ROOT/'results/strategy_comparison.csv')
    computed=pd.read_csv(ROOT/'q2_notebook_results/strategy_comparison.csv')
    pd.testing.assert_frame_equal(reference,computed,check_exact=True)
    print('Notebook and script results match exactly.')


if __name__=='__main__':execute(build())
