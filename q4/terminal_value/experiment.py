"""Reproducible stage runner; all computational artifacts are under .tmp."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.tmp/q4_terminal_value/matplotlib'))
import argparse
import hashlib
import json
import pickle
import time
import numpy as np
import pandas as pd
from core import (TMP, CAP, E0, EMIN, XGRID, Q, Reward, ValueModel, backtest,
                  build_labels, historical_future, load_data, save_pickle,
                  signature, summary, teacher)

OUT=Path(__file__).resolve().parent
FEATURES=('economic','pca','hybrid')
KINDS=('linear','quadratic')


def log(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:
        x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def run_key(data):
    return hashlib.sha256((data['key']+signature([Path(__file__)])).encode()).hexdigest()


def indices(data,start,end):
    return np.flatnonzero((data['dates']>=pd.Timestamp(start))&(data['dates']<=pd.Timestamp(end)))


def cached_run(data,idx,rewards,name,slots=False):
    folder=TMP/'runs'; folder.mkdir(exist_ok=True)
    path=folder/f'{name}.pkl'
    reward_spec=[(r.kind,r.a,r.b,None if r.values is None else r.values.tolist()) for r in rewards]
    key=hashlib.sha256((run_key(data)+repr(list(idx))+repr(reward_spec)).encode()).hexdigest()
    if path.exists():
        cached=pickle.loads(path.read_bytes())
        if cached['key']==key and (not slots or cached['slots'] is not None):
            return cached['daily'],cached['slots']
    daily,details=backtest(data,idx,rewards,slots)
    save_pickle(path,dict(key=key,daily=daily,slots=details))
    daily.to_csv(folder/f'{name}_daily.csv',index=False)
    if details is not None:
        details.to_csv(folder/f'{name}_slots.csv.gz',index=False,compression='gzip')
    return daily,details


def configs():
    for kind in KINDS:
        for feature in FEATURES:
            for k in ([0] if feature=='economic' else [2,3,5]):
                for alpha in [.1,1.,10.,100.]:
                    yield dict(kind=kind,feature=feature,components=k,alpha=alpha)


def cid(cfg):
    return f"{cfg['kind']}_{cfg['feature']}_k{cfg['components']}_a{cfg['alpha']:g}"


def validation(data,labels):
    path=TMP/'selection.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('key')==run_key(data):
            log('读取已完成的开发期选参结果'); return saved
    rows=[]
    for no,cfg in enumerate(configs(),1):
        monthly=[]
        for month in range(5,9):
            start=pd.Timestamp(2025,month,1); end=start+pd.offsets.MonthEnd(0)
            idx=indices(data,start,end)
            train=[labels[d] for d in sorted(labels) if d<idx[0]]
            model=ValueModel(**cfg).fit(train)
            rewards=model.predict([labels[d] for d in idx])
            daily,_=cached_run(data,idx,rewards,f'validation_{cid(cfg)}_m{month}')
            monthly.append(summary(daily))
        row=cfg|dict(config_id=cid(cfg),cash_cost=sum(x['cash_cost'] for x in monthly),
            inventory_adjusted_045=sum(x['inventory_adjusted_045'] for x in monthly),
            inventory_adjusted_090=sum(x['inventory_adjusted_090'] for x in monthly),
            fallbacks=sum(x['fallbacks'] for x in monthly),
            solve_seconds=sum(x['solve_seconds'] for x in monthly),
            monthly_end_storage=[x['end_storage'] for x in monthly])
        rows.append(row)
        pd.DataFrame(rows).to_csv(TMP/'validation_grid.csv',index=False)
        log(f"选参 {no}/56：{cid(cfg)}，开发期现金费用 {row['cash_cost']:,.2f} 元")
    table=pd.DataFrame(rows)
    selected={}
    for kind in KINDS:
        for feature in FEATURES:
            subset=table[(table.kind==kind)&(table.feature==feature)].sort_values(['cash_cost','components','alpha'])
            best=subset.iloc[0]
            selected[f'{kind}_{feature}']={k:best[k] for k in ('kind','feature','components','alpha')}
            selected[f'{kind}_{feature}']['components']=int(best.components)
    best_quad=table[table.kind=='quadratic'].sort_values('cash_cost').iloc[0]
    selection=dict(key=run_key(data),selected=selected,
                   primary=f'{best_quad.kind}_{best_quad.feature}',
                   selection_metric='May-Aug cash cost; inventory sensitivity reported separately',
                   validation_months=[5,6,7,8],frozen_after='2025-08-31')
    write_json(path,selection)
    return selection


def rewards_for(name,labels,idx,model=None):
    if name=='fixed': return [Reward('fixed') for _ in idx]
    if name=='direct_quadratic':
        return [Reward('quadratic',labels[d]['a'],labels[d]['b']) for d in idx]
    if name=='direct_pwl': return [Reward('pwl',values=labels[d]['values']) for d in idx]
    return model.predict([labels[d] for d in idx])


def evaluate(data,labels,selection,explore=False):
    phase='exploratory' if explore else 'frozen'
    start='2025-02-01' if explore else '2025-09-01'
    idx=indices(data,start,'2025-12-31')
    names=['fixed',*selection['selected'],'direct_quadratic','direct_pwl']
    tables=[]; diagnostics=[]
    for name in names:
        model=None
        value_fit_seconds=0.; value_predict_seconds=0.
        if name in selection['selected']:
            cfg=selection['selected'][name]
            if explore:
                # Monthly expanding refits; hyperparameters selected in development => exploratory.
                rewards=[]
                for month in range(2,13):
                    month_idx=[d for d in idx if data['dates'][d].month==month]
                    train=[labels[d] for d in sorted(labels) if d<month_idx[0]]
                    if len(train)<21:
                        rewards.extend([Reward('fixed') for _ in month_idx])
                    else:
                        t=time.perf_counter()
                        model=ValueModel(**cfg).fit(train)
                        value_fit_seconds+=time.perf_counter()-t
                        t=time.perf_counter()
                        rewards.extend(model.predict([labels[d] for d in month_idx]))
                        value_predict_seconds+=time.perf_counter()-t
            else:
                train=[labels[d] for d in sorted(labels) if d<idx[0]]
                t=time.perf_counter()
                model=ValueModel(**cfg).fit(train)
                value_fit_seconds=time.perf_counter()-t
                frozen_before=pickle.dumps(model,protocol=5)
                t=time.perf_counter()
                rewards=rewards_for(name,labels,idx,model)
                value_predict_seconds=time.perf_counter()-t
                assert pickle.dumps(model,protocol=5)==frozen_before
                save_pickle(TMP/f'{name}_frozen_model.pkl',model)
        else:
            rewards=rewards_for(name,labels,idx)
        prediction_rows=[]
        for d,r in zip(idx,rewards):
            if r.kind!='fixed':
                err=np.array([r.value(x) for x in XGRID])-labels[d]['values']
                prediction_rows.append(dict(date=labels[d]['date'],value_rmse=float(np.sqrt(np.mean(err**2))),
                    low_error=r.a-labels[d]['marginal_low'] if r.kind!='pwl' else 0,
                    high_error=r.a-r.b*CAP-labels[d]['marginal_high'] if r.kind!='pwl' else 0))
        daily,slots=cached_run(data,idx,rewards,f'{phase}_{name}',slots=True)
        # Fee reconstruction and cross-day continuity are checked from serialized-level details.
        assert np.allclose(slots.planned_cost,slots.actual_price*slots.planned_grid_kwh)
        assert np.allclose(slots.emergency_cost,5*slots.actual_price*slots.emergency_kwh)
        assert np.allclose(slots.storage_start.to_numpy()[1:],slots.storage_end.to_numpy()[:-1],atol=.003)
        assert np.isclose(slots.planned_cost.sum()+slots.emergency_cost.sum(),daily.total_cost.sum())
        row=dict(strategy=name,phase=phase,**summary(daily))
        row.update(value_fit_seconds=value_fit_seconds,value_predict_seconds=value_predict_seconds,
                   teacher_seconds=sum(labels[d]['seconds'] for d in idx) if name.startswith('direct') else 0.)
        if prediction_rows:
            pr=pd.DataFrame(prediction_rows)
            row['mean_value_rmse_yuan']=float(pr.value_rmse.mean())
            pr.to_csv(TMP/f'{phase}_{name}_value_errors.csv',index=False)
        tables.append(row)
        log(f"{phase} {name}：{row['cash_cost']:,.2f} 元，期末 {row['end_storage']:.1f} kWh")
    table=pd.DataFrame(tables)
    baseline=float(table.loc[table.strategy=='fixed','cash_cost'].iloc[0])
    table['saving_yuan']=baseline-table.cash_cost
    table['saving_percent']=100*table.saving_yuan/baseline
    table.to_csv(TMP/f'{phase}_summary.csv',index=False)
    return table


def diagnostic_days(data,labels):
    candidates=[d for d in sorted(labels) if 5<=data['dates'][d].month<=8]
    spreads=np.array([np.ptp(labels[d]['price']) for d in candidates])
    deficits=np.array([np.maximum(labels[d]['net'],0).sum() for d in candidates])
    surpluses=np.array([np.maximum(-labels[d]['net'],0).sum() for d in candidates])
    picks=[candidates[int(np.argmax(spreads))],candidates[int(np.argmax(deficits))],
           candidates[int(np.argmax(surpluses))],candidates[int(np.argsort(spreads)[len(spreads)//2])]]
    return list(dict.fromkeys(picks))


def diagnostics(data,labels):
    path=TMP/'diagnostics.pkl'
    if path.exists():
        cached=pickle.loads(path.read_bytes())
        if cached['key']==run_key(data): return cached
    rows=[]; curves={}
    for d in diagnostic_days(data,labels):
        f=historical_future(data,d); ref=labels[d]
        curves[labels[d]['date']]={}
        for terminal,hourly in [(6000,True),(6000,False),(4800,True),(7200,True)]:
            result=ref if (terminal,hourly)==(6000,True) else teacher(f,terminal,hourly)
            name=f"{'hourly' if hourly else '10min'}_terminal{terminal}"
            curves[labels[d]['date']][name]=result['values']
            rows.append(dict(date=labels[d]['date'],variant=name,
                max_value_change_yuan=float(np.max(np.abs(result['values']-ref['values']))),
                value_fit_rmse_yuan=result['fit_rmse_yuan'],a=result['a'],b=result['b'],
                low_marginal=result['marginal_low'],high_marginal=result['marginal_high'],
                linear_fit_rmse=float(np.sqrt(np.mean((result['linear']*XGRID-result['values'])**2)))))
        log(f"粒度及终点敏感性完成：{labels[d]['date']}")
    pd.DataFrame(rows).to_csv(TMP/'teacher_sensitivity.csv',index=False)
    result=dict(key=run_key(data),rows=rows,curves=curves)
    save_pickle(path,result)
    return result


def markdown_table(frame):
    headers=[str(c) for c in frame.columns]
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in frame.itertuples(index=False,name=None):
        cells=[(f'{x:.3e}' if 1e-9<abs(x)<.001 else f'{x:,.2f}')
               if isinstance(x,(float,np.floating)) else str(x) for x in row]
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def report(data,labels,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':150})
    figdir=TMP/'figures'; figdir.mkdir(exist_ok=True)
    frozen=pd.read_csv(TMP/'frozen_summary.csv')
    exploratory=pd.read_csv(TMP/'exploratory_summary.csv')
    sens=pd.read_csv(TMP/'teacher_sensitivity.csv')
    diag=pickle.loads((TMP/'diagnostics.pkl').read_bytes())
    validation_table=pd.read_csv(TMP/'validation_grid.csv')
    label_df=pd.read_csv(TMP/'labels.csv')
    names={'fixed':'Fixed penalty','linear_economic':'Linear / Economic','linear_pca':'Linear / PCA',
           'linear_hybrid':'Linear / Hybrid','quadratic_economic':'Quadratic / Economic',
           'quadratic_pca':'Quadratic / PCA','quadratic_hybrid':'Quadratic / Hybrid',
           'direct_quadratic':'Direct quadratic','direct_pwl':'Direct piecewise'}
    fig,ax=plt.subplots(figsize=(10,5))
    colors=['#34495e' if n=='fixed' else '#277c8e' if n.startswith('linear') else '#d08037'
            if n.startswith('quadratic') else '#855b99' for n in frozen.strategy]
    ax.barh([names[n] for n in frozen.strategy],frozen.saving_yuan/1000,color=colors)
    ax.axvline(0,color='black',lw=.7); ax.invert_yaxis()
    ax.set_xlabel('Cash saving vs fixed penalty (thousand yuan)');ax.set_title('Frozen evaluation: September–December')
    fig.tight_layout();fig.savefig(figdir/'01_frozen_savings.png');plt.close(fig)
    count=len(diag['curves']); fig,axs=plt.subplots(1,count,figsize=(4.3*count,3.7),squeeze=False)
    for ax,(day,variants) in zip(axs[0],diag['curves'].items()):
        d=next(d for d in labels if labels[d]['date']==day); row=labels[d]
        xx=np.linspace(0,CAP,100)
        ax.plot(XGRID/1000,row['values'], 'o',label='Teacher (hourly)',color='#34495e')
        ax.plot(xx/1000,row['a']*xx-.5*row['b']*xx**2,label='Quadratic',color='#d08037')
        ax.plot(xx/1000,row['linear']*xx,label='Linear',color='#277c8e',ls='--')
        ax.set_title(day);ax.set_xlabel('Usable end energy (MWh)');ax.set_ylabel('Future cost saving (yuan)')
    axs[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'02_value_curves.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for n in ['fixed',selection['primary'],'direct_quadratic','direct_pwl']:
        daily=pd.read_csv(TMP/f'runs/frozen_{n}_daily.csv')
        ax.plot(pd.to_datetime(daily.date),daily.end_storage,label=names[n],lw=1,alpha=.8)
    ax.set_ylabel('Actual end-of-day storage (kWh)');ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'03_actual_end_storage.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(pd.to_datetime(label_df.date),label_df.marginal_low,label='Marginal value at minimum SOC')
    ax.plot(pd.to_datetime(label_df.date),label_df.marginal_high,label='Marginal value at maximum SOC')
    ax.set_ylabel('Yuan per internal kWh');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'04_marginal_labels.png');plt.close(fig)
    primary=frozen.set_index('strategy').loc[selection['primary']]
    best=frozen.sort_values('cash_cost').iloc[0]
    tab=frozen[['strategy','cash_cost','saving_yuan','saving_percent','emergency_kwh','end_storage']].rename(columns={
        'strategy':'策略','cash_cost':'实际费用（元）','saving_yuan':'较固定惩罚节费（元）',
        'saving_percent':'节费率（%）','emergency_kwh':'紧急购电（kWh）','end_storage':'最终储电（kWh）'})
    inv=frozen[['strategy','cash_cost','inventory_adjusted_045','inventory_adjusted_090']].rename(columns={
        'strategy':'策略','cash_cost':'不调整库存','inventory_adjusted_045':'按0.45估值','inventory_adjusted_090':'按0.9估值'})
    err=frozen[['strategy','mean_value_rmse_yuan','mean_abs_terminal_gap','boundary_fraction','solve_seconds','fallbacks']].fillna(0)
    selected_table=pd.DataFrame([dict(strategy=k,**v) for k,v in selection['selected'].items()])
    timing=frozen[['strategy','value_fit_seconds','value_predict_seconds','teacher_seconds','solve_seconds']]
    base=frozen.set_index('strategy').loc['fixed']
    gains=[float(base[c]-primary[c]) for c in ('cash_cost','inventory_adjusted_045','inventory_adjusted_090')]
    stable=all(x>0 for x in gains) or all(x<0 for x in gains) or all(abs(x)<.01 for x in gains)
    verdict='降低' if primary.saving_yuan>0 else '增加'
    body=fr'''# Q4-2：预测驱动的期末储能价值实验报告

## 结论

开发期预先选中的二次主模型为 **{selection['primary']}**。9—12月冻结评价中，实际费用为 **{primary.cash_cost:,.2f} 元**，相对统一执行口径的固定惩罚方案{verdict} **{abs(primary.saving_yuan):,.2f} 元**，节费率 **{primary.saving_percent:.4f}%**。按0、0.45、0.9元/内部kWh处理期末库存时，改进方向{'一致' if stable else '不一致，结论依赖库存估值'}。

冻结期费用最低的观察结果是 **{best.strategy}**；该排名是事后描述，不据此重新选择主模型。二次奖励或 PCA 不保证优于简单方案，以本表为准。

{markdown_table(tab)}

![冻结评价节费]({figdir/'01_frozen_savings.png'})

## 模型输入、标签与输出

当天净负荷采用原 Q2 的因果 PCA 岭回归预测和历史残差场景，固定取0.85分位数。价格取前一周同日同时刻。未来六天的负荷、光伏、价格均取前一周同日曲线；它们在决策日零点全部已知。数据中的10分钟记录沿用原 Q2 区间口径，每小时聚合连续6个时段。

令 $$x=E_{{末}}-1200$$，有效范围为0—9600 kWh。未来老师从次日开始规划六天，小时粒度，末端回到6000 kWh，中间不固定每日末端。七个初始可用电量对应未来费用 $$J_d(x)$$，标签为 $$R_d(x)=J_d(0)-J_d(x)$$。老师只优化确定性预测的正常购电费，没有未来紧急购电风险项，也不预知实际未来。

线性奖励为 $$\lambda x$$；主模型奖励为 $$ax-bx^2/2$$。约束 $$b\geq0$$、$$a-b\times9600\geq0$$。学习模型输入经济特征、价格与净负荷 PCA 分数或两者组合，输出两端边际价值并投影到非负递减约束，再换算 $$a,b$$。PCA 分别对144维未来小时价格与净负荷曲线中心化降维；回归前再标准化全部特征。经济特征为33维，保留相对日期顺序。

每条标签记录信息截止日期、六个参考日期和配置；包含完整元数据的标签缓存在 `.tmp/q4_terminal_value/labels.pkl`。标签从1月8日开始，共{len(labels)}天。年末未来日期允许进入2026年，但所有输入仍从2025年已经发生的参考日构造，不读取2026年实测数据。

直接二次对照每天现场算七点价值后拟合二次函数；直接分段对照使用七点线性插值。两者均使用相同预测，不是完美预知下界。前者帮助识别学习误差，后者帮助识别二次近似误差。分段插值本身也有离散近似误差。

## 训练、选参与评价口径

1—4月积累标签，5—8月按月扩展训练，每月统一6000 kWh初始电量，按四个月现金费用之和选超参数。比较56组配置；每种奖励与特征组合分别保留一组，二次主模型也只在开发期选择。

{markdown_table(selected_table)}

8月31日冻结价值模型、PCA和标准化参数，9月1日各策略重新从6000 kWh开始，连续运行到12月31日。底层 Q2 预测按固定算法使用新历史，不属于重新训练价值回归。9—12月在此前项目中曾被查看，因此这里只称“本轮参数冻结后的评价”。0.85分位数沿用现有开发结果，不为各策略分别优化。

全年探索使用已选设置按月重训，训练只包含当月之前标签，至少21个历史标签才启用学习，否则回退固定惩罚。2月1日有24个历史标签。该结果使用了开发期选出的超参数，不能作为独立全年样本外证据。

## 实际执行与库存

所有方案固定零点购电量，实际富余先充电、缺口先放电，剩余缺口才按真实价格5倍紧急购电。实际电量跨日连续。该规则不同于原 Q4-2 固定充放电执行，因此不将原报告费用作为可直接比较的基线。

奖励作用于计划末电量；实际末电量可能偏离，甚至削弱奖励效果。该实验没有价值感知的实时执行。储能范围1200—10800 kWh，功率5000 kW，两端效率均0.9。费用只包含真实价格下的计划费用与紧急购电费用，不扣奖励、不加优化软惩罚。

库存调整为现金费用减去统一估值乘以“期末减期初电量”；只是会计敏感性，不是允许向外网售电。有限预测视野的老师末端约束不能消除评价期库存差异。

{markdown_table(inv)}

![实际末电量]({figdir/'03_actual_end_storage.png'})

## 价值近似与执行差异

下表依次给出策略、七点价值预测RMSE的日均值（元）、计划与实际末电量绝对差的日均值（kWh）、实际储能边界触及比例、日计划求解耗时（秒）、二次求解回退次数。固定惩罚行的价值误差填0仅表示不适用，不代表完美拟合。日计划耗时来自首次计算并随缓存保存。

{markdown_table(err)}

冻结期估值与调度的分项耗时如下（秒）。学习方案包含一次冻结拟合、批量评价特征变换与预测；直接方案的老师耗时为相应日期首次生成七点标签的累计时间。所有学习方案还共同依赖历史老师标签，全年358天标签生成总耗时约{sum(r['seconds'] for r in labels.values()):.2f}秒，不能把离线训练标签成本忽略。当前问题规模小，直接估值本身已很快，学习的必要性不能只凭“省计算”论证。

{markdown_table(timing)}

![价值曲线拟合]({figdir/'02_value_curves.png'})

![边际价值标签]({figdir/'04_marginal_labels.png'})

## 老师粒度与末端边界敏感性

开发期按最大价差、最大净缺电、最大富余光伏和中位价差选代表日，重复日期去重。对比小时/10分钟老师以及4800、6000、7200 kWh的未来末端电量。价值变化相对“小时、6000”基准；这项检查不是完整策略敏感性回测。

{markdown_table(sens)}

代表日中，更改六天末端参考电量对初始储能价值的影响接近数值精度；小时与10分钟粒度则存在可见差异。这说明这些样本的远期边界影响较弱，但不能据此证明六天规划优于更短视野，也不能把小时估值视为精确值。

## 全年探索性结果

{markdown_table(exploratory[['strategy','cash_cost','saving_yuan','saving_percent','emergency_kwh','end_storage']])}

## 验证与局限

针对未来数据修改不影响预测、标签及当天计划，效率口径，二次内点解与线性规划一维回退一致性，二次退化为线性，分段线性对照，冻结变换不拟合评价数据，以及实际执行能量平衡，提供独立回归测试。

所有正式回测检查逐时供需、储能递推、容量、功率、互斥、跨日连续和费用复算。冻结评价最大实际能量平衡误差为 {frozen.max_balance_error.max():.3g} kWh，最大计划能量平衡误差为 {frozen.max_plan_balance_error.max():.3g} kWh。允许数值误差0.003 kWh；求解内部以MWh缩放。二次求解失败会执行同一目标的线性规划加一维凸搜索，记录原因，不静默替换模型。

主要限制：只有一年数据；未来供需使用周复制；老师没有随机风险和未来滚动信息价值；小时聚合可能损失短时机会；实际执行仍然贪心；七点与二次拟合均为近似；0.45库存估值不是唯一经济价值。模型拟合系数不是客观唯一的“真实价格”。

## 复现

运行环境为 `/Users/jinyu/miniconda3/envs/mcm/bin/python`，隔离 Clarabel/OSQP 位于 `.tmp/q4_terminal_value/deps`。二次规划优先使用内点法 Clarabel，OSQP 为第二选择，仍失败时进入同目标的一维凸搜索。在项目根目录运行：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/jinyu/miniconda3/envs/mcm/bin/python q4/terminal_value/experiment.py --stage all
PYTHONDONTWRITEBYTECODE=1 /Users/jinyu/miniconda3/envs/mcm/bin/python -m unittest discover -s q4/terminal_value -p 'test_*.py' -v
```

阶段可选 `labels`、`diagnostics`、`validate`、`evaluate`、`explore`、`report`、`all`。缓存按输入数据、核心代码与实验代码指纹失效；已有完整结果再次运行时复用。Notebook提供同一接口和结果表，不另实现模型。未修改现有提交文件及Q4-3。
'''
    (OUT/'REPORT.md').write_text(body,encoding='utf-8')
    metadata=dict(data_key=data['key'],experiment_key=run_key(data),primary=selection['primary'],
        runtime=sys.executable,labels=len(labels),validation_configurations=len(validation_table),
        teacher_seconds=sum(r['seconds'] for r in labels.values()),
        frozen_fallbacks=int(frozen.fallbacks.sum()),exploratory_fallbacks=int(exploratory.fallbacks.sum()),
        constraints_passed=True,inventory_direction_stable=stable)
    write_json(TMP/'run_metadata.json',metadata)
    log(f"报告已生成：{OUT/'REPORT.md'}")


def run(stage='all'):
    TMP.mkdir(parents=True,exist_ok=True)
    log('读取输入并准备因果 Q2 预测')
    data=load_data();labels=build_labels(data,log)
    if stage=='labels': return
    if stage in ('diagnostics','all'): diagnostics(data,labels)
    if stage=='diagnostics': return
    selection=validation(data,labels)
    if stage=='validate': return
    if stage in ('evaluate','all'): evaluate(data,labels,selection)
    if stage in ('explore','all'): evaluate(data,labels,selection,True)
    if stage in ('report','all'): report(data,labels,selection)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','labels','diagnostics','validate','evaluate','explore','report'],default='all')
    run(parser.parse_args().stage)
