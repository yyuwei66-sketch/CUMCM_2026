"""Q2: past-only scenario planning and causal battery execution.

Energies use the grid/AC side, kWh per ten-minute interval.
The planner is optimal for its historical-scenario surrogate, not for the
unknown stochastic system or the subsequent greedy execution policy.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix

DEFAULT = dict(interval_minutes=10, charge_efficiency=.9, discharge_efficiency=.9,
               storage_min_kwh=1200., storage_max_kwh=10800., initial_storage_kwh=6000.,
               planned_terminal_kwh=6000., max_charge_power_kw=5000.,
               max_discharge_power_kw=5000., emergency_multiplier=5.,
               scenario_days=28, mean_days=7, tolerance=1e-6)
REPORT_DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
METHODS = ['mean7_battery', 'saa28_battery', 'saa28_no_battery']
PRIMARY = 'saa28_battery'


def history_scenarios(load, pv, day_index, method, cfg):
    """Only complete days STRICTLY before the planning day are visible."""
    if day_index < 1:
        raise ValueError('Jan 1 has no historical data; use the explicit cold-start rule.')
    window = cfg['mean_days'] if method == 'mean7_battery' else cfg['scenario_days']
    first = max(0, day_index-window)
    scenario = (load[first:day_index]-pv[first:day_index])*cfg['interval_minutes']/60
    if method == 'mean7_battery':
        scenario = scenario.mean(axis=0, keepdims=True)
    return scenario, first


def plan_day(scenarios, prices, initial_kwh, cfg, battery=True):
    """min p*g + 5*p*mean(h); g+h_s+d-c >= net_s.

    All scenarios share the day-ahead battery schedule. h_s is a hypothetical
    emergency amount with that fixed battery schedule. Surplus may be unused.
    The actual controller below is a separately specified causal recourse rule.
    """
    a, p = np.asarray(scenarios, float), np.asarray(prices, float)
    if a.ndim != 2 or a.shape[1] != len(p) or not np.isfinite(a).all():
        raise ValueError('Invalid scenarios')
    if not (np.isfinite(p).all() and (p > 0).all()):
        raise ValueError('Prices must be positive')
    s, n = a.shape
    if not battery:
        # Any empirical 80%-quantile minimises g+5*mean((net-g)+).
        rank = max(0, int(np.ceil((1-1/cfg['emergency_multiplier'])*s))-1)
        g = np.maximum(np.sort(a, axis=0)[rank], 0)
        h = np.maximum(a-g, 0)
        return dict(grid=g, charge=np.zeros(n), discharge=np.zeros(n),
                    storage=np.full(n+1, initial_kwh), expected_emergency=h.mean(axis=0),
                    objective=float(p@g+cfg['emergency_multiplier']*(h@p).mean()),
                    solver='empirical quantile', plan_residual=0.)
    # x = [g(n), c(n), d(n), E(n+1), h(s*n)]
    h0, m = 4*n+1, 4*n+1+s*n
    obj = np.zeros(m); obj[:n] = p
    obj[h0:] = np.tile(cfg['emergency_multiplier']*p/s, s)
    ec, ed = cfg['charge_efficiency'], cfg['discharge_efficiency']
    mc = cfg['max_charge_power_kw']*cfg['interval_minutes']/60
    md = cfg['max_discharge_power_kw']*cfg['interval_minutes']/60
    lo, hi = cfg['storage_min_kwh'], cfg['storage_max_kwh']
    lower = np.zeros(m); upper = np.full(m, np.inf)
    upper[n:2*n], upper[2*n:3*n] = mc, md
    lower[3*n:4*n+1], upper[3*n:4*n+1] = lo, hi
    lower[3*n] = upper[3*n] = initial_kwh
    lower[4*n] = upper[4*n] = cfg['planned_terminal_kwh']
    t = np.arange(n)
    eq = coo_matrix((np.r_[-ec*np.ones(n), np.ones(n)/ed, -np.ones(n), np.ones(n)],
                     (np.tile(t,4), np.r_[n+t,2*n+t,3*n+t,3*n+t+1])), shape=(n,m)).tocsr()
    rows = np.arange(s*n); slots = np.tile(t,s)
    ub = coo_matrix((np.r_[-np.ones(s*n),np.ones(s*n),-np.ones(s*n),-np.ones(s*n)],
                     (np.tile(rows,4),np.r_[slots,n+slots,2*n+slots,h0+rows])),
                    shape=(s*n,m)).tocsr()
    b = -a.ravel()
    lp = linprog(obj, A_ub=ub, b_ub=b, A_eq=eq, b_eq=np.zeros(n),
                 bounds=np.column_stack([lower,upper]), method='highs')
    if not lp.success:
        raise RuntimeError(lp.message)
    x, solver = lp.x, 'LP'
    if ((x[n:2*n]>cfg['tolerance']) & (x[2*n:3*n]>cfg['tolerance'])).any():
        # A binary fallback prevents artificial simultaneous charge/discharge.
        mode = coo_matrix((np.r_[np.ones(n),-mc*np.ones(n),np.ones(n),md*np.ones(n)],
                    (np.r_[t,t,n+t,n+t],np.r_[n+t,m+t,2*n+t,m+t])),shape=(2*n,m+n)).tocsr()
        res = milp(np.r_[obj,np.zeros(n)], integrality=np.r_[np.zeros(m),np.ones(n)],
                   bounds=Bounds(np.r_[lower,np.zeros(n)],np.r_[upper,np.ones(n)]),
                   constraints=[LinearConstraint(hstack([eq,csr_matrix((n,n))]),0,0),
                     LinearConstraint(hstack([ub,csr_matrix((s*n,n))]),-np.inf,b),
                     LinearConstraint(mode,-np.inf,np.r_[np.zeros(n),md*np.ones(n)])],
                   options={'mip_rel_gap':1e-8})
        if not res.success:
            raise RuntimeError(res.message)
        x, solver = res.x[:m], 'MILP'
    if np.max(ub@x-b)>cfg['tolerance'] or np.max(abs(eq@x))>cfg['tolerance']:
        raise AssertionError('Planning constraints failed')
    return dict(grid=np.maximum(x[:n],0),charge=np.maximum(x[n:2*n],0),
                discharge=np.maximum(x[2*n:3*n],0),storage=x[3*n:4*n+1],
                expected_emergency=x[h0:].reshape(s,n).mean(axis=0),objective=float(obj@x),
                solver=solver,plan_residual=float(max(np.max(ub@x-b),np.max(abs(eq@x)),0)))


def execute_day(grid, actual_load, actual_pv, initial_kwh, cfg, battery=True):
    """Online greedy rule; iteration t reads only actual interval t and E_t.

    Consume paid-for grid and PV; store surplus within limits; discharge for a
    deficit; buy any remaining shortage as emergency energy. No future lookup.
    Real-time balancing assumes observation/actuation inside the 10-min slot.
    """
    n = len(grid); e = np.empty(n+1); e[0] = initial_kwh
    c, d, h, w = [np.zeros(n) for _ in range(4)]
    ec, ed = cfg['charge_efficiency'], cfg['discharge_efficiency']
    mc = cfg['max_charge_power_kw']*cfg['interval_minutes']/60 if battery else 0
    md = cfg['max_discharge_power_kw']*cfg['interval_minutes']/60 if battery else 0
    for t in range(n):
        r = grid[t]+actual_pv[t]-actual_load[t]
        if r >= 0:
            c[t] = min(r,mc,max(0,(cfg['storage_max_kwh']-e[t])/ec))
            w[t] = r-c[t]
        else:
            d[t] = min(-r,md,max(0,(e[t]-cfg['storage_min_kwh'])*ed))
            h[t] = -r-d[t]
        e[t+1] = e[t]+ec*c[t]-d[t]/ed
    return dict(charge=c,discharge=d,emergency=h,unused=w,storage=e)


def run_strategy(load, pv, prices, dates, method, cfg=None, progress=True):
    cfg = dict(DEFAULT if cfg is None else cfg)
    if method not in METHODS: raise ValueError(method)
    if load.shape != pv.shape or load.shape[1] != 144 or len(prices)!=144:
        raise ValueError('Expect day x 144 inputs')
    state = cfg['initial_storage_kwh']; frames=[]; diary=[]
    battery = method != 'saa28_no_battery'
    for k, date in enumerate(dates):
        if k == 0:
            # Jan 1: no prior observation. No scheduled purchase, idle battery,
            # direct PV use, emergency fills deficit. January is warm-up only.
            plan = dict(grid=np.zeros(144),charge=np.zeros(144),discharge=np.zeros(144),
                        storage=np.full(145,state), expected_emergency=np.zeros(144),
                        objective=0.,solver='cold-start',plan_residual=0.)
            first = 0
        else:
            scenarios,first = history_scenarios(load,pv,k,method,cfg)
            plan = plan_day(scenarios,prices,state,cfg,battery)
        l,v = load[k]*cfg['interval_minutes']/60,pv[k]*cfg['interval_minutes']/60
        actual = execute_day(plan['grid'],l,v,state,cfg,battery and k>0)
        e = actual['storage']; state = float(e[-1])  # Never reset actual SOC.
        intervals = pd.date_range(date, periods=144, freq='10min')
        frame = pd.DataFrame(dict(date=str(pd.Timestamp(date).date()),slot=np.arange(1,145),
             interval_start=intervals,interval_end=intervals+pd.Timedelta(minutes=10),
             price_yuan_per_kwh=prices,load_kwh=l,pv_kwh=v,
             planned_grid_kwh=plan['grid'],planned_charge_kwh=plan['charge'],
             planned_discharge_kwh=plan['discharge'],planned_storage_start_kwh=plan['storage'][:-1],
             planned_storage_end_kwh=plan['storage'][1:],
             forecast_emergency_kwh=plan['expected_emergency'],charge_kwh=actual['charge'],
             discharge_kwh=actual['discharge'],emergency_kwh=actual['emergency'],
             unused_energy_kwh=actual['unused'],storage_start_kwh=e[:-1],storage_end_kwh=e[1:]))
        # Unused supply may contain bought energy. Do not label all of it curtailment.
        frame['planned_cost_yuan'] = prices*plan['grid']
        frame['emergency_cost_yuan'] = cfg['emergency_multiplier']*prices*actual['emergency']
        frame['total_cost_yuan'] = frame.planned_cost_yuan+frame.emergency_cost_yuan
        frame['evaluation'] = pd.Timestamp(date) >= pd.Timestamp('2025-02-01')
        frames.append(frame)
        diary.append(dict(date=str(pd.Timestamp(date).date()),solver=plan['solver'],
           history_first_date=None if k==0 else str(pd.Timestamp(dates[first]).date()),
           history_last_date=None if k==0 else str(pd.Timestamp(dates[k-1]).date()),
           history_days=0 if k==0 else k-first,objective_surrogate_yuan=plan['objective'],
           planning_constraint_residual=plan['plan_residual'],initial_actual_kwh=float(e[0]),
           terminal_actual_kwh=float(e[-1])))
        if progress and ((k+1)%60==0 or k==len(dates)-1):
            print(f'{method}: {k+1}/{len(dates)} days',flush=True)
    return pd.concat(frames,ignore_index=True),pd.DataFrame(diary)


def daily_summary(dispatch):
    sumcols=['planned_grid_kwh','emergency_kwh','planned_cost_yuan','emergency_cost_yuan',
             'total_cost_yuan','charge_kwh','discharge_kwh','unused_energy_kwh','load_kwh','pv_kwh']
    out=dispatch.groupby('date')[sumcols].sum()
    out['initial_storage_kwh']=dispatch.groupby('date').storage_start_kwh.first()
    out['terminal_storage_kwh']=dispatch.groupby('date').storage_end_kwh.last()
    out['emergency_slots']=dispatch.assign(event=dispatch.emergency_kwh>1e-6).groupby('date').event.sum()
    return out.reset_index()


def verify_physics(dispatch,cfg=None):
    cfg=DEFAULT if cfg is None else cfg; x=dispatch; tol=cfg['tolerance']
    def maximum(a):return float(np.max(np.abs(a)))
    residual=x.planned_grid_kwh+x.pv_kwh+x.discharge_kwh+x.emergency_kwh-x.load_kwh-x.charge_kwh-x.unused_energy_kwh
    evolution=x.storage_end_kwh-x.storage_start_kwh-cfg['charge_efficiency']*x.charge_kwh+x.discharge_kwh/cfg['discharge_efficiency']
    numerics={
      'energy_balance_kwh':maximum(residual), 'battery_update_kwh':maximum(evolution),
      'continuous_state_kwh':maximum(x.storage_start_kwh.to_numpy()[1:]-x.storage_end_kwh.to_numpy()[:-1]),
      'soc_lower_violation_kwh':max(0,float(cfg['storage_min_kwh']-x.storage_end_kwh.min())),
      'soc_upper_violation_kwh':max(0,float(x.storage_end_kwh.max()-cfg['storage_max_kwh'])),
      'charge_power_violation_kw':max(0,float(x.charge_kwh.max()*60/cfg['interval_minutes']-cfg['max_charge_power_kw'])),
      'discharge_power_violation_kw':max(0,float(x.discharge_kwh.max()*60/cfg['interval_minutes']-cfg['max_discharge_power_kw'])),
      'simultaneous_charge_discharge_slots':int(((x.charge_kwh>tol)&(x.discharge_kwh>tol)).sum()),
      'planned_cost_error_yuan':maximum(x.planned_cost_yuan-x.price_yuan_per_kwh*x.planned_grid_kwh),
      'emergency_cost_error_yuan':maximum(x.emergency_cost_yuan-cfg['emergency_multiplier']*x.price_yuan_per_kwh*x.emergency_kwh),
      'total_cost_error_yuan':maximum(x.total_cost_yuan-x.planned_cost_yuan-x.emergency_cost_yuan),
      'negative_flow_violation_kwh':max(0,-float(x[['planned_grid_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_energy_kwh']].min().min()))}
    checks=[dict(check=k,value=v,tolerance=tol,passed=v<=tol) for k,v in numerics.items()]
    checks += [dict(check='finite_numeric_values',value=None,tolerance=None,passed=bool(np.isfinite(x.select_dtypes('number')).all().all())),
               dict(check='continuous_time_axis',value=None,tolerance=None,passed=bool((pd.to_datetime(x.interval_start).iloc[1:].to_numpy()==pd.to_datetime(x.interval_end).iloc[:-1].to_numpy()).all()))]
    result=pd.DataFrame(checks)
    if not result.passed.all():raise AssertionError(result[~result.passed].to_string())
    return result


def clock(minute):return f'{minute//60:02d}:{minute%60:02d}'


def emergency_events(dispatch):
    rows=[]
    for day,x in dispatch.groupby('date',sort=True):
        x=x.sort_values('slot'); a=x.emergency_kwh.to_numpy(); t=0
        while t<len(a):
            if a[t]<=1e-6: t+=1;continue
            start=t
            while t<len(a) and a[t]>1e-6:t+=1
            rows.append(dict(date=day,interval=f'{clock(start*10)}-{clock(t*10)}',
                             emergency_kwh=float(a[start:t].sum()),start_slot=start+1,end_slot=t))
    return pd.DataFrame(rows,columns=['date','interval','emergency_kwh','start_slot','end_slot'])


def make_tables(dispatch):
    x=dispatch[dispatch.evaluation].copy(); d=daily_summary(x)
    grid=x.pivot(index='date',columns='slot',values='planned_grid_kwh')
    grid.columns=[f'{clock((i-1)*10)}-{clock(i*10)}' for i in grid.columns]
    grid['全天计划购电量_kWh']=d.set_index('date').planned_grid_kwh
    grid['全天计划购电费_元']=d.set_index('date').planned_cost_yuan
    storage=[]
    for day,g in x.groupby('date',sort=True):
        for h in range(0,24,4):
            z=g[(g.slot>h*6)&(g.slot<=(h+4)*6)]
            storage.append(dict(date=day,interval=f'{clock(h*60)}-{clock((h+4)*60)}',
                 charge_kwh=float(z.charge_kwh.sum()),discharge_kwh=float(z.discharge_kwh.sum()),
                 storage_start_kwh=float(z.storage_start_kwh.iloc[0]),storage_end_kwh=float(z.storage_end_kwh.iloc[-1])))
    storage=pd.DataFrame(storage)
    wanted=x[x.date.isin(REPORT_DATES)&x.slot.isin([61,73,85,97,109,121])].copy()
    wanted['interval']=[f'{clock((i-1)*10)}-{clock(i*10)}' for i in wanted.slot]
    return dict(planned_purchase_wide=grid.reset_index(),storage_4h=storage,
          emergency_events=emergency_events(x),daily_summary=d,
          paper_table1=wanted[['date','interval','planned_grid_kwh']],
          paper_table2=storage[storage.date.isin(REPORT_DATES)].reset_index(drop=True),
          paper_table3=emergency_events(x[x.date.isin(REPORT_DATES)]),
          paper_daily_totals=d[d.date.isin(REPORT_DATES)].reset_index(drop=True))


def compare_strategies(results):
    rows=[]
    for name,(allx,_) in results.items():
        x=allx[allx.evaluation]; d=daily_summary(x)
        rows.append(dict(strategy=name,days=len(d),slots=len(x),
             **{k:float(x[k].sum()) for k in ['planned_grid_kwh','emergency_kwh','planned_cost_yuan','emergency_cost_yuan','total_cost_yuan','unused_energy_kwh']},
             emergency_days=int((d.emergency_slots>0).sum()),initial_storage_kwh=float(x.storage_start_kwh.iloc[0]),
             terminal_storage_kwh=float(x.storage_end_kwh.iloc[-1])))
    return pd.DataFrame(rows)


def make_figures(results,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels={'mean7_battery':'7-day mean + battery','saa28_battery':'28-day SAA + battery','saa28_no_battery':'28-day SAA, no battery'}
    colors={'mean7_battery':'#87969F','saa28_battery':'#245C82','saa28_no_battery':'#C88437'}
    fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
    for name,(allx,_) in results.items():
        d=daily_summary(allx[allx.evaluation]); ax.plot(pd.to_datetime(d.date),d.total_cost_yuan.cumsum()/1e6,label=labels[name],color=colors[name])
    ax.set(ylabel='Cumulative cost (million yuan)',title='Q2 | Realized cost, February–December 2025')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    fig.savefig(directory/'01_cumulative_cost.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
    for j,(name,(allx,_)) in enumerate(results.items()):
        x=allx[allx.evaluation].copy();x['month']=pd.to_datetime(x.date).dt.month
        m=x.groupby('month').emergency_kwh.sum()
        ax.bar(np.arange(11)+(j-1)*.25,m.to_numpy()/1000,width=.24,label=labels[name],color=colors[name])
    ax.set(xticks=np.arange(11),xticklabels=list(range(2,13)),xlabel='Month',ylabel='Emergency energy (MWh)',title='Q2 | Monthly emergency purchases')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    fig.savefig(directory/'02_monthly_emergency.png',dpi=160);plt.close(fig)
    x=results[PRIMARY][0]
    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    for date,axesrow in zip(REPORT_DATES,axes):
        g=x[x.date==date];a,b=axesrow;t=np.arange(144)/6
        a.step(t,(g.load_kwh-g.pv_kwh)*6,where='post',label='Actual net load',color='#777777',lw=1)
        a.step(t,g.planned_grid_kwh*6,where='post',label='Planned grid',color='#245C82',lw=1)
        a.fill_between(t,g.emergency_kwh*6,step='post',label='Emergency',color='#C88437',alpha=.7)
        a.set(title=date,ylabel='Power (kW)',xlim=(0,24));a.axhline(0,color='#999999',lw=.5)
        b.plot(np.arange(145)/6,np.r_[g.storage_start_kwh.iloc[0],g.storage_end_kwh],label='Actual storage',color='#245C82')
        b.plot(np.arange(145)/6,np.r_[g.planned_storage_start_kwh.iloc[0],g.planned_storage_end_kwh],label='Planned storage',color='#999999',linestyle='--')
        b.axhline(1200,color='#BBBBBB',lw=.8);b.axhline(10800,color='#BBBBBB',lw=.8)
        b.set(title=date,ylabel='Stored energy (kWh)',xlim=(0,24),ylim=(0,12000))
        for ax in axesrow:ax.set_xticks(range(0,25,4));ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8);axes[0,1].legend(frameon=False,fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Hour of day')
    fig.savefig(directory/'03_selected_days.png',dpi=150);plt.close(fig)


def export_results(results,out,cfg=None):
    cfg=DEFAULT if cfg is None else cfg;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tables=make_tables(results[PRIMARY][0]);comparison=compare_strategies(results)
    comparison.to_csv(out/'strategy_comparison.csv',index=False,encoding='utf-8-sig')
    checks=[]
    for name,(x,diary) in results.items():
        q=verify_physics(x,cfg);q.insert(0,'strategy',name);checks.append(q)
        directory=out/name;directory.mkdir(exist_ok=True)
        x.to_csv(directory/'dispatch_10min_all_year.csv',index=False,encoding='utf-8-sig')
        diary.to_csv(directory/'planning_log.csv',index=False,encoding='utf-8-sig')
        daily_summary(x).to_csv(directory/'daily_summary_all_year.csv',index=False,encoding='utf-8-sig')
    pd.concat(checks,ignore_index=True).to_csv(out/'physical_checks.csv',index=False,encoding='utf-8-sig')
    for name,frame in tables.items():frame.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    payload={k:json.loads(v.to_json(orient='split',index=False)) for k,v in tables.items()}
    payload['comparison']=json.loads(comparison.to_json(orient='split',index=False))
    # Full precision matrices for spreadsheet output; DataFrame.to_json defaults to 10 decimal places.
    for name,frame in list(tables.items())+[('comparison',comparison)]:
        payload[name]={'columns':list(frame.columns),'data':frame.astype(object).where(pd.notna(frame),None).values.tolist()}
    (out/'workbook_data.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    summary=dict(primary_strategy=PRIMARY,parameters=cfg,
      evaluation_start='2025-02-01',evaluation_end='2025-12-31',evaluation_days=334,
      time_mapping='PROVISIONAL: endpoint observation interpreted as preceding ten-minute average',
      optimality='Historical-scenario planning surrogate only; greedy real-time control is heuristic.',
      checks_passed=bool(pd.concat(checks).passed.all()),comparison=comparison.to_dict('records'))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return tables,comparison
