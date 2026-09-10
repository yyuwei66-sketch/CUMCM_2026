"""Independent reconciliation and causality tests; run after run_q2.py."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from q2_core import DEFAULT,METHODS,history_scenarios,plan_day,execute_day,emergency_events
from run_q2 import load_inputs

ROOT=Path(__file__).resolve().parent


def run_checks(output=None):
    out=ROOT/'results' if output is None else Path(output)
    cfg=json.loads((ROOT/'parameters.json').read_text());tol=cfg['tolerance']
    load,pv,p,dates=load_inputs(ROOT/'input');checks=[]
    def check(name,condition,detail=''):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
        if not condition:raise AssertionError(name+': '+str(detail))
    input_columns=list(pd.read_csv(ROOT/'input/q2_actual_load_pv.csv',nrows=0).columns)
    price_columns=list(pd.read_csv(ROOT/'input/q2_fixed_prices.csv',nrows=0).columns)
    check('raw-only allowed input schema',input_columns==['date','slot','load_kw','pv_kw'] and price_columns==['slot','price_yuan_per_kwh'],
          'Separate load/PV and attachment-1 price tables; no attachment-3/4 columns.')
    for name in METHODS:
        x=pd.read_csv(out/name/'dispatch_10min_all_year.csv',float_precision='round_trip')
        diary=pd.read_csv(out/name/'planning_log.csv')
        ev=x[x.evaluation]
        check(name+': complete full-year trace',len(x)==52560 and x.date.nunique()==365)
        check(name+': complete evaluation trace',len(ev)==48096 and ev.date.nunique()==334)
        check(name+': attachment 1 prices exact',np.array_equal(x.price_yuan_per_kwh,np.tile(p,365)))
        check(name+': attachment 2 values exact',np.allclose(x.load_kwh,load.ravel()/6,rtol=0,atol=1e-10) and np.allclose(x.pv_kwh,pv.ravel()/6,rtol=0,atol=1e-10))
        check(name+': initial state Jan 1',abs(x.storage_start_kwh.iloc[0]-6000)<tol)
        check(name+': day-boundary continuity',np.max(abs(x.groupby('date').storage_start_kwh.first().to_numpy()[1:]-x.groupby('date').storage_end_kwh.last().to_numpy()[:-1]))<tol)
        check(name+': history strictly precedes plan date',bool((pd.to_datetime(diary.history_last_date.iloc[1:])<pd.to_datetime(diary.date.iloc[1:])).all()))
        check(name+': allowed history window',(diary.history_days<= (7 if name=='mean7_battery' else 28)).all())
        check(name+': fixed purchase settlement',abs(x.planned_cost_yuan.sum()-np.dot(x.price_yuan_per_kwh,x.planned_grid_kwh))<1e-6)
        check(name+': emergency settlement',abs(x.emergency_cost_yuan.sum()-5*np.dot(x.price_yuan_per_kwh,x.emergency_kwh))<1e-6)
        c=x.charge_kwh.to_numpy();d=x.discharge_kwh.to_numpy();e0=x.storage_start_kwh.to_numpy();e1=x.storage_end_kwh.to_numpy()
        check(name+': actual energy conservation',np.max(abs(x.planned_grid_kwh+x.pv_kwh+d+x.emergency_kwh-x.load_kwh-c-x.unused_energy_kwh))<tol)
        check(name+': actual storage conservation',np.max(abs(e1-e0-.9*c+d/.9))<tol)
        check(name+': actual SOC and power limits',min(e0.min(),e1.min())>=1200-tol and max(e0.max(),e1.max())<=10800+tol and max(c.max(),d.max())<=5000/6+tol)
        check(name+': no simultaneous actual flows',not ((c>tol)&(d>tol)).any())
        check(name+': planned SOC evolution',np.max(abs(x.planned_storage_end_kwh-x.planned_storage_start_kwh-.9*x.planned_charge_kwh+x.planned_discharge_kwh/.9))<tol)
        check(name+': no simultaneous planned flows',not ((x.planned_charge_kwh>tol)&(x.planned_discharge_kwh>tol)).any())
        check(name+': planning state equals measured start',np.max(abs(x.groupby('date').planned_storage_start_kwh.first()-x.groupby('date').storage_start_kwh.first()))<tol)
        check(name+': planned terminal target',np.max(abs(x.groupby('date').planned_storage_end_kwh.last()-6000))<tol)
        events=emergency_events(ev)
        check(name+': merged emergency events conserve energy',abs(events.emergency_kwh.sum()-ev.emergency_kwh.sum())<1e-6)
        check(name+': planned solver constraints',(diary.planning_constraint_residual<=tol).all())
    # Metamorphic leakage test: future data changes cannot alter an earlier plan.
    origin=150;s,_=history_scenarios(load,pv,origin,'saa28_battery',cfg)
    changed_load=load.copy();changed_pv=pv.copy();changed_load[origin:]*=11;changed_pv[origin:]=0
    s2,_=history_scenarios(changed_load,changed_pv,origin,'saa28_battery',cfg)
    check('future actual mutation cannot change historical scenarios',np.array_equal(s,s2))
    a=plan_day(s,p,4321,cfg);b=plan_day(s2,p,4321,cfg)
    check('future actual mutation cannot change day-ahead purchases',np.array_equal(a['grid'],b['grid']))
    # With no battery, validate analytic empirical-quantile objective by a distinct bounded minimizer.
    toy=np.array([[10.],[20.],[30.],[40.],[100.]])
    q=plan_day(toy,np.array([2.]),6000,cfg,False)
    f=lambda g:2*g+10*np.maximum(toy[:,0]-g,0).mean()
    independent=minimize_scalar(f,bounds=(0,110),method='bounded')
    check('quantile plan matches independent scalar cost minimum',abs(q['objective']-independent.fun)<1e-4)
    # Future-slot mutation leaves the whole execution prefix unchanged.
    original=execute_day(np.full(144,600.),load[100]/6,pv[100]/6,6000,cfg)
    future=load[100]/6;future=future.copy();future[72:]*=10
    mutated=execute_day(np.full(144,600.),future,pv[100]/6,6000,cfg)
    check('execution has no future-slot lookup',all(np.array_equal(original[k][:72],mutated[k][:72]) for k in ['charge','discharge','emergency','unused','storage']))
    # Full battery: bought surplus is paid and discarded, not retroactively refunded.
    z=execute_day(np.array([100.]),np.array([10.]),np.array([0.]),10800,cfg)
    check('paid-for surplus discarded at full SOC',z['unused'][0]==90 and z['charge'][0]==0 and z['emergency'][0]==0)
    # Empty operating SOC: all deficits require emergency supply.
    z=execute_day(np.array([0.]),np.array([100.]),np.array([0.]),1200,cfg)
    check('SOC floor triggers exact emergency coverage',z['emergency'][0]==100 and z['discharge'][0]==0 and z['storage'][-1]==1200)
    # Extremely high surplus cannot exceed charge power.
    z=execute_day(np.array([2000.]),np.array([0.]),np.array([0.]),1200,cfg)
    check('charge-power saturation',abs(z['charge'][0]-5000/6)<tol and z['unused'][0]>0)
    data={'status':'PASS','checks':checks,'passed_count':len(checks)}
    (out/'independent_checks.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(checks)} independent checks: PASS')
    return data


if __name__=='__main__':run_checks()
