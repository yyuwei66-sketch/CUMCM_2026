"""Independently verify saved results using the original input and scalar energy rules."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    cfg = json.loads((ROOT/'parameters.json').read_text(encoding='utf-8'))
    summary = json.loads((ROOT/'results'/'summary.json').read_text(encoding='utf-8'))
    source = pd.read_csv(ROOT/'input'/'q1_typical_day.csv',float_precision='round_trip')
    df = pd.read_csv(ROOT/'results'/'dispatch_10min.csv',float_precision='round_trip')
    results = []
    tol = cfg['physical_tolerance']
    def check(name, ok):
        if not bool(ok):
            raise AssertionError(name)
        results.append({'check':name,'passed':True})
    check('Input SHA256 matches solution',hashlib.sha256((ROOT/'input'/'q1_typical_day.csv').read_bytes()).hexdigest()==summary['input_sha256'])
    check('Configuration matches solution',cfg==summary['parameters'])
    check('Exactly 144 contiguous slots',len(df)==144 and np.array_equal(df.slot,np.arange(1,145)))
    expected_labels = lambda values: [f'{int(v)//60:02d}:{int(v)%60:02d}' for v in values]
    check('Physical interval labels preserved',df.interval_start.tolist()==expected_labels(source.interval_start_minute) and df.interval_end.tolist()==expected_labels(source.interval_end_minute))
    load = source.load_kw.to_numpy()/6
    pv = source.pv_forecast_kw.to_numpy()/6
    price = source.price_yuan_per_kwh.to_numpy()
    for col, expected in [('load_kwh',load),('pv_kwh',pv),('price_yuan_per_kwh',price)]:
        check('Source values: '+col,np.allclose(df[col],expected,atol=1e-10,rtol=0))
    check('Nonnegative decisions',(df[['grid_kwh','charge_kwh','discharge_kwh','curtailment_kwh']].to_numpy()>=-tol).all())
    check('Supply-demand balance',np.max(abs(df.grid_kwh+pv+df.discharge_kwh-load-df.charge_kwh-df.curtailment_kwh))<tol)
    storage = float(cfg['initial_storage_kwh'])
    ec, ed = cfg['charge_efficiency'],cfg['discharge_efficiency']
    reconstructed = [storage]
    for row in df.itertuples():
        if abs(row.storage_start_kwh-storage)>tol:
            raise AssertionError('Stored energy continuity')
        storage += ec*row.charge_kwh-row.discharge_kwh/ed
        if abs(row.storage_end_kwh-storage)>tol:
            raise AssertionError('Stored energy update')
        reconstructed.append(storage)
    check('Independent storage reconstruction',True)
    check('Storage bounds',min(reconstructed)>=cfg['storage_min_kwh']-tol and max(reconstructed)<=cfg['storage_max_kwh']+tol)
    check('Terminal stored energy',abs(storage-cfg['terminal_storage_kwh'])<tol)
    check('Charge power limit',df.charge_kwh.max()*6<=cfg['max_charge_power_kw']+tol)
    check('Discharge power limit',df.discharge_kwh.max()*6<=cfg['max_discharge_power_kw']+tol)
    check('No simultaneous charge and discharge',not ((df.charge_kwh>tol)&(df.discharge_kwh>tol)).any())
    check('Curtailment limited to available PV',(df.curtailment_kwh<=pv+tol).all())
    cost = float(price@df.grid_kwh.to_numpy())
    base = float(price@np.maximum(load-pv,0))
    check('Total cost independently recomputed',abs(cost-summary['cost_yuan'])<tol)
    check('Grid purchase independently recomputed',abs(df.grid_kwh.sum()-summary['grid_kwh'])<tol)
    check('Baseline recomputed',abs(base-summary['baseline_cost_yuan'])<tol)
    check('Cost no worse than no-storage baseline',cost<=base+tol)
    # This is a certificate consistency check; the solver establishes the bound.
    check('Reported LP lower bound does not exceed attained cost',summary['lp_lower_bound_yuan']<=cost+tol)
    check('Reported per-slot costs match source price',np.allclose(df.purchase_cost_yuan,price*df.grid_kwh,atol=tol,rtol=0))
    table1 = pd.read_csv(ROOT/'results'/'paper_table1.csv',float_precision='round_trip')
    expected = df.iloc[[h*6 for h in [10,12,14,16,18,20]]]
    check('Paper table1 matches selected physical intervals',np.allclose(table1.grid_kwh,expected.grid_kwh,atol=tol,rtol=0))
    table2 = pd.read_csv(ROOT/'results'/'paper_table2.csv',float_precision='round_trip')
    check('Four-hour charging totals',np.allclose(table2.charge_kwh,df.charge_kwh.to_numpy().reshape(6,24).sum(axis=1),atol=tol,rtol=0))
    check('Four-hour discharging totals',np.allclose(table2.discharge_kwh,df.discharge_kwh.to_numpy().reshape(6,24).sum(axis=1),atol=tol,rtol=0))
    state_table = pd.read_csv(ROOT/'results'/'storage_states.csv',float_precision='round_trip')
    check('145 saved boundary states',len(state_table)==145 and np.allclose(state_table.stored_energy_kwh,reconstructed,atol=tol,rtol=0))
    path = ROOT/'results'/'independent_checks.json'
    path.write_text(json.dumps({'status':'PASS','checks':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Independent checks: PASS ({len(results)} checks)')
    print(f'Cost: {cost:.4f} yuan')
    print(f'Ending stored energy: {storage:.4f} kWh')


if __name__ == '__main__':
    main()
