"""Run: python run_q2.py. All methods include causal January warm-up."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from q2_core import DEFAULT,METHODS,run_strategy,export_results,make_figures

ROOT=Path(__file__).resolve().parent


def load_inputs(directory):
    frame=pd.read_csv(Path(directory)/'q2_actual_load_pv.csv',float_precision='round_trip')
    price=pd.read_csv(Path(directory)/'q2_fixed_prices.csv',float_precision='round_trip')
    assert list(frame.columns)==['date','slot','load_kw','pv_kw']
    assert len(frame)==52560 and len(price)==144
    assert frame.groupby('date').size().eq(144).all() and not frame.duplicated(['date','slot']).any()
    frame=frame.sort_values(['date','slot'])
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(frame.date.unique())))
    assert dates.equals(pd.date_range('2025-01-01','2025-12-31'))
    assert np.array_equal(frame.slot,np.tile(np.arange(1,145),365))
    return frame.load_kw.to_numpy().reshape(365,144),frame.pv_kw.to_numpy().reshape(365,144),price.price_yuan_per_kwh.to_numpy(),dates


if __name__=='__main__':
    started=time.perf_counter()
    cfg=json.loads((ROOT/'parameters.json').read_text())
    load,pv,price,dates=load_inputs(ROOT/'input')
    results={name:run_strategy(load,pv,price,dates,name,cfg) for name in METHODS}
    tables,comparison=export_results(results,ROOT/'results',cfg)
    make_figures(results,ROOT/'results'/'figures')
    print(comparison.to_string(index=False))
    print(f'Elapsed: {time.perf_counter()-started:.1f} seconds')
