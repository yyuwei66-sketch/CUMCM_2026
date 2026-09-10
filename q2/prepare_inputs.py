"""Read-only extraction from original attachments 1 and 2."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import openpyxl

ROOT=Path(__file__).resolve().parent


def prepare(raw=None,out=None):
    raw=ROOT/'raw' if raw is None else Path(raw)
    out=ROOT/'input' if out is None else Path(out);out.mkdir(parents=True,exist_ok=True)
    w=openpyxl.load_workbook(raw/'附件2.xlsx',read_only=True,data_only=True)
    blocks=[]; axes=[]
    for name in ['小区负载','光伏发电实际功率']:
        rows=list(w[name].values)
        dates=pd.DatetimeIndex([r[0] for r in rows[1:]])
        values=np.array([r[1:] for r in rows[1:]],dtype=float)
        assert values.shape==(365,144),(name,values.shape)
        assert dates.equals(pd.date_range('2025-01-01','2025-12-31'))
        assert np.isfinite(values).all() and (values>=0).all()
        blocks.append(values);axes.append([str(v) for v in rows[0][1:]])
    assert axes[0]==axes[1]
    w.close()
    w=openpyxl.load_workbook(raw/'附件1.xlsx',read_only=True,data_only=True)
    rows=list(w.active.values);w.close()
    assert len(rows)==145
    # Time header normalization handles text + Excel time values consistently.
    def minute(v):
        if hasattr(v,'hour'):return v.hour*60+v.minute if v.hour or v.minute else 1440
        text=str(v).replace('：',':')
        if '+1' in text:return 1440
        h,m=map(int,text.split(':')[:2]);return 60*h+m if h or m else 1440
    offsets=[minute(r[0]) for r in rows[1:]]
    assert offsets==list(range(10,1441,10))
    # Attachment 2 uses matching endpoint headers; do not silently shift columns.
    assert [minute(v) for v in axes[0]]==offsets
    prices=np.array([r[1] for r in rows[1:]],float)
    assert np.isfinite(prices).all() and (prices>0).all()
    actual=pd.DataFrame({'date':np.repeat(dates.strftime('%Y-%m-%d'),144),
        'slot':np.tile(np.arange(1,145),365),'load_kw':blocks[0].ravel(),'pv_kw':blocks[1].ravel()})
    actual.to_csv(out/'q2_actual_load_pv.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'slot':np.arange(1,145),'price_yuan_per_kwh':prices}).to_csv(out/'q2_fixed_prices.csv',index=False,encoding='utf-8-sig')
    metadata={'source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw/'附件1.xlsx',raw/'附件2.xlsx']},
       'days':365,'slots_per_day':144,'observations':52560,
       'input_columns':['date','slot','load_kw','pv_kw'],
       'prices_source':'附件1.xlsx only; no attachment 3 or 4 fields are read',
       'time_assumption':'sample at t represents average on [t-10min,t); provisional',
       'source_numeric_missing':0,'numeric_imputation':False,'negative_net_load_preserved':True}
    (out/'input_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


if __name__=='__main__':print(json.dumps(prepare(),ensure_ascii=False,indent=2))
