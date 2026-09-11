from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog, Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FROZEN_FORECAST = {
    "window_days": 60,
    "k_load": 3,
    "k_pv": 2,
    "alpha_load": 0.3,
    "alpha_pv": 10.0,
    "scenario_days": 28,
}
FROZEN_Q_STAR = 0.85
Q_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
DEV_START, DEV_END = "2025-05-01", "2025-08-31"
EVAL_START, EVAL_END = "2025-02-01", "2025-12-31"

CFG = {
    "eta_c": 0.9,
    "eta_d": 0.9,
    "storage_min_kwh": 1200.0,
    "storage_max_kwh": 10800.0,
    "initial_storage_kwh": 6000.0,
    "terminal_storage_kwh": 6000.0,
    "max_charge_kw": 5000.0,
    "max_discharge_kw": 5000.0,
    "emergency_multiplier": 5.0,
    "interval_hours": 1/6,
}

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)

def load_inputs(actual_path, price_path):
    x = pd.read_csv(actual_path, encoding="utf-8-sig")
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["date","slot"]).reset_index(drop=True)
    require(len(x)==365*144, "actual data must contain 365×144 rows")
    require(x[["load_kw","pv_kw"]].notna().all().all(), "actual data has missing values")
    require((x[["load_kw","pv_kw"]] >= 0).all().all(), "actual data has negative load/PV")
    dates = pd.DatetimeIndex(x["date"].drop_duplicates())
    load = x["load_kw"].to_numpy(float).reshape(-1,144)
    pv = x["pv_kw"].to_numpy(float).reshape(-1,144)

    p = pd.read_csv(price_path, encoding="utf-8-sig").sort_values("slot")
    require(len(p)==144, "price file must contain 144 rows")
    prices = p["price_yuan_per_kwh"].to_numpy(float)
    require(np.isfinite(prices).all() and (prices>0).all(), "invalid price data")
    return dates, load, pv, prices

def pca_basis(curves, k):
    mu = curves.mean(axis=0)
    _, _, vt = np.linalg.svd(curves-mu, full_matrices=False)
    return mu, vt[:k]

def weekday7(date):
    x = np.zeros(7)
    x[date.weekday()] = 1.0
    return x

def annual1(date):
    a = 2*np.pi*date.dayofyear/365.25
    return np.array([np.sin(a), np.cos(a)])

def forecast_scenarios(dates, load, pv):
    W = FROZEN_FORECAST["window_days"]
    KL = FROZEN_FORECAST["k_load"]
    KP = FROZEN_FORECAST["k_pv"]
    AL = FROZEN_FORECAST["alpha_load"]
    AP = FROZEN_FORECAST["alpha_pv"]
    S = FROZEN_FORECAST["scenario_days"]

    pred_l = np.full_like(load, np.nan)
    pred_p = np.full_like(pv, np.nan)
    scenarios = {}
    residuals = []

    # H2 PV history requires lag7. Starting Jan-15 gives a small warm-up fit;
    # by Feb-1 the model has only ended historical days, and from Mar onward up to W=60.
    for d in range(14, len(dates)):
        h0 = max(0, d-W)
        mu_l, phi_l = pca_basis(load[h0:d], KL)
        mu_p, phi_p = pca_basis(pv[h0:d], KP)

        sl = (load[h0:d]-mu_l) @ phi_l.T
        sp = (pv[h0:d]-mu_p) @ phi_p.T
        train = np.arange(max(h0+7, 7), d)

        def f_l(j):
            jj = j-h0
            return np.r_[weekday7(dates[j]), sl[jj-1]]

        def f_p(j):
            jj = j-h0
            return np.r_[
                annual1(dates[j]),
                sp[jj-1],
                sp[jj-7],
                sp[jj-7:jj].mean(axis=0),
            ]

        XL = np.vstack([f_l(j) for j in train])
        XP = np.vstack([f_p(j) for j in train])
        yL = np.vstack([sl[j-h0] for j in train])
        yP = np.vstack([sp[j-h0] for j in train])

        ml = make_pipeline(StandardScaler(), Ridge(alpha=AL))
        mp = make_pipeline(StandardScaler(), Ridge(alpha=AP))
        ml.fit(XL, yL)
        mp.fit(XP, yP)

        sh_l = np.atleast_1d(ml.predict(f_l(d)[None,:])[0])
        sh_p = np.atleast_1d(mp.predict(f_p(d)[None,:])[0])
        pred_l[d] = np.maximum(mu_l + sh_l @ phi_l, 0)
        pred_p[d] = np.maximum(mu_p + sh_p @ phi_p, 0)

        if residuals:
            recent = residuals[-S:]
            eL = np.array([r[1] for r in recent])
            eP = np.array([r[2] for r in recent])
            scenarios[d] = (
                np.maximum(pred_l[d][None,:] + eL, 0)
                - np.maximum(pred_p[d][None,:] + eP, 0)
            )

        residuals.append((d, load[d]-pred_l[d], pv[d]-pred_p[d]))

    return pred_l, pred_p, scenarios

def solve_dispatch(net_kwh, prices):
    net = np.asarray(net_kwh, float)
    n = len(net)
    require(n==144, "day must contain 144 slots")

    ec, ed = CFG["eta_c"], CFG["eta_d"]
    mc = CFG["max_charge_kw"] * CFG["interval_hours"]
    md = CFG["max_discharge_kw"] * CFG["interval_hours"]
    lo, hi = CFG["storage_min_kwh"], CFG["storage_max_kwh"]

    # x = [g,c,d,u,E_0..E_n]
    count = 5*n + 1
    obj = np.zeros(count)
    obj[:n] = prices

    rows, cols, data = [], [], []
    rhs = np.zeros(2*n)
    t = np.arange(n)

    for rr, cc, dd in [
        (t, t, np.ones(n)),
        (t, 2*n+t, np.ones(n)),
        (t, n+t, -np.ones(n)),
        (t, 3*n+t, -np.ones(n)),
    ]:
        rows.extend(rr); cols.extend(cc); data.extend(dd)
    rhs[:n] = net

    r = n+t
    for rr, cc, dd in [
        (r, 4*n+t+1, np.ones(n)),
        (r, 4*n+t, -np.ones(n)),
        (r, n+t, -ec*np.ones(n)),
        (r, 2*n+t, np.ones(n)/ed),
    ]:
        rows.extend(rr); cols.extend(cc); data.extend(dd)

    eq = coo_matrix((data,(rows,cols)), shape=(2*n,count)).tocsr()
    lower = np.zeros(count)
    upper = np.full(count, np.inf)
    upper[n:2*n] = mc
    upper[2*n:3*n] = md
    lower[4*n:] = lo
    upper[4*n:] = hi
    lower[4*n] = upper[4*n] = CFG["initial_storage_kwh"]
    lower[5*n] = upper[5*n] = CFG["terminal_storage_kwh"]

    res = linprog(
        obj, A_eq=eq, b_eq=rhs,
        bounds=np.column_stack([lower,upper]),
        method="highs",
    )
    require(res.success, "LP failed: "+res.message)
    x = res.x
    solver = "LP"

    overlap = (x[n:2*n] > 1e-7) & (x[2*n:3*n] > 1e-7)
    if overlap.any():
        m = count+n
        mode_rows = np.r_[t,t,n+t,n+t]
        mode_cols = np.r_[n+t,count+t,2*n+t,count+t]
        mode_data = np.r_[np.ones(n),-mc*np.ones(n),np.ones(n),md*np.ones(n)]
        mode = coo_matrix((mode_data,(mode_rows,mode_cols)), shape=(2*n,m)).tocsr()
        res2 = milp(
            np.r_[obj,np.zeros(n)],
            integrality=np.r_[np.zeros(count),np.ones(n)],
            bounds=Bounds(np.r_[lower,np.zeros(n)], np.r_[upper,np.ones(n)]),
            constraints=[
                LinearConstraint(hstack([eq,csr_matrix((2*n,n))]), rhs, rhs),
                LinearConstraint(mode, -np.inf, np.r_[np.zeros(n),md*np.ones(n)]),
            ],
            options={"mip_rel_gap":1e-8},
        )
        require(res2.success, "MILP failed: "+res2.message)
        x = res2.x[:count]
        solver = "MILP"

    return {
        "grid": np.maximum(x[:n],0),
        "charge": np.maximum(x[n:2*n],0),
        "discharge": np.maximum(x[2*n:3*n],0),
        "storage": x[4*n:],
        "solver": solver,
        "residual": float(np.max(np.abs(eq@x-rhs))),
    }

def execute_fixed_plan(plan, actual_net_kwh):
    supply = plan["grid"] + plan["discharge"] - plan["charge"]
    emergency = np.maximum(actual_net_kwh-supply, 0)
    unused = np.maximum(supply-actual_net_kwh, 0)
    balance = plan["grid"]+plan["discharge"]+emergency-actual_net_kwh-plan["charge"]-unused
    require(np.max(np.abs(balance))<1e-6, "execution balance failed")
    return emergency, unused

def run_q(dates, load, pv, prices, scenarios, day_idx, q, keep_slots=False):
    daily, slots = [], []
    for d in day_idx:
        qnet_kw = np.quantile(scenarios[d], q, axis=0)
        plan = solve_dispatch(qnet_kw*CFG["interval_hours"], prices)
        actual_net = (load[d]-pv[d])*CFG["interval_hours"]
        emergency, unused = execute_fixed_plan(plan, actual_net)
        pc = float(prices@plan["grid"])
        ec = float(CFG["emergency_multiplier"]*prices@emergency)

        daily.append({
            "q":q, "date":str(dates[d].date()),
            "planned_grid_kwh":float(plan["grid"].sum()),
            "planned_charge_kwh":float(plan["charge"].sum()),
            "planned_discharge_kwh":float(plan["discharge"].sum()),
            "emergency_kwh":float(emergency.sum()),
            "unused_kwh":float(unused.sum()),
            "planned_cost_yuan":pc,
            "emergency_cost_yuan":ec,
            "total_cost_yuan":pc+ec,
            "emergency_slots":int((emergency>1e-7).sum()),
            "solver":plan["solver"],
            "max_balance_residual":plan["residual"],
            "min_storage_kwh":float(plan["storage"].min()),
            "max_storage_kwh":float(plan["storage"].max()),
        })

        if keep_slots:
            for t in range(144):
                slots.append({
                    "date":str(dates[d].date()), "slot":t+1,
                    "forecast_net_q85_kw":float(qnet_kw[t]),
                    "actual_load_kw":float(load[d,t]),
                    "actual_pv_kw":float(pv[d,t]),
                    "actual_net_kwh":float(actual_net[t]),
                    "price_yuan_per_kwh":float(prices[t]),
                    "planned_grid_kwh":float(plan["grid"][t]),
                    "planned_charge_kwh":float(plan["charge"][t]),
                    "planned_discharge_kwh":float(plan["discharge"][t]),
                    "planned_storage_start_kwh":float(plan["storage"][t]),
                    "planned_storage_end_kwh":float(plan["storage"][t+1]),
                    "emergency_kwh":float(emergency[t]),
                    "unused_kwh":float(unused[t]),
                })
    return pd.DataFrame(daily), pd.DataFrame(slots)

def emergency_events(slot_df):
    rows = []
    for date, x in slot_df.groupby("date", sort=True):
        a = x.sort_values("slot")["emergency_kwh"].to_numpy()
        t=0
        def clock(m):
            if m==1440: return "24:00"
            return f"{(m//60)%24}:{m%60:02d}"
        while t<144:
            if a[t] <= 1e-7:
                t += 1
                continue
            start=t
            while t<144 and a[t] > 1e-7:
                t += 1
            rows.append({
                "date":date,
                "interval":f"{clock(start*10)}-{clock(t*10)}",
                "emergency_kwh":float(a[start:t].sum()),
                "start_slot":start+1,
                "end_slot":t,
            })
    return pd.DataFrame(rows)

def run_all(actual_path, price_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dates, load, pv, prices = load_inputs(actual_path, price_path)
    _, _, scenarios = forecast_scenarios(dates, load, pv)

    dev = np.where((dates>=pd.Timestamp(DEV_START)) & (dates<=pd.Timestamp(DEV_END)))[0]
    eval_idx = np.where((dates>=pd.Timestamp(EVAL_START)) & (dates<=pd.Timestamp(EVAL_END)))[0]
    require(all(d in scenarios for d in dev), "development scenarios missing")
    require(all(d in scenarios for d in eval_idx), "evaluation scenarios missing")

    scan_rows=[]
    for q in Q_GRID:
        ddf,_ = run_q(dates,load,pv,prices,scenarios,dev,float(q),False)
        scan_rows.append({
            "q":float(q),
            "planned_cost_yuan":float(ddf.planned_cost_yuan.sum()),
            "emergency_cost_yuan":float(ddf.emergency_cost_yuan.sum()),
            "total_cost_yuan":float(ddf.total_cost_yuan.sum()),
            "emergency_kwh":float(ddf.emergency_kwh.sum()),
            "unused_kwh":float(ddf.unused_kwh.sum()),
            "emergency_days":int((ddf.emergency_kwh>1e-7).sum()),
        })
    scan = pd.DataFrame(scan_rows)
    q_star=float(scan.loc[scan.total_cost_yuan.idxmin(),"q"])
    require(abs(q_star-FROZEN_Q_STAR)<1e-12, f"q* changed: {q_star}")
    scan["delta_cost_yuan"] = scan.total_cost_yuan - scan.total_cost_yuan.min()
    scan["delta_pct"] = scan.delta_cost_yuan / scan.total_cost_yuan.min() * 100
    scan.to_csv(outdir/"qstar_development_scan.csv",index=False,encoding="utf-8-sig")

    final_daily, final_slots = run_q(
        dates,load,pv,prices,scenarios,eval_idx,q_star,True
    )
    final_daily.to_csv(outdir/"final_strategy_daily.csv",index=False,encoding="utf-8-sig")
    final_slots.to_csv(outdir/"final_strategy_slots.csv",index=False,encoding="utf-8-sig")
    emergency_events(final_slots).to_csv(
        outdir/"final_emergency_events.csv",index=False,encoding="utf-8-sig"
    )

    require(len(final_slots)==334*144, "final trace length mismatch")
    require(final_slots.date.nunique()==334, "final date count mismatch")
    require((final_slots.planned_charge_kwh*final_slots.planned_discharge_kwh <= 1e-7).all(),
            "simultaneous charge/discharge detected")
    require(final_slots.planned_storage_start_kwh.min()>=1199.999999, "storage lower bound")
    require(final_slots.planned_storage_end_kwh.max()<=10800.000001, "storage upper bound")
    require(final_daily.max_balance_residual.max()<1e-6, "balance residual too large")

    report = {
        "q_star": q_star,
        "development_period": [DEV_START,DEV_END],
        "evaluation_period": [EVAL_START,EVAL_END],
        "evaluation_days": 334,
        "planned_purchase_kwh": float(final_daily.planned_grid_kwh.sum()),
        "emergency_purchase_kwh": float(final_daily.emergency_kwh.sum()),
        "planned_cost_yuan": float(final_daily.planned_cost_yuan.sum()),
        "emergency_cost_yuan": float(final_daily.emergency_cost_yuan.sum()),
        "total_cost_yuan": float(final_daily.total_cost_yuan.sum()),
        "status": "PASS",
    }
    (outdir/"verification_summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8"
    )
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--actual", type=Path, default=Path("input/q2_actual_load_pv.csv"))
    ap.add_argument("--price", type=Path, default=Path("input/q2_fixed_prices.csv"))
    ap.add_argument("--out", type=Path, default=Path("rerun_results"))
    args=ap.parse_args()
    t=time.perf_counter()
    report=run_all(args.actual,args.price,args.out)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"elapsed={time.perf_counter()-t:.2f}s")

if __name__=="__main__":
    main()
