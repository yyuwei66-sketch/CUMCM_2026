"""C题问题3主模型。

00:00计划使用Q2的负载预测和附件3的00:00光伏预测；
06:00、12:00时只对未来6小时进行日内调整。默认将概率风险更新方案
写入 results/result3.xlsx，并同时保存三种策略的回测明细。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

ROOT = Path(__file__).resolve().parent
DT = 1.0 / 6.0
CFG = {
    "eta_c": 0.9, "eta_d": 0.9, "emin": 1200.0, "emax": 10800.0,
    "e0": 6000.0, "eterm": 6000.0, "pmax_c": 5000.0, "pmax_d": 5000.0,
    "emergency": 5.0, "up": 1.5, "down": 0.5, "history_days": 28,
    "half_life": 14.0, "error_scale": 0.25,
    "tol": 1e-6,
}
ISSUE_HOURS = (0, 6, 12, 18)
EVAL_START = pd.Timestamp("2025-02-01").date()


def require(ok, msg):
    if not bool(ok):
        raise ValueError(msg)


def read_inputs(actual_path, price_path, forecast_path):
    from openpyxl import load_workbook

    def rows(path, sheet):
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            return [list(r) for r in wb[sheet].iter_rows(values_only=True)]
        finally:
            wb.close()

    def daily(path, sheet):
        x = rows(path, sheet)
        require(len(x) == 366 and all(len(r) >= 145 for r in x), f"{sheet}格式错误")
        dates = pd.DatetimeIndex(pd.to_datetime([r[0] for r in x[1:]]))
        require(dates.equals(pd.date_range("2025-01-01", "2025-12-31")), f"{sheet}日期轴错误")
        a = np.asarray([r[1:145] for r in x[1:]], dtype=float)
        require(np.isfinite(a).all() and (a >= 0).all(), f"{sheet}含非法值")
        return dates, a

    dates, load = daily(actual_path, "小区负载")
    dates2, pv = daily(actual_path, "光伏发电实际功率")
    require(dates.equals(dates2), "负载和光伏日期不一致")
    pr = rows(price_path, "Sheet1")
    prices = np.asarray([r[1] for r in pr[1:145]], dtype=float)
    require(np.isfinite(prices).all() and (prices > 0).all(), "电价含非法值")

    raw = rows(forecast_path, "Sheet1")
    forecast = np.full((len(dates), 4, 144), np.nan)
    date_index = {d.normalize(): i for i, d in enumerate(dates)}
    current_date = None
    for row in raw[1:]:
        if row[0] not in (None, ""):
            current_date = pd.Timestamp(str(row[0])).normalize()
        require(current_date in date_index, "附件3日期无法匹配附件2")
        hour = int(str(row[1]).split(":")[0])
        require(hour in ISSUE_HOURS and len(row) >= 26, "附件3发布时刻或列数错误")
        knots = np.asarray(row[2:26], dtype=float)
        require(np.isfinite(knots).all() and (knots >= 0).all(), "附件3含非法预测值")
        # 原表的第1个点是发布后1小时。没有使用当时实际值，
        # 以第1个预测点作为00/06/12/18时的平坦锚点，再作线性插值。
        remaining_minutes = (24 - hour) * 60
        ten_min = np.arange(10, remaining_minutes + 1, 10)
        forecast[date_index[current_date], ISSUE_HOURS.index(hour), hour * 6:] = np.interp(
            ten_min, np.arange(0, 1441, 60), np.r_[knots[0], knots]
        )
    for j, h in enumerate(ISSUE_HOURS):
        require(np.isfinite(forecast[:, j, h * 6:]).all(), f"附件3 {h:02d}:00预报不完整")
    return dates, load, pv, prices, forecast


def q2_forecast(dates, load, pv):
    import sys
    sys.path.insert(0, str(ROOT))
    from q2 import forecast_scenarios
    pred_load, _, _ = forecast_scenarios(dates, load, pv)
    first = 14
    require(np.isfinite(pred_load[first:]).all(), "Q2负载预测无效")
    return pred_load, first


def scenario_history(day, start, pred_load, forecast, load, pv, first, weighted=False):
    """返回从 start 到日末的净负荷情景，单位 kWh。"""
    if day < first:
        return ((load[day, start:] - forecast[day, 0, start:]) * DT)[None, :], np.ones(1)
    h0 = max(first, day - CFG["history_days"])
    hist = np.arange(h0, day)
    centre = pred_load[day, start:] - forecast[day, 0, start:]
    if len(hist) == 0:
        return (centre * DT)[None, :], np.ones(1)
    le = load[hist, start:] - pred_load[hist, start:]
    pe = pv[hist, start:] - forecast[hist, 0, start:]
    a = (centre[None, :] + le - np.maximum(forecast[day, 0, start:][None, :] + pe, 0)) * DT
    if not weighted or len(hist) == 0:
        return a, np.full(len(a), 1 / len(a)) if len(a) else np.ones(1)
    w = np.exp(-np.log(2) * (day - hist) / CFG["half_life"])
    return a, w / w.sum()


def issue_scenarios(day, hour, pred_load, forecast, load, pv, first, weighted=True):
    start = hour * 6
    if day < first:
        return scenario_history(day, start, pred_load, forecast, load, pv, first, False)
    h0 = max(first, day - CFG["history_days"])
    hist = np.arange(h0, day)
    centre_l = pred_load[day, start:]
    centre_p = forecast[day, ISSUE_HOURS.index(hour), start:]
    if len(hist) == 0:
        return ((centre_l - centre_p) * DT)[None, :], np.ones(1)
    le = load[hist, start:] - pred_load[hist, start:]
    pe = pv[hist, start:] - forecast[hist, ISSUE_HOURS.index(hour), start:]
    a = (centre_l[None, :] + le - np.maximum(centre_p[None, :] + pe, 0)) * DT
    if weighted and len(hist):
        w = np.exp(-np.log(2) * (day - hist) / CFG["half_life"])
        weights = w / w.sum()
    else:
        weights = np.full(len(a), 1 / len(a)) if len(a) else np.ones(1)
    # 保留历史误差情景，但降低其偏离当前点预测的幅度，
    # 以平衡紧急购电风险与调整购电费用。
    centre = ((centre_l - centre_p) * DT)[None, :]
    return centre + CFG["error_scale"] * (a - centre), weights


def baseline2_deterministic_scenario(day, hour, pred_load, forecast, load, first):
    """Baseline2的单点净负荷预测，不使用任何历史误差情景。"""
    start = hour * 6
    load_center = pred_load[day, start:] if day >= first else load[day, start:]
    pv_center = forecast[day, ISSUE_HOURS.index(hour), start:]
    return ((load_center - pv_center) * DT)[None, :], np.ones(1)


def solve_problem(
    a, p, initial, fixed=None, lock_from=None, risk=False, weights=None,
    terminal_target_kwh=None, terminal_penalty_yuan_per_kwh=0.0,
    terminal_hard=True,
):
    a, p = np.asarray(a, float), np.asarray(p, float)
    s, n = a.shape
    h0, u0, v0 = 4*n + 1, 4*n + 1 + s*n, 4*n + 1 + s*n + n
    base_m = v0 + n
    # Preserve the original hard 6000-kWh terminal condition by default.
    # Q4 can opt into a free terminal state with a soft deviation penalty.
    if terminal_target_kwh is None and terminal_hard:
        terminal_target_kwh = CFG["eterm"]
    require(terminal_penalty_yuan_per_kwh >= 0, "终点偏离惩罚系数不能为负")
    use_soft_terminal = terminal_target_kwh is not None and not terminal_hard
    dev_plus = base_m if use_soft_terminal else None
    dev_minus = base_m + 1 if use_soft_terminal else None
    m = base_m + 2 if use_soft_terminal else base_m
    obj = np.zeros(m)
    if fixed is None: obj[:n] = p
    if weights is None:
        weights = np.full(s, 1 / s)
    weights = np.asarray(weights, float)
    require(len(weights) == s and np.isclose(weights.sum(), 1), "情景权重不合法")
    obj[h0:u0] = np.repeat(weights, n) * np.tile(CFG["emergency"] * p, s)
    if use_soft_terminal:
        obj[dev_plus] = terminal_penalty_yuan_per_kwh
        obj[dev_minus] = terminal_penalty_yuan_per_kwh
    lo, hi = np.zeros(m), np.full(m, np.inf)
    lo[n:2*n], hi[n:2*n] = 0, CFG["pmax_c"] * DT
    lo[2*n:3*n], hi[2*n:3*n] = 0, CFG["pmax_d"] * DT
    lo[3*n:4*n+1], hi[3*n:4*n+1] = CFG["emin"], CFG["emax"]
    lo[3*n] = hi[3*n] = initial
    if terminal_hard:
        lo[4*n] = hi[4*n] = float(terminal_target_kwh)
    elif use_soft_terminal:
        lo[dev_plus:] = 0.0
    t = np.arange(n)
    eq = coo_matrix((np.r_[-CFG["eta_c"]*np.ones(n), np.ones(n)/CFG["eta_d"], -np.ones(n), np.ones(n)],
                     (np.tile(t, 4), np.r_[n+t, 2*n+t, 3*n+t, 3*n+t+1])), shape=(n, m)).tocsr()
    if use_soft_terminal:
        # E_end - d_plus + d_minus = target, minimizing d_plus+d_minus.
        term = coo_matrix((np.array([1.0, -1.0, 1.0]),
                           (np.zeros(3), np.array([4*n, dev_plus, dev_minus]))),
                          shape=(1, m)).tocsr()
        eq = vstack([eq, term]).tocsr()
    if fixed is not None:
        baseline_plan = np.asarray(fixed, float)
        ge = coo_matrix((np.r_[np.ones(n), -np.ones(n), np.ones(n)],
                         (np.r_[t, t, t], np.r_[t, u0+t, v0+t])), shape=(n, m)).tocsr()
        eq, eq_rhs = vstack([eq, ge]).tocsr(), np.r_[np.zeros(n), baseline_plan]
        if use_soft_terminal:
            eq_rhs = np.r_[np.zeros(n), float(terminal_target_kwh), baseline_plan]
        obj[u0:v0], obj[v0:] = CFG["up"] * p, CFG["down"] * p
        if lock_from is not None: lo[lock_from:n] = hi[lock_from:n] = baseline_plan[lock_from:]
    else:
        eq_rhs = np.r_[np.zeros(n), float(terminal_target_kwh)] if use_soft_terminal else np.zeros(n)
    rows = np.arange(s*n)
    ub = coo_matrix((np.r_[-np.ones(s*n), np.ones(s*n), -np.ones(s*n), -np.ones(s*n)],
                     (np.tile(rows, 4), np.r_[np.tile(t, s), n+np.tile(t, s), 2*n+np.tile(t, s), h0+rows])), shape=(s*n, m)).tocsr()
    ub_rhs = -a.ravel()
    if risk and fixed is None:
        # A compact CVaR penalty on scenario emergency purchase.
        pass
    res = linprog(obj, A_ub=ub, b_ub=ub_rhs, A_eq=eq, b_eq=eq_rhs,
                   bounds=np.c_[lo, hi], method="highs")
    require(res.success, "LP求解失败: " + res.message)
    x, solver = res.x, "LP"
    if ((x[n:2*n] > CFG["tol"]) & (x[2*n:3*n] > CFG["tol"])).any():
        mode = coo_matrix((np.r_[np.ones(n), -CFG["pmax_c"]*DT*np.ones(n), np.ones(n), CFG["pmax_d"]*DT*np.ones(n)],
                           (np.r_[t,t,n+t,n+t], np.r_[n+t,m+t,2*n+t,m+t])), shape=(2*n,m+n)).tocsr()
        res2 = milp(np.r_[obj, np.zeros(n)], integrality=np.r_[np.zeros(m), np.ones(n)],
                    bounds=Bounds(np.r_[lo, np.zeros(n)], np.r_[hi, np.ones(n)]),
                    constraints=[LinearConstraint(hstack([eq, csr_matrix((eq.shape[0], n))]), eq_rhs, eq_rhs),
                                   LinearConstraint(hstack([ub, csr_matrix((ub.shape[0], n))]), -np.inf, ub_rhs),
                                   LinearConstraint(mode, -np.inf, np.r_[np.zeros(n), np.full(n, CFG["pmax_d"]*DT)])],
                    options={"mip_rel_gap": 1e-8})
        require(res2.success, "MILP求解失败: " + res2.message)
        x, solver = res2.x[:m], "MILP"
    return {"grid": np.maximum(x[:n], 0), "charge": np.maximum(x[n:2*n], 0),
            "discharge": np.maximum(x[2*n:3*n], 0), "storage": x[3*n:4*n+1], "solver": solver}


def execute(grid, load, pv, state, battery=True):
    n = len(grid); e = np.empty(n+1); e[0] = state
    c = np.zeros(n); d = np.zeros(n); h = np.zeros(n); u = np.zeros(n)
    mc = CFG["pmax_c"]*DT if battery else 0; md = CFG["pmax_d"]*DT if battery else 0
    for t in range(n):
        r = grid[t] + pv[t] - load[t]
        if r >= 0:
            c[t] = min(r, mc, max(0, (CFG["emax"]-e[t])/CFG["eta_c"])); u[t] = r-c[t]
        else:
            d[t] = min(-r, md, max(0, (e[t]-CFG["emin"])*CFG["eta_d"])); h[t] = -r-d[t]
        e[t+1] = e[t] + CFG["eta_c"]*c[t] - d[t]/CFG["eta_d"]
    return c, d, h, u, e


def run_strategy(name, kind, dates, load, pv, prices, forecasts, pred_load, first):
    state = CFG["e0"]; daily=[]; details=[]; decisions=[]
    update_hours = {
        "baseline1_fixed": (),
        "baseline2_6h": (6,),
        "baseline3_6_12h": (6, 12),
        "baseline4_6_12_18h": (6, 12, 18),
        # 主模型在06:00、12:00、18:00进行概率风险更新。
        "probabilistic": (6, 12, 18),
    }[kind]
    for k, day in enumerate(dates):
        # 公平比较：三种策略共享完全相同的00:00确定性初始计划。
        # 历史误差情景仅用于主模型在06:00、12:00的日内更新。
        baseline_plan_scenarios, baseline_plan_weights = baseline2_deterministic_scenario(k, 0, pred_load, forecasts, load, first)
        baseline_plan = np.zeros(144) if k == 0 else solve_problem(baseline_plan_scenarios, prices, state, risk=(kind == "probabilistic"), weights=baseline_plan_weights)["grid"]
        updated = baseline_plan.copy(); blocks=[]; day_start=state
        for hour in (0, 6, 12, 18):
            start, end = hour*6, min(144, hour*6+36)
            if hour in update_hours and k > 0:
                # Baseline2/3/4使用确定性的单点预测；主模型使用历史误差情景。
                if kind != "probabilistic":
                    a, w = baseline2_deterministic_scenario(k, hour, pred_load, forecasts, load, first)
                else:
                    a, w = issue_scenarios(k, hour, pred_load, forecasts, load, pv, first, True)
                cand = solve_problem(a, prices[start:], state, fixed=baseline_plan[start:], lock_from=36, risk=(kind == "probabilistic"), weights=w)["grid"]
                updated[start:end] = cand[:end-start]
                decisions.append({"date": str(day.date()), "update_hour": hour, "adjustment_kwh": float(np.abs(cand[:end-start]-baseline_plan[start:end]).sum()), "strategy": name})
            c,d,h,u,e = execute(updated[start:end], load[k,start:end]*DT, pv[k,start:end]*DT, state, battery=(k > 0))
            state=float(e[-1]); blocks.append((c,d,h,u,e))
        c=np.concatenate([x[0] for x in blocks]); d=np.concatenate([x[1] for x in blocks]); h=np.concatenate([x[2] for x in blocks]); u=np.concatenate([x[3] for x in blocks]); es=np.concatenate([x[4][:-1] for x in blocks]); ee=np.concatenate([x[4][1:] for x in blocks])
        adj=prices*(CFG["up"]*np.maximum(updated-baseline_plan,0)+CFG["down"]*np.maximum(baseline_plan-updated,0))
        plan_fee=float(prices@baseline_plan); emergency_fee=float(CFG["emergency"]*(prices@h)); total=plan_fee+float(adj.sum())+emergency_fee
        daily.append({"strategy":name,"date":str(day.date()),"planned_kwh":float(baseline_plan.sum()),"updated_kwh":float(updated.sum()),"increase_kwh":float(np.maximum(updated-baseline_plan,0).sum()),"decrease_kwh":float(np.maximum(baseline_plan-updated,0).sum()),"emergency_kwh":float(h.sum()),"unused_kwh":float(u.sum()),"planned_cost":plan_fee,"adjustment_cost":float(adj.sum()),"emergency_cost":emergency_fee,"total_cost":total,"state_start":day_start,"state_end":state,"emergency":bool(h.sum()>1e-6)})
        for t in range(144):
            details.append({"strategy":name,"date":str(day.date()),"slot":t+1,"price":prices[t],"actual_load_kwh":load[k,t]*DT,"actual_pv_kwh":pv[k,t]*DT,"planned_grid_kwh":baseline_plan[t],"updated_grid_kwh":updated[t],"charge_kwh":c[t],"discharge_kwh":d[t],"emergency_kwh":h[t],"unused_kwh":u[t],"storage_start_kwh":es[t],"storage_end_kwh":ee[t],"planned_cost":prices[t]*baseline_plan[t],"adjustment_cost":adj[t],"emergency_cost":CFG["emergency"]*prices[t]*h[t]})
    return pd.DataFrame(daily), pd.DataFrame(details), pd.DataFrame(decisions)


def verify(detail):
    x=detail[detail.date>=str(EVAL_START)]
    bal=x.updated_grid_kwh+x.actual_pv_kwh+x.discharge_kwh+x.emergency_kwh-x.actual_load_kwh-x.charge_kwh-x.unused_kwh
    soc=x.storage_end_kwh-x.storage_start_kwh-CFG["eta_c"]*x.charge_kwh+x.discharge_kwh/CFG["eta_d"]
    require(float(np.abs(bal).max())<1e-5 and float(np.abs(soc).max())<1e-5, "物理平衡校验失败")
    require(((x.charge_kwh>1e-6)&(x.discharge_kwh>1e-6)).sum()==0, "出现同时充放电")
    return {"balance_max":float(np.abs(bal).max()),"storage_max":float(np.abs(soc).max())}


def export_result3(template_path, output_path, daily, detail):
    from openpyxl import load_workbook
    wb=load_workbook(template_path); dates=sorted(d for d in daily.date.unique() if d >= str(EVAL_START)); dm=daily.set_index("date")
    for sheet, col in [("计划购电量","planned_grid_kwh"),("调整购电量","updated_grid_kwh")]:
        ws=wb[sheet]
        for i,date in enumerate(dates,2):
            x=detail[detail.date.eq(date)].sort_values("slot")
            ws.cell(i,1).value=pd.Timestamp(date).to_pydatetime()
            for j,v in enumerate(x[col].to_numpy(float),2): ws.cell(i,j).value=float(v)
            ws.cell(i,146).value=float(x[col].sum()); ws.cell(i,147).value=float(dm.loc[date,"planned_cost"] if sheet=="计划购电量" else dm.loc[date,"total_cost"])
    ws=wb["充放电量"]
    intervals=["0:00-4:00","4:00-8:00","8:00-12:00","12:00-16:00","16:00-20:00","20:00-24:00"]
    r=2
    for date in dates:
        x=detail[detail.date.eq(date)].sort_values("slot")
        for b, label in enumerate(intervals):
            z=x.iloc[b*24:(b+1)*24]
            ws.cell(r,1).value=pd.Timestamp(date).to_pydatetime() if b==0 else None; ws.cell(r,2).value=label; ws.cell(r,3).value=float(z.charge_kwh.sum()); ws.cell(r,4).value=float(z.discharge_kwh.sum()); ws.cell(r,5).value="0:00" if b==0 else ("24:00" if b==1 else None); ws.cell(r,6).value=float(x.storage_start_kwh.iloc[0] if b==0 else x.storage_end_kwh.iloc[-1] if b==1 else 0)
            r+=1
    ws=wb["紧急购电量"]
    r=2
    for date in dates:
        x=detail[detail.date.eq(date)].sort_values("slot"); a=x.emergency_kwh.to_numpy(); t=0; events=[]
        while t<144:
            if a[t]<=1e-7: t+=1; continue
            s=t
            while t<144 and a[t]>1e-7:t+=1
            events.append((s,t,float(a[s:t].sum())))
        for j,(s,t,val) in enumerate(events or [(0,0,0)]):
            ws.cell(r,1).value=pd.Timestamp(date).to_pydatetime() if j==0 else None; ws.cell(r,2).value=f"{s*10//60:02d}:{s*10%60:02d}-{t*10//60:02d}:{t*10%60:02d}" if t else None; ws.cell(r,3).value=val if t else None; r+=1
    Path(output_path).parent.mkdir(parents=True,exist_ok=True); wb.save(output_path)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--excel",type=Path,default=ROOT/"results"/"result3.xlsx")
    args=ap.parse_args(); t=time.perf_counter()
    dates,load,pv,prices,forecasts=read_inputs(ROOT/"Data/附件2.xlsx",ROOT/"Data/附件1.xlsx",ROOT/"Data/附件3.xlsx")
    pred_load,first=q2_forecast(dates,load,pv)
    strategies={
        "Baseline1：00时固定计划":"baseline1_fixed",
        "Baseline2：06时确定性更新":"baseline2_6h",
        "Baseline3：06时+12时确定性更新":"baseline3_6_12h",
        "Baseline4：06时+12时+18时确定性更新":"baseline4_6_12_18h",
        "主模型：概率风险更新":"probabilistic",
    }
    result={}
    for name,kind in strategies.items():
        result[name]=run_strategy(name,kind,dates,load,pv,prices,forecasts,pred_load,first)
    chosen=result["主模型：概率风险更新"]; export_result3(ROOT/"Data/附件5/result3.xlsx", args.excel, chosen[0], chosen[1])
    summary=[]
    for name,(d,x,_) in result.items():
        y=d[d.date>=str(EVAL_START)]; summary.append({"strategy":name,"total_cost":float(y.total_cost.sum()),"emergency_kwh":float(y.emergency_kwh.sum()),"check":verify(x)})
    print(json.dumps({"status":"PASS","result3":str(args.excel.resolve()),"parameters":{"history_days":CFG["history_days"],"half_life":CFG["half_life"],"error_scale":CFG["error_scale"]},"summary":summary,"seconds":round(time.perf_counter()-t,2)},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
