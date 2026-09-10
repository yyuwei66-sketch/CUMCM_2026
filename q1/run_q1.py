"""CUMCM 2026 C Q1: reproducible dispatch calculation and figures.

Run: python run_q1.py
Optional strict integer solve: python run_q1.py --solver milp
Inputs are read-only. Generated CSV tables use explicit physical interval labels,
not the unresolved labels in the original competition template.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def solve_dispatch(price, load, pv, cfg, solver='auto'):
    """All three arrays contain price (yuan/kWh), load and PV energies (kWh).

    Continuous vector ordering: g[0:n], c[0:n], d[0:n], w[0:n], E[0:n+1].
    Strict MILP appends binary charge-mode z[0:n].
    """
    price, load, pv = [np.asarray(v, dtype=float) for v in (price, load, pv)]
    n = len(price)
    require(n > 0 and len(load) == n and len(pv) == n, 'Input lengths differ or are empty')
    require(np.isfinite(np.r_[price, load, pv]).all(), 'Nonfinite input')
    require((price > 0).all() and (load >= 0).all() and (pv >= 0).all(), 'This Q1 solver expects positive prices and nonnegative load/PV')
    require(solver in ('auto', 'milp'), 'Unknown solver mode')
    ec, ed = cfg['charge_efficiency'], cfg['discharge_efficiency']
    require(0 < ec <= 1 and 0 < ed <= 1, 'Efficiencies must lie in (0,1]')
    lo, hi = cfg['storage_min_kwh'], cfg['storage_max_kwh']
    e0, en = cfg['initial_storage_kwh'], cfg['terminal_storage_kwh']
    require(0 <= lo <= e0 <= hi and lo <= en <= hi, 'Invalid storage boundaries')
    require(cfg['interval_minutes'] > 0, 'Invalid interval duration')
    require(not cfg['allow_grid_export'], 'Grid export is not implemented; do not silently change the model')
    mc = cfg['max_charge_power_kw'] * cfg['interval_minutes'] / 60
    md = cfg['max_discharge_power_kw'] * cfg['interval_minutes'] / 60
    require(mc >= 0 and md >= 0, 'Negative power limit')
    count = 5 * n + 1
    objective = np.zeros(count)
    objective[:n] = price
    eq = np.zeros((2 * n, count))
    rhs = np.zeros(2 * n)
    for t in range(n):
        # g + PV + d = load + c + w
        eq[t, t], eq[t, n+t], eq[t, 2*n+t], eq[t, 3*n+t] = 1, -1, 1, -1
        rhs[t] = load[t] - pv[t]
        # E[t+1] - E[t] - ec*c + d/ed = 0
        eq[n+t, 4*n+t+1], eq[n+t, 4*n+t] = 1, -1
        eq[n+t, n+t], eq[n+t, 2*n+t] = -ec, 1/ed
    bounds = ([(0, None)] * n + [(0, mc)] * n + [(0, md)] * n
              + [(0, float(v) if cfg['allow_pv_curtailment'] else 0) for v in pv]
              + [(lo, hi)] * (n+1))
    bounds[4*n], bounds[5*n] = (e0, e0), (en, en)
    started = time.perf_counter()
    relaxed = linprog(objective, A_eq=eq, b_eq=rhs, bounds=bounds, method='highs')
    require(relaxed.success, 'LP failed: ' + relaxed.message)
    tol = cfg['physical_tolerance']
    overlap = (relaxed.x[n:2*n] > tol) & (relaxed.x[2*n:3*n] > tol)
    if solver == 'auto' and not overlap.any():
        x = relaxed.x
        proof = 'LP optimum satisfies charge/discharge exclusivity; hence it also attains the MILP optimum within solver tolerance.'
        used = 'HiGHS LP; exclusivity verified'
        gap = 0.0
    else:
        full_obj = np.r_[objective, np.zeros(n)]
        lower = np.array([b[0] for b in bounds] + [0] * n)
        upper = np.array([np.inf if b[1] is None else b[1] for b in bounds] + [1] * n)
        full_eq = np.pad(eq, ((0,0), (0,n)))
        modes = np.zeros((2*n, count+n))
        for t in range(n):
            modes[t, n+t], modes[t, count+t] = 1, -mc
            modes[n+t, 2*n+t], modes[n+t, count+t] = 1, md
        result = milp(full_obj, integrality=np.r_[np.zeros(count), np.ones(n)],
                      bounds=Bounds(lower, upper),
                      constraints=[LinearConstraint(full_eq, rhs, rhs),
                                   LinearConstraint(modes, np.full(2*n, -np.inf), np.r_[np.zeros(n), np.full(n, md)])],
                      options={'mip_rel_gap': 1e-9})
        require(result.success, 'MILP failed: ' + result.message)
        x = result.x[:count]
        used = 'HiGHS MILP'
        gap = float(result.mip_gap)
        proof = 'MILP solver reports optimality at the configured relative gap tolerance.'
    return {'grid': x[:n], 'charge': x[n:2*n], 'discharge': x[2*n:3*n],
            'curtailment': x[3*n:4*n], 'storage': x[4*n:],
            'cost': float(objective @ x), 'lp_lower_bound': float(relaxed.fun),
            'solver': used, 'mip_gap': gap, 'optimality_basis': proof,
            'elapsed_seconds': time.perf_counter() - started}


def clock(minutes):
    return f'{int(minutes)//60:02d}:{int(minutes)%60:02d}'


def check_solution(solution, price, load, pv, cfg):
    g, c, d, w, e = [solution[k] for k in ('grid', 'charge', 'discharge', 'curtailment', 'storage')]
    tol = cfg['physical_tolerance']
    ec, ed = cfg['charge_efficiency'], cfg['discharge_efficiency']
    duration = cfg['interval_minutes']/60
    balance = g + pv + d - load - c - w
    evolution = e[1:] - e[:-1] - ec*c + d/ed
    violations = {
        'balance_max_residual_kwh': float(np.max(abs(balance))),
        'storage_update_max_residual_kwh': float(np.max(abs(evolution))),
        'nonnegativity_violation_kwh': float(max(0, -np.min(np.r_[g,c,d,w]))),
        'curtailment_upper_violation_kwh': float(max(0, np.max(w-pv))),
        'storage_bound_violation_kwh': float(max(0, cfg['storage_min_kwh']-e.min(), e.max()-cfg['storage_max_kwh'])),
        'power_bound_violation_kw': float(max(0, c.max()/duration-cfg['max_charge_power_kw'], d.max()/duration-cfg['max_discharge_power_kw'])),
        'initial_storage_error_kwh': float(abs(e[0]-cfg['initial_storage_kwh'])),
        'terminal_storage_error_kwh': float(abs(e[-1]-cfg['terminal_storage_kwh'])),
        'simultaneous_charge_discharge_slots': int(((c > tol) & (d > tol)).sum()),
        'cost_recalculation_error_yuan': float(abs(price@g-solution['cost'])),
    }
    require(all(v <= tol for v in violations.values()), 'Physical check failed: ' + str(violations))
    if not cfg['allow_pv_curtailment']:
        require(np.max(abs(w)) <= tol, 'Curtailment forbidden by configuration')
    return violations


def make_figures(source, schedule, states, directory, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    directory.mkdir(exist_ok=True)
    x = np.r_[source.interval_start_minute.to_numpy()/60, 24]
    def step(ax, y, **kwargs):
        ax.step(x, np.r_[y, y[-1]], where='post', **kwargs)
    def finish(ax):
        ax.set_xlim(0, 24); ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlabel('Hour of day'); ax.grid(axis='y', alpha=.22)
    fig, ax = plt.subplots(figsize=(10,4.3), layout='constrained')
    step(ax, source.load_kw.to_numpy(), color='#244A73', linewidth=1.8, label='Load')
    step(ax, source.pv_forecast_kw.to_numpy(), color='#D29324', linewidth=1.8, label='PV forecast')
    ax.set_title('Q1 | Typical-day load and PV'); ax.set_ylabel('Power (kW)')
    ax.legend(loc='upper left', frameon=False); finish(ax)
    fig.savefig(directory/'01_load_pv.png', dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4.3), layout='constrained')
    factor = 60/cfg['interval_minutes']
    step(ax, schedule.grid_kwh.to_numpy()*factor, color='#333333', linewidth=1.5, label='Grid purchase')
    step(ax, schedule.charge_kwh.to_numpy()*factor, color='#3D7DA6', linewidth=1.4, label='Battery charge (+)')
    step(ax, -schedule.discharge_kwh.to_numpy()*factor, color='#B56D32', linewidth=1.4, label='Battery discharge (-)')
    ax.axhline(0, color='#777777', linewidth=.7)
    ax.set_title('Q1 | Grid and battery dispatch'); ax.set_ylabel('Power (kW)')
    ax.legend(loc='upper left', ncol=3, frameon=False); finish(ax)
    ax.margins(y=.15)
    fig.savefig(directory/'02_dispatch.png', dpi=200); plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(10,6), sharex=True, layout='constrained')
    axes[0].plot(states.minute/60, states.stored_energy_kwh, color='#244A73', linewidth=1.8)
    for limit in (cfg['storage_min_kwh'], cfg['storage_max_kwh']):
        axes[0].axhline(limit, color='#888888', linestyle='--', linewidth=.9)
    axes[0].set_ylim(0, max(12000, cfg['storage_max_kwh']*1.1))
    axes[0].set_ylabel('Stored energy (kWh)'); axes[0].set_title('Q1 | Stored energy and electricity price')
    axes[0].grid(axis='y', alpha=.22)
    step(axes[1], source.price_yuan_per_kwh.to_numpy(), color='#B56D32', linewidth=1.7)
    axes[1].set_ylabel('Price (yuan/kWh)'); finish(axes[1])
    fig.savefig(directory/'03_storage_price.png', dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', choices=['auto', 'milp'], default='auto')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'parameters.json').read_text(encoding='utf-8'))
    input_path = ROOT/'input'/'q1_typical_day.csv'
    source = pd.read_csv(input_path, float_precision='round_trip')
    require(len(source) == 144 and cfg['interval_minutes'] == 10, 'This runner expects one full day in 10-minute intervals')
    require(np.array_equal(source.interval_start_minute, np.arange(0,1440,10)), 'Unexpected time mapping')
    require(np.array_equal(source.interval_end_minute, np.arange(10,1441,10)), 'Unexpected end-time mapping')
    price = source.price_yuan_per_kwh.to_numpy()
    load, pv = source.load_kw.to_numpy()/6, source.pv_forecast_kw.to_numpy()/6
    solution = solve_dispatch(price, load, pv, cfg, args.solver)
    checks = check_solution(solution, price, load, pv, cfg)
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    schedule = pd.DataFrame({'slot': source.slot, 'interval_start': [clock(m) for m in source.interval_start_minute],
                             'interval_end': [clock(m) for m in source.interval_end_minute],
                             'price_yuan_per_kwh': price, 'load_kwh': load, 'pv_kwh': pv,
                             'grid_kwh': solution['grid'], 'charge_kwh': solution['charge'],
                             'discharge_kwh': solution['discharge'], 'curtailment_kwh': solution['curtailment'],
                             'storage_start_kwh': solution['storage'][:-1], 'storage_end_kwh': solution['storage'][1:]})
    schedule['purchase_cost_yuan'] = schedule.price_yuan_per_kwh * schedule.grid_kwh
    schedule['balance_residual_kwh'] = (schedule.grid_kwh+schedule.pv_kwh+schedule.discharge_kwh
                                      -schedule.load_kwh-schedule.charge_kwh-schedule.curtailment_kwh)
    states = pd.DataFrame({'minute': np.arange(0,1441,10), 'time': [clock(m) for m in range(0,1441,10)],
                           'stored_energy_kwh': solution['storage']})
    wanted = [10,12,14,16,18,20]
    table1 = schedule.iloc[[h*6 for h in wanted]][['interval_start','interval_end','grid_kwh']].reset_index(drop=True)
    table2 = pd.DataFrame([{
        'interval_start': clock(h*60), 'interval_end': clock((h+4)*60),
        'charge_kwh': float(solution['charge'][h*6:(h+4)*6].sum()),
        'discharge_kwh': float(solution['discharge'][h*6:(h+4)*6].sum()),
        'storage_start_kwh': float(solution['storage'][h*6]),
        'storage_end_kwh': float(solution['storage'][(h+4)*6])} for h in range(0,24,4)])
    baseline_grid = np.maximum(load-pv,0)
    baseline_cost = float(price@baseline_grid)
    baseline = pd.DataFrame({'interval_start': schedule.interval_start, 'interval_end': schedule.interval_end,
                             'grid_kwh': baseline_grid, 'curtailment_kwh': np.maximum(pv-load,0),
                             'purchase_cost_yuan': price*baseline_grid})
    for name, frame in [('dispatch_10min',schedule), ('storage_states',states), ('paper_table1',table1),
                        ('paper_table2',table2), ('baseline_no_storage',baseline)]:
        frame.to_csv(out/(name+'.csv'), index=False, encoding='utf-8-sig')
    summary = {
        'status': 'PASS', 'time_mapping_status': 'PROVISIONAL_UNRESOLVED_TEMPLATE_LABELS',
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(), 'parameters': cfg,
        'solver': solution['solver'], 'optimality_basis': solution['optimality_basis'],
        'lp_lower_bound_yuan': solution['lp_lower_bound'], 'mip_gap': solution['mip_gap'],
        'solve_seconds': solution['elapsed_seconds'], 'cost_yuan': solution['cost'],
        'grid_kwh': float(solution['grid'].sum()), 'load_kwh': float(load.sum()), 'pv_kwh': float(pv.sum()),
        'charge_kwh': float(solution['charge'].sum()), 'discharge_kwh': float(solution['discharge'].sum()),
        'curtailment_kwh': float(solution['curtailment'].sum()),
        'baseline_cost_yuan': baseline_cost, 'savings_yuan': baseline_cost-solution['cost'],
        'savings_percent': 100*(baseline_cost-solution['cost'])/baseline_cost,
        'physical_checks': checks,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.no_plots:
        make_figures(source,schedule,states,out/'figures',cfg)
    report = f'''# 第一问计算结果

时间口径：{cfg['assumption_note']}

求解方法：{solution['solver']}。

| 指标 | 结果 |
|---|---:|
| 全天购电量（kWh） | {summary['grid_kwh']:.4f} |
| 全天购电费用（元） | {summary['cost_yuan']:.4f} |
| 无储能购电费用（元） | {baseline_cost:.4f} |
| 节省费用（元） | {summary['savings_yuan']:.4f} |
| 费用下降比例 | {summary['savings_percent']:.4f}% |
| 弃光电量（kWh） | {summary['curtailment_kwh']:.4f} |
| 初始/结束储电量（kWh） | {solution['storage'][0]:.4f} / {solution['storage'][-1]:.4f} |

物理约束校验全部通过。完整未舍入结果见dispatch_10min.csv，题面指定时段汇总见paper_table1.csv、paper_table2.csv。

当前最优性依据：{solution['optimality_basis']}

4小时汇总中充电和放电都非零，表示不同10分钟时段分别充电和放电，不表示同时发生。
不同版本求解器可能返回等价的最优充放电安排，应以费用、约束和模型假设核对，不要求所有决策逐格相同。
'''
    (out/'第一问结果说明.md').write_text(report,encoding='utf-8')
    print('Q1 SOLVED: PASS')
    print(f'Cost: {summary["cost_yuan"]:.4f} yuan')
    print(f'Grid purchase: {summary["grid_kwh"]:.4f} kWh')
    print(f'Savings vs no storage: {summary["savings_percent"]:.4f}%')
    print('Physical checks: PASS')
    print('Output:',out)
    print('Time labels: provisional; original competition template is not filled.')


if __name__ == '__main__':
    main()
