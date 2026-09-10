"""CUMCM 2026 C题第一问：单文件最终运行版。

功能：
1. 读取 q1_typical_day.csv；模型参数已内置在本文件中；
2. 求解最优购电/储能调度；
3. 检查能量平衡、储能状态、功率边界等物理约束；
4. 直接按官方 result1.xlsx 模板生成最终 q1/results/result1.xlsx；
5. 保存后重新读取 Excel，独立校验最终结果。

不生成图片，不生成中间 CSV，不生成 Markdown/JSON 报告。

默认运行：
    python q1.py

可选严格 MILP：
    python q1.py --solver milp
"""
from __future__ import annotations

import argparse
import math
import posixpath
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

SCRIPT_DIR = Path(__file__).resolve().parent
# 兼容 q1.py 放在 q1/ 或项目根目录两种情况。
if (SCRIPT_DIR / 'input' / 'q1_typical_day.csv').exists():
    Q1_DIR = SCRIPT_DIR
elif (SCRIPT_DIR / 'q1' / 'input' / 'q1_typical_day.csv').exists():
    Q1_DIR = SCRIPT_DIR / 'q1'
else:
    Q1_DIR = SCRIPT_DIR
PROJECT_ROOT = Q1_DIR.parent

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
RNS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
TAG = lambda name: '{' + NS + '}' + name
SHEETS = ['计划购电量', '充放电量']
CONVENTION = (
    '采样值用于此前10分钟区间；保留原模板标签并按时间匹配。'
    '模板次日00:00–00:10按典型日计划周期延拓，等于本日00:00–00:10。'
    '此为建模约定，并非赛事官方确认。'
)


# ============================================================
# Q1 固定模型参数（已内置，无需外部 JSON）
# ============================================================
CFG = {
    "interval_minutes": 10,
    "charge_efficiency": 0.9,
    "discharge_efficiency": 0.9,
    "storage_min_kwh": 1200.0,
    "storage_max_kwh": 10800.0,
    "initial_storage_kwh": 6000.0,
    "terminal_storage_kwh": 6000.0,
    "max_charge_power_kw": 5000.0,
    "max_discharge_power_kw": 5000.0,
    "allow_pv_curtailment": True,
    "allow_grid_export": False,
    "physical_tolerance": 0.000001,
    "assumption_note": (
        "暂将采样值视为此前10分钟区间平均功率；充放电效率各90%；"
        "初末电量6000kWh。对官方result1.xlsx中的时间标签按物理时段匹配；"
        "若模板覆盖至次日00:10，则采用典型日周期延拓约定。"
    ),
}


def default_template_path():
    """优先使用整理后的模板；没有时直接读取附件5中的官方原始模板。"""
    candidates = [
        PROJECT_ROOT / "Data_preprocessed" / "templates" / "result1.xlsx",
        PROJECT_ROOT / "Data_preprocessed" / "raw" / "附件5" / "result1.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


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


def number(value, label):
    require(not isinstance(value, bool), f'{label}: boolean is not a number')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label}: missing/non-numeric value {value!r}') from exc
    require(math.isfinite(value), f'{label}: nonfinite value')
    return value


def close(actual, expected, label, tol=1e-6):
    require(abs(number(actual, label)-number(expected, label)) <= tol,
            f'{label}: {actual} != {expected} (absolute tolerance {tol})')


def minute(label):
    match = re.fullmatch(r'(\d{1,2}):(\d{2})(?::00)?(?:\+(\d+))?', str(label).strip())
    require(match is not None, f'Unrecognized clock label: {label!r}')
    h, m, day = int(match[1]), int(match[2]), int(match[3] or 0)
    require(0 <= h <= 24 and 0 <= m < 60 and (h < 24 or m == 0), f'Invalid time {label}')
    return day*1440 + h*60 + m


def interval(label):
    parts = re.split(r'\s*[-–—~～]\s*', str(label).strip())
    require(len(parts) == 2, f'Unrecognized interval: {label!r}')
    start, end = map(minute, parts)
    require(end > start, f'Nonpositive interval: {label!r}')
    return start, end


def workbook_parts(path):
    """Read actual XLSX cell values without relying on a spreadsheet engine."""
    with ZipFile(path) as z:
        book = ET.fromstring(z.read('xl/workbook.xml'))
        relationships = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        paths = {rel.attrib['Id']: rel.attrib['Target'] for rel in relationships}
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        result = {}
        for sh in book.find(TAG('sheets')):
            target = paths[sh.attrib['{' + RNS + '}id']]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            xml = ET.fromstring(z.read(target))
            cells = {}
            for cell in xml.iter(TAG('c')):
                kind = cell.get('t', 'n')
                v = cell.find(TAG('v'))
                if kind == 's':
                    value = strings[int(v.text)] if v is not None else None
                elif kind == 'inlineStr':
                    value = ''.join(cell.find(TAG('is')).itertext())
                elif v is None or v.text is None:
                    value = None
                elif kind in ('str','e','b'):
                    value = v.text
                else:
                    value = number(v.text, f'{sh.attrib["name"]}!{cell.get("r")}')
                cells[cell.attrib['r']] = {'value':value, 'type':kind,
                                         'formula':cell.find(TAG('f')) is not None}
            result[sh.attrib['name']] = {'path':target, 'xml':xml, 'cells':cells}
    return result


def cell_value(sheet, address):
    return sheet['cells'].get(address, {}).get('value')


def validated_dispatch(dispatch, source, cfg):
    """Recompute energy, power, storage and cost from source and dispatch records."""
    tol = number(cfg.get('physical_tolerance', 1e-6), 'physical_tolerance')
    require(0 < tol <= 1e-4, 'Unexpected physical tolerance')
    close(cfg['interval_minutes'], 10, '10-minute intervals required')
    require(len(dispatch) == len(source) == 144, 'Exactly 144 input and dispatch records required')
    require(not cfg.get('allow_grid_export', False), 'Grid export is unsupported')
    ec, ed = number(cfg['charge_efficiency'],'eta_c'), number(cfg['discharge_efficiency'],'eta_d')
    require(0 < ec <= 1 and 0 < ed <= 1, 'Invalid efficiencies')
    lo, hi = number(cfg['storage_min_kwh'],'storage min'), number(cfg['storage_max_kwh'],'storage max')
    initial = number(cfg['initial_storage_kwh'],'initial storage')
    close(initial, cfg['terminal_storage_kwh'], 'Q1 requires equal initial/terminal storage', tol)
    require(lo <= initial <= hi, 'Initial storage out of range')
    rows, state = {}, initial
    for i,(d,s) in enumerate(zip(dispatch,source)):
        start, end = i*10, (i+1)*10
        close(s['slot'], i+1, 'Input slot')
        close(s['interval_start_minute'], start, 'Input interval start')
        close(s['interval_end_minute'], end, 'Input interval end')
        if 'sample_offset_minutes' in s:
            close(s['sample_offset_minutes'], end, 'Sample applies to preceding interval')
        require((minute(d['interval_start']),minute(d['interval_end'])) == (start,end),
                f'Dispatch row {i+1} is not physical interval {start}–{end}')
        close(d['slot'], i+1, 'Dispatch slot')
        price = number(s['price_yuan_per_kwh'],'source price')
        load = number(s['load_kw'],'source load')/6
        pv = number(s['pv_forecast_kw'],'source PV')/6
        require(price > 0 and load >= 0 and pv >= 0, 'Invalid source price/load/PV')
        g,c,dis,w,e0,e1 = [number(d[k],k) for k in
            ('grid_kwh','charge_kwh','discharge_kwh','curtailment_kwh','storage_start_kwh','storage_end_kwh')]
        for name,val in [('load_kwh',load),('pv_kwh',pv),('price_yuan_per_kwh',price)]:
            close(d[name],val,f'Original source {name} row {i+1}',tol)
        require(min(g,c,dis,w) >= -tol, 'Negative energy decision')
        require(w <= pv+tol, 'Curtailment exceeds PV')
        if not cfg.get('allow_pv_curtailment', False): close(w,0,'Curtailment forbidden',tol)
        require(not (c > tol and dis > tol), 'Simultaneous charge/discharge')
        require(c*6 <= number(cfg['max_charge_power_kw'],'charge limit')+tol, 'Charge limit exceeded')
        require(dis*6 <= number(cfg['max_discharge_power_kw'],'discharge limit')+tol, 'Discharge limit exceeded')
        close(g+pv+dis,load+c+w,'Energy balance',tol)
        close(e0,state,'Stored energy continuity',tol)
        state += ec*c-dis/ed
        close(e1,state,'Independent stored energy update',tol)
        require(lo-tol <= state <= hi+tol, 'Storage capacity bounds exceeded')
        close(d['purchase_cost_yuan'],price*g,'Dispatch interval cost',tol)
        rows[start] = {'grid':max(0.0,g),'charge':max(0.0,c),'discharge':max(0.0,dis),
                       'curtailment':max(0.0,w),'price':price,'load':load,'pv':pv,'storage_start':e0,'storage_end':e1}
    close(state, cfg['terminal_storage_kwh'], 'Reconstructed terminal energy', tol)
    return rows


def expected_cells(template, rows, *, accept_periodic_template=False):
    parts = workbook_parts(template)
    require(list(parts) == SHEETS, f'Unexpected template sheets: {list(parts)}')
    p,b = [parts[name] for name in SHEETS]
    for addr,val in {'A1':'时间段','B1':'购电量'}.items():
        require(cell_value(p,addr) == val, f'Unexpected plan header {addr}')
    for addr,val in {'A1':'时间段','B1':'充电量','C1':'放电量','D1':'时刻','E1':'储电量'}.items():
        require(cell_value(b,addr) == val, f'Unexpected battery header {addr}')
    plan = [(address,info['value']) for address,info in p['cells'].items()
            if re.fullmatch(r'A\d+',address) and address != 'A1' and info['value'] is not None]
    plan.sort(key=lambda x:int(x[0][1:]))
    require(len(plan) == 144, 'Template must contain 144 intervals')
    parsed = [interval(label) for _,label in plan]
    require(all(end-start == 10 for start,end in parsed), 'Template interval duration must be 10 minutes')
    require(all(parsed[i][1] == parsed[i+1][0] for i in range(143)), 'Template intervals are not contiguous')
    starts = [start for start,_ in parsed]
    require(starts in [list(range(0,1440,10)),list(range(10,1450,10))], 'Unexpected template time coverage')
    if starts[0] == 10:
        require(accept_periodic_template,
                'Shifted template requires the documented typical-day periodic-extension convention. '
                'Pass --accept-periodic-template only when using that convention.')
    require(len(set(start%1440 for start in starts)) == 144, 'Template interval mapping is not one-to-one')
    out = {name:{} for name in SHEETS}
    for (address,_),(start,_) in zip(plan,parsed):
        out[SHEETS[0]]['B'+address[1:]] = rows[start%1440]['grid']
    for r,begin in enumerate(range(0,1440,240),2):
        require(interval(cell_value(b,f'A{r}')) == (begin,begin+240), 'Unexpected 4-hour template label')
        for col,key in [('B','charge'),('C','discharge')]:
            out[SHEETS[1]][f'{col}{r}'] = math.fsum(rows[t][key] for t in range(begin,begin+240,10))
    require(minute(cell_value(b,'D2')) == 0 and minute(cell_value(b,'D3')) == 1440,
            'Unexpected storage boundary labels')
    out[SHEETS[1]]['E2'] = rows[0]['storage_start']
    out[SHEETS[1]]['E3'] = rows[1430]['storage_end']
    return out, parts


def verify_saved(output, template, dispatch, source, cfg, *, accept_periodic_template=False):
    """Read SAVED Excel, match labels, then independently recompute all controls."""
    rows = validated_dispatch(dispatch,source,cfg)
    expected, original = expected_cells(template, rows, accept_periodic_template=accept_periodic_template)
    saved = workbook_parts(output)
    require(list(saved) == list(original), 'Workbook sheets or sheet order changed')
    count = 0
    for name, targets in expected.items():
        for address,value in targets.items():
            info = saved[name]['cells'].get(address,{})
            require(info.get('type') == 'n' and not info.get('formula'),
                    f'{name}!{address}: result must be a populated numeric value')
            close(info.get('value'),value,f'Saved Excel {name}!{address}')
            count += 1
        # Every cell outside the result positions must retain the template value.
        for address in set(original[name]['cells']) | set(saved[name]['cells']):
            if address not in targets:
                require(cell_value(original[name],address) == cell_value(saved[name],address),
                        f'Unexpected value/label change: {name}!{address}')
    plan = saved[SHEETS[0]]
    actual_by_minute = {}
    for addr,info in plan['cells'].items():
        if re.fullmatch(r'A\d+',addr) and addr != 'A1' and info['value'] is not None:
            start,_ = interval(info['value'])
            actual_by_minute[start%1440] = number(cell_value(plan,'B'+addr[1:]),'Saved grid')
    close(math.fsum(actual_by_minute.values()),math.fsum(r['grid'] for r in rows.values()),'Saved daily purchase')
    # Prices use matched physical time, NEVER the same Excel/CSV row position.
    actual_cost = math.fsum(actual_by_minute[t]*rows[t]['price'] for t in rows)
    reference_cost = math.fsum(r['grid']*r['price'] for r in rows.values())
    close(actual_cost,reference_cost,'Saved daily cost')
    selected = {}
    for h in (10,12,14,16,18,20):
        close(actual_by_minute[h*60],rows[h*60]['grid'],f'Specified interval {h}:00–{h}:10')
        selected[f'{h:02d}:00-{h:02d}:10'] = actual_by_minute[h*60]
    battery = saved[SHEETS[1]]
    csum = math.fsum(number(cell_value(battery,f'B{r}'),'Saved charge') for r in range(2,8))
    dsum = math.fsum(number(cell_value(battery,f'C{r}'),'Saved discharge') for r in range(2,8))
    e0,e1 = [number(cell_value(battery,a),'Saved boundary storage') for a in ('E2','E3')]
    close(e1-e0,number(cfg['charge_efficiency'],'eta_c')*csum-dsum/number(cfg['discharge_efficiency'],'eta_d'),
          'Saved daily storage energy equation')
    return {'status':'PASS','numeric_result_cells':count,'plan_intervals':len(actual_by_minute),
            'purchase_kwh':math.fsum(actual_by_minute.values()),'cost_yuan':actual_cost,
            'charge_kwh':csum,'discharge_kwh':dsum,'initial_kwh':e0,'terminal_kwh':e1,
            'specified_intervals_kwh':selected,
            'four_hour_blocks':[{'interval':cell_value(battery,f'A{r}'),
                                'charge_kwh':cell_value(battery,f'B{r}'),
                                'discharge_kwh':cell_value(battery,f'C{r}')} for r in range(2,8)],
            'template_first_grid_kwh':cell_value(plan,'B2'),
            'template_last_grid_kwh':cell_value(plan,'B145'),'time_convention':CONVENTION,
            'checks':['input and dispatch match','physical constraints','numeric cells complete',
                      'all 144 label-matched values','template labels preserved','six specified intervals',
                      'daily purchase and price-matched cost','six charge/discharge blocks','boundary energy']}


def export_result(template, output, dispatch, source, cfg, *, accept_periodic_template=False):
    """Fill existing XML numeric cells; leave every unrelated ZIP entry unchanged.

    Intended for the user's local Python project, with no extra Excel dependency.
    The temporary workbook is read back and verified before replacing output.
    """
    template,output = Path(template).resolve(),Path(output).resolve()
    require(template != output, 'Do not overwrite the raw template')
    rows = validated_dispatch(dispatch,source,cfg)
    expected,parts = expected_cells(template,rows,accept_periodic_template=accept_periodic_template)
    updates = {}
    with ZipFile(template) as z:
        for name,values in expected.items():
            path = parts[name]['path']
            data = z.read(path).decode('utf-8')
            for addr,value in values.items():
                # Replace only the cell body; preserve its address/style attributes
                # and all other original XML bytes, including namespace prefixes.
                pattern = re.compile(r'(<(?:\w+:)?c\b(?=[^>]*\br="'+re.escape(addr)+r'")[^>]*?)(?:\s*/>|>.*?</(?:\w+:)?c>)',re.DOTALL)
                match = pattern.search(data)
                if match is not None:
                    head = re.sub(r'\s+t="[^"]*"','',match[1].rstrip('/'))
                    prefix = re.match(r'<((?:\w+:)?)c\b',head)[1]
                    fragment = head+f' t="n"><{prefix}v>{value:.17g}</{prefix}v></{prefix}c>'
                    data = data[:match.start()]+fragment+data[match.end():]
                else:
                    # Blank template result cells can be entirely absent in XML.
                    # Insert them into their existing row in column order.
                    rownum = re.search(r'\d+$',addr)[0]
                    rowpat = re.compile(r'(<((?:\w+:)?)row\b(?=[^>]*\br="'+rownum+r'")[^>]*>)(.*?)(</(?:\w+:)?row>)',re.DOTALL)
                    rowmatch = rowpat.search(data)
                    require(rowmatch is not None, f'Template row {rownum} absent')
                    prefix = rowmatch[2]
                    fragment = f'<{prefix}c r="{addr}" t="n"><{prefix}v>{value:.17g}</{prefix}v></{prefix}c>'
                    body = rowmatch[3]
                    col = re.match(r'[A-Z]+',addr)[0]
                    def colnum(label):
                        n = 0
                        for char in label: n = n*26+ord(char)-64
                        return n
                    insert_at = len(body)
                    for other in re.finditer(r'<(?:\w+:)?c\b[^>]*\br="([A-Z]+)\d+"',body):
                        if colnum(other[1]) > colnum(col):
                            insert_at = other.start(); break
                    body = body[:insert_at]+fragment+body[insert_at:]
                    data = data[:rowmatch.start()]+rowmatch[1]+body+rowmatch[4]+data[rowmatch.end():]
            updates[path] = data.encode('utf-8')
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.xlsx',delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with ZipFile(tmp_path,'w') as dest:
                for entry in z.infolist():
                    dest.writestr(entry,updates.get(entry.filename,z.read(entry.filename)))
            result = verify_saved(tmp_path,template,dispatch,source,cfg,
                                  accept_periodic_template=accept_periodic_template)
            tmp_path.replace(output)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    # Also reopen the final pathname, after the atomic replacement.
    return verify_saved(output,template,dispatch,source,cfg,
                        accept_periodic_template=accept_periodic_template)


def build_dispatch_records(source: pd.DataFrame, solution: dict):
    """Convert the optimizer result to the records required by result1.xlsx export.

    Records stay in memory: no intermediate CSV files are written.
    """
    price = source.price_yuan_per_kwh.to_numpy(dtype=float)
    load = source.load_kw.to_numpy(dtype=float) / 6
    pv = source.pv_forecast_kw.to_numpy(dtype=float) / 6

    records = []
    for i in range(len(source)):
        records.append({
            'slot': int(source.iloc[i]['slot']),
            'interval_start': clock(int(source.iloc[i]['interval_start_minute'])),
            'interval_end': clock(int(source.iloc[i]['interval_end_minute'])),
            'price_yuan_per_kwh': float(price[i]),
            'load_kwh': float(load[i]),
            'pv_kwh': float(pv[i]),
            'grid_kwh': float(solution['grid'][i]),
            'charge_kwh': float(solution['charge'][i]),
            'discharge_kwh': float(solution['discharge'][i]),
            'curtailment_kwh': float(solution['curtailment'][i]),
            'storage_start_kwh': float(solution['storage'][i]),
            'storage_end_kwh': float(solution['storage'][i + 1]),
            'purchase_cost_yuan': float(price[i] * solution['grid'][i]),
        })
    return records


def solve_from_input(input_path: Path, solver: str):
    """Read Q1 input and solve using the parameters embedded in CFG."""
    cfg = dict(CFG)
    source = pd.read_csv(input_path, float_precision='round_trip')
    require(len(source) == 144 and cfg['interval_minutes'] == 10,
            'This runner expects one full day in 10-minute intervals')
    require(np.array_equal(source.interval_start_minute, np.arange(0, 1440, 10)),
            'Unexpected time mapping')
    require(np.array_equal(source.interval_end_minute, np.arange(10, 1441, 10)),
            'Unexpected end-time mapping')

    price = source.price_yuan_per_kwh.to_numpy()
    load = source.load_kw.to_numpy() / 6
    pv = source.pv_forecast_kw.to_numpy() / 6
    solution = solve_dispatch(price, load, pv, cfg, solver)
    checks = check_solution(solution, price, load, pv, cfg)
    return cfg, source, solution, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', choices=['auto', 'milp'], default='auto')
    parser.add_argument('--input', type=Path, default=Q1_DIR / 'input' / 'q1_typical_day.csv')
    parser.add_argument('--template', type=Path, default=default_template_path())
    parser.add_argument('--output', type=Path, default=Q1_DIR / 'results' / 'result1.xlsx')
    args = parser.parse_args()

    input_path = args.input.resolve()
    template_path = args.template.resolve()
    output_path = args.output.resolve()

    for path, label in [
        (input_path, 'input'),
        (template_path, 'template'),
    ]:
        require(path.exists(), f'Missing {label} file: {path}')

    # 1) Solve Q1 and check all physical constraints.
    cfg, source, solution, _checks = solve_from_input(
        input_path, args.solver
    )

    # 2) Keep dispatch in memory; do not write intermediate CSVs.
    dispatch_records = build_dispatch_records(source, solution)
    source_records = source.to_dict(orient='records')

    # 3) Fill the official template and independently verify the SAVED workbook.
    validation = export_result(
        template_path,
        output_path,
        dispatch_records,
        source_records,
        cfg,
        accept_periodic_template=True,
    )

    # 4) Console-only summary. The only generated file is result1.xlsx.
    price = source.price_yuan_per_kwh.to_numpy(dtype=float)
    load = source.load_kw.to_numpy(dtype=float) / 6
    pv = source.pv_forecast_kw.to_numpy(dtype=float) / 6
    baseline_grid = np.maximum(load - pv, 0)
    baseline_cost = float(price @ baseline_grid)
    savings = baseline_cost - solution['cost']
    savings_percent = 100 * savings / baseline_cost if baseline_cost else 0.0

    print('=' * 60)
    print('Q1 COMPLETED: PASS')
    print(f'Solver: {solution["solver"]}')
    print(f'Grid purchase: {solution["grid"].sum():.4f} kWh')
    print(f'Purchase cost: {solution["cost"]:.4f} yuan')
    print(f'Baseline cost: {baseline_cost:.4f} yuan')
    print(f'Savings: {savings:.4f} yuan ({savings_percent:.4f}%)')
    print(f'Initial / terminal storage: '
          f'{solution["storage"][0]:.4f} / {solution["storage"][-1]:.4f} kWh')
    print(f'Saved-XLSX validation: {validation["status"]}')
    print(f'Output: {output_path}')
    print('Generated artifacts: result1.xlsx only')
    print('=' * 60)


if __name__ == '__main__':
    main()
