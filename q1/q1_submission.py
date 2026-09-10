"""Q1 template export and independent saved-XLSX validation (standard library only).

Default entry point: python q1/run_q1.py
Saved-file check and regression tests: python q1/run_q1.py --verify-only --self-test
The optional low-level CSV CLI remains available for external callers.

The input samples represent the preceding 10-minute interval. The original
template spans 00:10 today to 00:10 tomorrow. Its last interval uses a periodic
extension of this typical-day plan. This is a documented modeling convention,
not an official correction or a forecast of an unknown next day.
After integration, call export_result() with in-memory records to avoid CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import posixpath
import re
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
RNS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
TAG = lambda name: '{' + NS + '}' + name
SHEETS = ['计划购电量', '充放电量']
CONVENTION = ('采样值用于此前10分钟区间；保留原模板标签并按时间匹配。'
              '模板次日00:00–00:10按典型日计划周期延拓，等于本日00:00–00:10。'
              '此为建模约定，并非赛事官方确认。')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


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


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


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


def regression_checks(output, template, dispatch, source, cfg):
    """Reject former row-order export and corrupt saved files; leave no artifacts."""
    import copy
    outcomes = {}

    def reject(name, operation):
        try:
            operation()
        except (ValueError, KeyError) as exc:
            outcomes[name] = {'status':'PASS', 'rejection':str(exc)}
        else:
            raise AssertionError('Invalid sample accepted: '+name)

    def verify(path, d=dispatch, s=source, c=cfg):
        return verify_saved(path,template,d,s,c,accept_periodic_template=True)

    verify(output)
    reject('empty_template', lambda: verify(template))
    reject('periodic_convention_required', lambda: verify_saved(output,template,dispatch,source,cfg))
    with tempfile.TemporaryDirectory(prefix='q1_submission_test_') as directory:
        parts = workbook_parts(output)
        sheet_path = parts[SHEETS[0]]['path']
        def bad_workbook(name, replacements):
            xml = copy.deepcopy(parts[SHEETS[0]]['xml'])
            cells = {c.get('r'):c for c in xml.iter(TAG('c'))}
            for addr,(kind,value,formula) in replacements.items():
                cell = cells[addr]
                for child in list(cell): cell.remove(child)
                cell.set('t',kind)
                if formula: ET.SubElement(cell,TAG('f')).text = formula
                ET.SubElement(cell,TAG('v')).text = str(value)
            path = Path(directory)/(name+'.xlsx')
            with ZipFile(output) as original, ZipFile(path,'w') as dest:
                for entry in original.infolist():
                    dest.writestr(entry,ET.tostring(xml,encoding='utf-8') if entry.filename == sheet_path else original.read(entry.filename))
            return path
        old_order = bad_workbook('old_row_order',{
            f'B{i+2}':('n',row['grid_kwh'],None) for i,row in enumerate(dispatch)})
        reject('old_row_order',lambda: verify(old_order))
        for name,replacements in {
            'numeric_text':{'B2':('str',dispatch[1]['grid_kwh'],None)},
            'cached_formula':{'B2':('n',dispatch[1]['grid_kwh'],'1+1')},
            'nonfinite':{'B2':('n','NaN',None)},
            'changed_label':{'A2':('str','0:00-0:10',None)},
        }.items():
            path = bad_workbook(name,replacements)
            reject(name,lambda: verify(path))
    wrong_source = copy.deepcopy(source)
    wrong_source[0]['price_yuan_per_kwh'] = float(source[0]['price_yuan_per_kwh'])+0.01
    reject('changed_source_price',lambda: verify(output,s=wrong_source))
    missing = copy.deepcopy(dispatch)
    del missing[0]['pv_kwh']
    reject('missing_dispatch_field',lambda: verify(output,d=missing))
    wrong_cfg = dict(cfg, charge_efficiency=float(cfg['charge_efficiency'])*0.99)
    reject('stale_configuration',lambda: verify(output,c=wrong_cfg))
    require(not Path(directory).exists(),'Temporary regression artifacts were not removed')
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['template','dispatch','input','parameters','output']:
        parser.add_argument('--'+arg, required=True, type=Path)
    parser.add_argument('--accept-periodic-template',action='store_true',help=CONVENTION)
    parser.add_argument('--verify-only',action='store_true',help='Read and verify an already saved Excel without changing it')
    args = parser.parse_args()
    cfg = json.loads(args.parameters.read_text(encoding='utf-8-sig'))
    dispatch,source = load_csv(args.dispatch),load_csv(args.input)
    operation = verify_saved if args.verify_only else export_result
    if args.verify_only:
        result = operation(args.output,args.template,dispatch,source,cfg,
                           accept_periodic_template=args.accept_periodic_template)
    else:
        result = operation(args.template,args.output,dispatch,source,cfg,
                           accept_periodic_template=args.accept_periodic_template)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
