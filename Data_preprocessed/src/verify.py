"""Independent checks on persisted data, source fidelity, boundaries and causal access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from preprocess import read_book, build_features, sha256, write_json
from data_access import CData

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        results.append({'check': name, 'passed': True})
    def csv(name, dates=()):
        return pd.read_csv(ROOT / 'processed' / (name + '.csv'), parse_dates=list(dates), float_precision='round_trip')
    a = csv('actuals_10min', ['source_date', 'sample_time', 'interval_start', 'interval_end'])
    q = csv('q1_typical_day')
    f = csv('pv_forecast_hourly', ['issue_time', 'target_time'])
    f10 = csv('pv_forecast_10min', ['issue_time', 'interval_end', 'interval_start', 'anchor_available_at'])
    source2 = read_book(ROOT / 'raw' / '附件2.xlsx')
    source4 = read_book(ROOT / 'raw' / '附件4.xlsx')
    for name, rows in [('load_kw', source2['小区负载']), ('pv_kw', source2['光伏发电实际功率']),
                       ('price_actual_yuan_per_kwh', source4['Sheet1'])]:
        source = np.asarray([r[1:] for r in rows[1:]], float).ravel()
        check('CSV exact numeric source fidelity: ' + name, np.array_equal(source, a[name].to_numpy()))
    source1 = np.asarray([r[1:] for r in read_book(ROOT / 'raw' / '附件1.xlsx')['Sheet1'][1:]], float)
    check('Q1 all numeric values exact', np.array_equal(source1, q[['price_yuan_per_kwh', 'load_kw', 'pv_forecast_kw']].to_numpy()))
    source3 = np.asarray([r[2:] for r in read_book(ROOT / 'raw' / '附件3.xlsx')['Sheet1'][1:]], float).ravel()
    check('Every hourly forecast source value exact', np.array_equal(source3, f.pv_forecast_kw.to_numpy()))
    check('Jan31 midnight belongs to original Jan31 row', a.loc[a.sample_time == pd.Timestamp('2025-02-01'), 'source_date'].iloc[0] == pd.Timestamp('2025-01-31'))
    check('All interval durations 10 minutes', (a.interval_end - a.interval_start).eq(pd.Timedelta(minutes=10)).all())
    cross = f[(f.issue_time == pd.Timestamp('2025-12-31 18:00')) & (f.lead_hours == 24)]
    check('Cross-year 24h forecast preserved', len(cross) == 1 and cross.target_time.iloc[0] == pd.Timestamp('2026-01-01 18:00') and not cross.target_in_actual_coverage.iloc[0])
    check('Negative net load retained', int((a.net_load_kw < 0).sum()) == int((a.pv_kw > a.load_kw).sum()))
    check('No artificial storage-power cap on PV', a.pv_kw.max() > 10000)
    hourly_knots = f10[f10.lead_minutes % 60 == 0].merge(f, left_on=['issue_time', 'interval_end'], right_on=['issue_time', 'target_time'], validate='one_to_one')
    check('All source forecast knots reproduced exactly', len(hourly_knots) == 35040 and np.array_equal(hourly_knots.pv_forecast_kw_interpolated, hourly_knots.pv_forecast_kw))
    first = f10[f10.issue_time == pd.Timestamp('2025-01-01')]
    check('Missing initial observation uses explicit forecast fallback', first.initial_anchor_kind.eq('first_hour_forecast_flat_fallback').all())
    # Counterfactual mutation: target-day and later measurements must not alter
    # any daily-plan input at/before the origin, including its lagged price features.
    origin = pd.Timestamp('2025-03-20')
    baseline = build_features(a, q)
    altered = a.copy()
    mask = altered.sample_time > origin
    altered.loc[mask, ['load_kw', 'pv_kw', 'price_actual_yuan_per_kwh']] += 12345.678
    perturbed = build_features(altered, q)
    cols = baseline.columns
    pd.testing.assert_frame_equal(baseline.loc[baseline.origin_time <= origin, cols],
                                  perturbed.loc[perturbed.origin_time <= origin, cols], check_exact=True)
    check('Future actual perturbation cannot change earlier plan features', True)
    api = CData(ROOT)
    check('History accessor stops at observation cutoff', api.history(origin).sample_time.max() == origin)
    features = api.day_ahead_features(origin, problem=3)
    check('Problem4 does not expose Q2/3 fixed price as known actual future price',
          'price_q2_q3_known_yuan_per_kwh' not in api.day_ahead_features(origin, problem=4))
    selected = api.pv_forecast(origin, features.interval_end)
    check('As-of forecast never uses future release', (selected.issue_time <= origin).all())
    # Change future forecast vintages, including ones for the very same targets.
    cache = api._cache['pv_forecast_10min']
    original = cache.copy()
    cache.loc[cache.issue_time > origin, 'pv_forecast_kw_interpolated'] += 99999
    pd.testing.assert_frame_equal(selected, api.pv_forecast(origin, features.interval_end), check_exact=True)
    api._cache['pv_forecast_10min'] = original
    check('Future forecast revisions cannot change an earlier as-of selection', True)
    noon = origin + pd.Timedelta(hours=12)
    remaining = pd.date_range(noon + pd.Timedelta(minutes=10), periods=72, freq='10min')
    check('At noon the noon vintage is selected', api.pv_forecast(noon, remaining).issue_time.eq(noon).all())
    try:
        api.pv_forecast('2024-12-31 23:00', [pd.Timestamp('2025-01-01 01:00')])
        raise AssertionError('Missing forecast silently filled')
    except ValueError:
        check('Missing already-issued forecast raises instead of future backfill', True)
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    check('All source/template hashes unchanged', all(sha256(ROOT / p) == h for p, h in manifest['source_sha256'].items()))
    check('All processed CSV hashes match manifest', all(sha256(ROOT / p) == h for p, h in manifest['output_sha256'].items()))
    write_json(ROOT / 'diagnostics' / 'independent_verification.json', {'status': 'passed', 'checks': results})
    print(json.dumps({'status': 'passed', 'independent_checks': len(results)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
