"""Time-aware access helpers. These enforce access cutoffs, not arbitrary model training behavior."""
from pathlib import Path
import pandas as pd


class CData:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self._cache = {}

    @staticmethod
    def _time(value):
        result = pd.Timestamp(value)
        if result.tzinfo is not None:
            raise ValueError('Use the unzoned local clock in the source files, not UTC conversion.')
        return result

    def _read(self, name, dates=()):
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.root / 'processed' / f'{name}.csv',
                                           parse_dates=list(dates), float_precision='round_trip')
        return self._cache[name]

    def history(self, as_of):
        """Measurements at or before as_of; assumes zero reporting latency."""
        cutoff = self._time(as_of)
        table = self._read('actuals_10min', ['source_date', 'sample_time', 'interval_start', 'interval_end', 'observed_available_at_assumed'])
        return table.loc[table.observed_available_at_assumed <= cutoff].copy()

    def day_ahead_features(self, date, problem=2):
        """Features for 00:00 on date, without target-day actual labels.

        Q4 excludes the Q2/3 fixed known price column; a Q4 price forecast must be
        supplied by the model, or a separately justified known-future-price scenario.
        """
        origin = self._time(date)
        if origin != origin.normalize():
            raise ValueError('Daily plan features require a midnight origin.')
        if problem not in (2, 3, 4):
            raise ValueError('problem must be 2, 3 or 4')
        table = self._read('day_ahead_features', ['origin_time', 'interval_start', 'interval_end',
                                                  'latest_history_sample_time', 'earliest_history_sample_time'])
        result = table[table.origin_time == origin].copy()
        if len(result) != 144:
            raise ValueError('Features cover 2025-02-01 through 2025-12-31 only.')
        if problem == 4:
            result = result.drop(columns='price_q2_q3_known_yuan_per_kwh')
        return result

    def pv_forecast(self, as_of, interval_ends):
        """Latest already-issued vintage for each requested future interval endpoint.

        The returned series remains a point-forecast interpolation; associated energy
        is an interval-average approximation. This helper is intended for Q3/Q4-Q3.
        """
        cutoff = self._time(as_of)
        targets = pd.DatetimeIndex([self._time(t) for t in interval_ends])
        if targets.has_duplicates or not (targets > cutoff).all():
            raise ValueError('Require unique target endpoints strictly after as_of.')
        table = self._read('pv_forecast_10min', ['issue_time', 'interval_start', 'interval_end', 'anchor_available_at'])
        result = table[(table.issue_time <= cutoff) & table.interval_end.isin(targets)]
        result = result.sort_values(['interval_end', 'issue_time']).drop_duplicates('interval_end', keep='last')
        result = result.set_index('interval_end').reindex(targets)
        if result.issue_time.isna().any():
            raise ValueError('An already-issued forecast is unavailable for at least one target. No future backfill performed.')
        return result.rename_axis('interval_end').reset_index()


if __name__ == '__main__':
    data = CData()
    origin = pd.Timestamp('2025-03-20 00:00')
    features = data.day_ahead_features(origin, problem=3)
    forecasts = data.pv_forecast(origin, features.interval_end)
    print('Date:', origin.date(), 'history records:', len(data.history(origin)))
    print('Plan feature rows:', len(features), 'forecast rows:', len(forecasts))
    print(forecasts[['issue_time', 'interval_end', 'pv_forecast_kw_interpolated']].head().to_string(index=False))
