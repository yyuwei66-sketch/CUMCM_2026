from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import p2

SCRIPT_DIR = Path(__file__).resolve().parent
M_GRID = [14, 28, 42, 60]

def crps_ensemble(samples, actual):
    z = np.sort(np.asarray(samples, float), axis=0)
    y = np.asarray(actual, float)
    S = z.shape[0]
    term1 = np.mean(np.abs(z - y), axis=0)
    weights = (2 * np.arange(1, S + 1) - 1 - S)[:, None]
    term2 = np.sum(weights * z, axis=0) / (S * S)
    return term1 - term2

def conformal_quantile(scores, coverage):
    scores = np.asarray(scores, float)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))

def build_base_quantiles(scenarios):
    return {
        d: np.quantile(arr, [0.05, 0.10, 0.90, 0.95], axis=0)
        for d, arr in scenarios.items()
    }

def evaluate_period(days, dates, load, pv, scenarios, base_q, M=None):
    rows = []
    for d in days:
        y = load[d] - pv[d]
        q05, q10, q90, q95 = base_q[d]
        lo80, hi80 = q10.copy(), q90.copy()
        lo90, hi90 = q05.copy(), q95.copy()
        ex80 = ex90 = 0.0

        if M is not None:
            past = [j for j in range(d - M, d) if j in base_q]
            s80, s90 = [], []
            for j in past:
                yy = load[j] - pv[j]
                a05, a10, a90, a95 = base_q[j]
                s80.append(np.maximum.reduce([
                    a10 - yy, yy - a90, np.zeros_like(yy)
                ]))
                s90.append(np.maximum.reduce([
                    a05 - yy, yy - a95, np.zeros_like(yy)
                ]))
            s80 = np.concatenate(s80)
            s90 = np.concatenate(s90)
            ex80 = conformal_quantile(s80, 0.80)
            ex90 = conformal_quantile(s90, 0.90)
            lo80 -= ex80
            hi80 += ex80
            lo90 -= ex90
            hi90 += ex90

        rows.append({
            "date": str(dates[d].date()),
            "crps_kw": float(crps_ensemble(scenarios[d], y).mean()),
            "cover80": float(np.mean((y >= lo80) & (y <= hi80))),
            "cover90": float(np.mean((y >= lo90) & (y <= hi90))),
            "width80_kw": float(np.mean(hi80 - lo80)),
            "width90_kw": float(np.mean(hi90 - lo90)),
            "expand80_kw": ex80,
            "expand90_kw": ex90,
        })

    daily = pd.DataFrame(rows)
    summary = {
        "crps_kw": float(daily["crps_kw"].mean()),
        "cover80": float(daily["cover80"].mean()),
        "cover90": float(daily["cover90"].mean()),
        "width80_kw": float(daily["width80_kw"].mean()),
        "width90_kw": float(daily["width90_kw"].mean()),
        "expand80_kw": float(daily["expand80_kw"].mean()),
        "expand90_kw": float(daily["expand90_kw"].mean()),
    }
    return summary, daily

def run_calibration(actual_path, price_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates, load, pv, _ = p2.load_inputs(actual_path, price_path)
    pred_l, pred_p, scenarios = p2.forecast_scenarios(dates, load, pv)
    base_q = build_base_quantiles(scenarios)

    dev = np.where(
        (dates >= pd.Timestamp(p2.DEV_START))
        & (dates <= pd.Timestamp(p2.DEV_END))
    )[0]
    test = np.where(
        (dates >= pd.Timestamp("2025-09-01"))
        & (dates <= pd.Timestamp("2025-12-31"))
    )[0]

    rows = []
    base, _ = evaluate_period(
        dev, dates, load, pv, scenarios, base_q, M=None
    )
    rows.append({
        "period": "development_May_Aug",
        "method": "base",
        "score_window": 0,
        "calibration_error": abs(base["cover80"] - 0.80)
                             + abs(base["cover90"] - 0.90),
        **base,
    })

    for M in M_GRID:
        m, _ = evaluate_period(
            dev, dates, load, pv, scenarios, base_q, M=M
        )
        rows.append({
            "period": "development_May_Aug",
            "method": "rolling_conformal",
            "score_window": M,
            "calibration_error": abs(m["cover80"] - 0.80)
                                 + abs(m["cover90"] - 0.90),
            **m,
        })

    table = pd.DataFrame(rows)
    cand = table[table["method"] == "rolling_conformal"].copy()
    cand["tie_width"] = cand["width80_kw"] + cand["width90_kw"]
    chosen = cand.sort_values(
        ["calibration_error", "tie_width"]
    ).iloc[0]
    M_star = int(chosen["score_window"])

    # Final test metrics for the chosen M.
    test_base, _ = evaluate_period(
        test, dates, load, pv, scenarios, base_q, M=None
    )
    test_cal, test_daily = evaluate_period(
        test, dates, load, pv, scenarios, base_q, M=M_star
    )

    table.to_csv(
        outdir / "Q2_final_probability_calibration_selection.csv",
        index=False, encoding="utf-8-sig",
    )
    test_daily.to_csv(
        outdir / "Q2_final_independent_test_daily_probability.csv",
        index=False, encoding="utf-8-sig",
    )

    test_summary = pd.DataFrame([
        {
            "period": "independent_test_Sep_Dec",
            "probability": "base",
            "M": 0,
            **test_base,
        },
        {
            "period": "independent_test_Sep_Dec",
            "probability": "calibrated",
            "M": M_star,
            **test_cal,
        },
    ])
    test_summary.to_csv(
        outdir / "Q2_probability_test_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    print(table.to_string(index=False))
    print(f"\nChosen M = {M_star}")
    print("\nSep-Dec:")
    print(test_summary.to_string(index=False))

    if M_star != 14:
        raise AssertionError(f"Expected M=14, got M={M_star}")

def main():
    default_actual, default_price, _ = p2.default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--actual", type=Path, default=default_actual)
    ap.add_argument("--price", type=Path, default=default_price)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "calibration_results",
    )
    args = ap.parse_args()
    run_calibration(args.actual, args.price, args.outdir)

if __name__ == "__main__":
    main()
