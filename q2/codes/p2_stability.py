from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from p2_joint_search import (
    load_actual, pca_basis, calendar_feature, history_feature,
    ridge_predict_all_alpha, VAL_START, VAL_END
)

SCRIPT_DIR = Path(__file__).resolve().parent

def single_series_errors(series, dates, W, K, h, cal, alpha, val_idx):
    out = []
    for d in val_idx:
        h0 = d - W
        mu, phi = pca_basis(series[h0:d], int(K))
        scores = (series[h0:d] - mu) @ phi.T
        train_days = np.arange(h0 + 7, d)
        local_train = train_days - h0
        local_target = d - h0

        X = np.vstack([
            np.r_[
                calendar_feature(dates[j], cal),
                history_feature(scores, int(local_train[k]), h),
            ]
            for k, j in enumerate(train_days)
        ])
        y = np.vstack([scores[jj] for jj in local_train])
        x0 = np.r_[
            calendar_feature(dates[d], cal),
            history_feature(scores, int(local_target), h),
        ]
        shat = ridge_predict_all_alpha(X, y, x0, [float(alpha)])[float(alpha)]
        pred = np.maximum(mu + shat @ phi, 0)
        out.append(series[d] - pred)
    return np.asarray(out)

def run_stability(actual_path, top10_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates, load, pv = load_actual(actual_path)
    val_idx = np.where((dates >= VAL_START) & (dates <= VAL_END))[0]
    top = pd.read_csv(top10_path)

    rows = []
    for r in top.itertuples(index=False):
        eL = single_series_errors(
            load, dates, int(r.W), int(r.KL), r.hL, r.calL,
            float(r.alphaL), val_idx,
        )
        eP = single_series_errors(
            pv, dates, int(r.W), int(r.KP), r.hP, r.calP,
            float(r.alphaP), val_idx,
        )
        eN = eL - eP

        row = r._asdict()
        row["overall_rmse"] = float(np.sqrt(np.mean(eN**2)))

        monthly_rmse = []
        monthly_mae = []
        for month, name in [(5, "May"), (6, "Jun"), (7, "Jul"), (8, "Aug")]:
            mask = np.array([dates[d].month == month for d in val_idx])
            x = eN[mask]
            rmse = float(np.sqrt(np.mean(x**2)))
            mae = float(np.mean(np.abs(x)))
            row[f"{name}_rmse"] = rmse
            row[f"{name}_mae"] = mae
            monthly_rmse.append(rmse)
            monthly_mae.append(mae)

        row["monthly_mean_rmse"] = float(np.mean(monthly_rmse))
        row["monthly_sd_rmse"] = float(np.std(monthly_rmse, ddof=0))
        row["worst_month_rmse"] = float(np.max(monthly_rmse))
        row["best_month_rmse"] = float(np.min(monthly_rmse))
        row["monthly_cv_pct"] = (
            row["monthly_sd_rmse"] / row["monthly_mean_rmse"] * 100.0
        )
        rows.append(row)
        print(
            f"W={r.W} rank={r.rank} "
            f"overall={row['overall_rmse']:.4f} "
            f"monthly_mean={row['monthly_mean_rmse']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values(["W", "rank"])
    df.to_csv(
        outdir / "top10_per_W_monthly_stability.csv",
        index=False, encoding="utf-8-sig",
    )

    # Per-W winner by monthly mean RMSE
    idx = df.groupby("W")["monthly_mean_rmse"].idxmin()
    byw = df.loc[idx].sort_values("W")
    byw.to_csv(
        outdir / "best_monthly_stable_by_W.csv",
        index=False, encoding="utf-8-sig",
    )

    winners = []
    for criterion, col in [
        ("pooled_overall", "overall_rmse"),
        ("mean_monthly", "monthly_mean_rmse"),
        ("minimax_worst_month", "worst_month_rmse"),
    ]:
        x = df.loc[df[col].idxmin()].to_dict()
        x["criterion"] = criterion
        winners.append(x)
    pd.DataFrame(winners).to_csv(
        outdir / "stability_winners.csv",
        index=False, encoding="utf-8-sig",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--actual",
        type=Path,
        default=SCRIPT_DIR / "input" / "q2_actual_load_pv.csv",
    )
    ap.add_argument(
        "--top10",
        type=Path,
        default=SCRIPT_DIR / "joint_search_results" / "top10_per_W.csv",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "stability_results",
    )
    args = ap.parse_args()

    if not args.actual.exists():
        raise FileNotFoundError(args.actual)
    if not args.top10.exists():
        raise FileNotFoundError(
            f"{args.top10}\n请先运行 p2_joint_search.py"
        )
    run_stability(args.actual, args.top10, args.outdir)

if __name__ == "__main__":
    main()
