from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# Exact candidate space used for the final joint search
# ============================================================
W_GRID = [30, 60, 90, 120]
KL_GRID = [1, 2, 3, 4, 5]
KP_GRID = [1, 2, 3, 4]
H_GRID = ["H0_lag1", "H1_lag1_lag7", "H2_lag1_lag7_mean7", "H3_lag1_lag7_mean7_trend7"]
CAL_L_GRID = ["fri_sat", "weekday7", "fri_sat+annual1", "fri_sat+annual2"]
CAL_P_GRID = ["annual1", "annual2", "weekday7+annual1", "weekday7+annual2"]
ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

VAL_START = pd.Timestamp("2025-05-01")
VAL_END = pd.Timestamp("2025-08-31")

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)

def load_actual(path):
    x = pd.read_csv(path, encoding="utf-8-sig")
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["date", "slot"]).reset_index(drop=True)
    require(len(x) == 365 * 144, "actual data must be 365×144")
    dates = pd.DatetimeIndex(x["date"].drop_duplicates())
    load = x["load_kw"].to_numpy(float).reshape(-1, 144)
    pv = x["pv_kw"].to_numpy(float).reshape(-1, 144)
    return dates, load, pv

def pca_basis(curves, k):
    mu = curves.mean(axis=0)
    _, _, vt = np.linalg.svd(curves - mu, full_matrices=False)
    return mu, vt[:k]

def weekday7(date):
    x = np.zeros(7)
    x[date.weekday()] = 1.0
    return x

def fri_sat(date):
    z = int(date.weekday() in (4, 5))
    x = np.zeros(2)
    x[z] = 1.0
    return x

def annual1(date):
    a = 2 * np.pi * date.dayofyear / 365.25
    return np.array([np.sin(a), np.cos(a)])

def annual2(date):
    a = 2 * np.pi * date.dayofyear / 365.25
    return np.array([np.sin(a), np.cos(a), np.sin(2*a), np.cos(2*a)])

def calendar_feature(date, name):
    if name == "fri_sat":
        return fri_sat(date)
    if name == "weekday7":
        return weekday7(date)
    if name == "annual1":
        return annual1(date)
    if name == "annual2":
        return annual2(date)
    if name == "fri_sat+annual1":
        return np.r_[fri_sat(date), annual1(date)]
    if name == "fri_sat+annual2":
        return np.r_[fri_sat(date), annual2(date)]
    if name == "weekday7+annual1":
        return np.r_[weekday7(date), annual1(date)]
    if name == "weekday7+annual2":
        return np.r_[weekday7(date), annual2(date)]
    raise ValueError(name)

def history_feature(scores, local_j, name):
    lag1 = scores[local_j - 1]
    if name == "H0_lag1":
        return np.atleast_1d(lag1)

    lag7 = scores[local_j - 7]
    if name == "H1_lag1_lag7":
        return np.r_[lag1, lag7]

    mean7 = scores[local_j - 7:local_j].mean(axis=0)
    if name == "H2_lag1_lag7_mean7":
        return np.r_[lag1, lag7, mean7]

    if name == "H3_lag1_lag7_mean7_trend7":
        trend7 = lag1 - mean7
        return np.r_[lag1, lag7, mean7, trend7]

    raise ValueError(name)

def ridge_predict_all_alpha(X, Y, x0, alphas):
    """
    Equivalent to StandardScaler() + Ridge(alpha, fit_intercept=True)
    for small dense design matrices.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    x0 = np.asarray(x0, float)

    mu_x = X.mean(axis=0)
    sd_x = X.std(axis=0, ddof=0)
    sd_x[sd_x < 1e-12] = 1.0
    Xs = (X - mu_x) / sd_x
    x0s = (x0 - mu_x) / sd_x

    mu_y = Y.mean(axis=0)
    Yc = Y - mu_y

    XtX = Xs.T @ Xs
    XtY = Xs.T @ Yc
    I = np.eye(X.shape[1])

    preds = {}
    for alpha in alphas:
        beta = np.linalg.solve(XtX + float(alpha) * I, XtY)
        preds[float(alpha)] = mu_y + x0s @ beta
    return preds

def candidate_table(kind, smoke=False):
    if kind == "load":
        Ks = KL_GRID if not smoke else [3, 4]
        Hs = H_GRID if not smoke else ["H0_lag1", "H2_lag1_lag7_mean7"]
        Cals = CAL_L_GRID if not smoke else ["weekday7", "fri_sat"]
    else:
        Ks = KP_GRID if not smoke else [2]
        Hs = H_GRID if not smoke else ["H2_lag1_lag7_mean7", "H0_lag1"]
        Cals = CAL_P_GRID if not smoke else ["annual1", "annual2"]

    alphas = ALPHA_GRID if not smoke else [0.3, 3.0, 10.0]
    rows = []
    cid = 0
    for K in Ks:
        for h in Hs:
            for cal in Cals:
                for alpha in alphas:
                    rows.append({
                        "candidate_id": cid,
                        "K": K,
                        "history": h,
                        "calendar": cal,
                        "alpha": float(alpha),
                    })
                    cid += 1
    return pd.DataFrame(rows)

def evaluate_series_candidates(series, dates, W, kind, val_idx, smoke=False):
    cands = candidate_table(kind, smoke=smoke)
    N = len(val_idx) * 144
    errors = np.empty((len(cands), N), dtype=np.float64)

    # Row lookup for fast filling.
    row_lookup = {
        (int(r.K), r.history, r.calendar, float(r.alpha)): int(r.candidate_id)
        for r in cands.itertuples(index=False)
    }

    Ks = sorted(cands["K"].unique())
    Hs = sorted(cands["history"].unique())
    Cals = sorted(cands["calendar"].unique())
    Alphas = sorted(cands["alpha"].unique())

    for di, d in enumerate(val_idx):
        require(d >= W, f"validation start is too early for W={W}")
        h0 = d - W

        for K in Ks:
            mu, phi = pca_basis(series[h0:d], int(K))
            scores = (series[h0:d] - mu) @ phi.T
            train_days = np.arange(h0 + 7, d)
            local_train = train_days - h0
            local_target = d - h0  # equals W

            y = np.vstack([scores[jj] for jj in local_train])

            for h in Hs:
                hist_train = [
                    history_feature(scores, int(jj), h)
                    for jj in local_train
                ]
                hist_target = history_feature(scores, int(local_target), h)

                for cal in Cals:
                    X = np.vstack([
                        np.r_[calendar_feature(dates[j], cal), hist_train[k]]
                        for k, j in enumerate(train_days)
                    ])
                    x0 = np.r_[calendar_feature(dates[d], cal), hist_target]

                    pred_scores = ridge_predict_all_alpha(
                        X, y, x0, Alphas
                    )

                    for alpha, shat in pred_scores.items():
                        pred_curve = np.maximum(mu + shat @ phi, 0)
                        err = series[d] - pred_curve
                        row = row_lookup[(int(K), h, cal, float(alpha))]
                        a = di * 144
                        errors[row, a:a+144] = err

    return cands, errors

def pairwise_net_rmse(load_err, pv_err):
    N = load_err.shape[1]
    l2 = np.sum(load_err * load_err, axis=1)[:, None]
    p2 = np.sum(pv_err * pv_err, axis=1)[None, :]
    cross = load_err @ pv_err.T
    mse = np.maximum((l2 + p2 - 2.0 * cross) / N, 0)
    return np.sqrt(mse)

def meta_row(W, lc, pc, rmse):
    return {
        "W": int(W),
        "load_id": int(lc["candidate_id"]),
        "pv_id": int(pc["candidate_id"]),
        "val_net_rmse": float(rmse),
        "KL": int(lc["K"]),
        "KP": int(pc["K"]),
        "hL": lc["history"],
        "hP": pc["history"],
        "calL": lc["calendar"],
        "calP": pc["calendar"],
        "alphaL": float(lc["alpha"]),
        "alphaP": float(pc["alpha"]),
    }

def update_profile(store, param, value, row):
    key = (param, str(value))
    if key not in store or row["val_net_rmse"] < store[key]["val_net_rmse"]:
        store[key] = row.copy()
        store[key]["parameter"] = param
        store[key]["fixed_value"] = value

def run_search(actual_path, outdir, smoke=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates, load, pv = load_actual(actual_path)
    val_idx = np.where((dates >= VAL_START) & (dates <= VAL_END))[0]

    W_grid = W_GRID if not smoke else [60]
    profile = {}
    best_by_W = []
    global_top = []
    top10_by_W = []

    t0 = time.perf_counter()

    for W in W_grid:
        print(f"[W={W}] evaluating Load candidates...")
        lmeta, lerr = evaluate_series_candidates(
            load, dates, W, "load", val_idx, smoke=smoke
        )
        print(f"[W={W}] evaluating PV candidates...")
        pmeta, perr = evaluate_series_candidates(
            pv, dates, W, "pv", val_idx, smoke=smoke
        )

        rmse = pairwise_net_rmse(lerr, perr)

        # best by W
        i, j = np.unravel_index(np.argmin(rmse), rmse.shape)
        best = meta_row(W, lmeta.iloc[i], pmeta.iloc[j], rmse[i, j])
        best_by_W.append(best)

        # top 100 per W (or all if smoke)
        k = min(100, rmse.size)
        flat = rmse.ravel()
        ids = np.argpartition(flat, k-1)[:k]
        ids = ids[np.argsort(flat[ids])]
        wrows = []
        for flat_id in ids:
            i, j = np.unravel_index(flat_id, rmse.shape)
            row = meta_row(W, lmeta.iloc[i], pmeta.iloc[j], rmse[i, j])
            wrows.append(row)
            global_top.append(row)
        top10_by_W.extend(wrows[:10])

        # Profile W
        update_profile(profile, "W", W, best)

        # Profiles over Load-side parameters
        for param, col in [
            ("KL", "K"), ("hL", "history"),
            ("calL", "calendar"), ("alphaL", "alpha"),
        ]:
            for value in lmeta[col].unique():
                li = np.where(lmeta[col].to_numpy() == value)[0]
                sub = rmse[li, :]
                ii, jj = np.unravel_index(np.argmin(sub), sub.shape)
                real_i = li[ii]
                row = meta_row(
                    W, lmeta.iloc[real_i], pmeta.iloc[jj], sub[ii, jj]
                )
                update_profile(profile, param, value, row)

        # Profiles over PV-side parameters
        for param, col in [
            ("KP", "K"), ("hP", "history"),
            ("calP", "calendar"), ("alphaP", "alpha"),
        ]:
            for value in pmeta[col].unique():
                pj = np.where(pmeta[col].to_numpy() == value)[0]
                sub = rmse[:, pj]
                ii, jj = np.unravel_index(np.argmin(sub), sub.shape)
                real_j = pj[jj]
                row = meta_row(
                    W, lmeta.iloc[ii], pmeta.iloc[real_j], sub[ii, jj]
                )
                update_profile(profile, param, value, row)

        # Keep compact candidate metadata for audit
        lmeta.assign(W=W).to_csv(
            outdir / f"load_candidates_W{W}.csv",
            index=False, encoding="utf-8-sig",
        )
        pmeta.assign(W=W).to_csv(
            outdir / f"pv_candidates_W{W}.csv",
            index=False, encoding="utf-8-sig",
        )

        print(
            f"[W={W}] best RMSE={best['val_net_rmse']:.6f} "
            f"({time.perf_counter()-t0:.1f}s)"
        )

    best_by_W_df = pd.DataFrame(best_by_W).sort_values("W")
    best_by_W_df.to_csv(
        outdir / "best_by_W_profile.csv",
        index=False, encoding="utf-8-sig",
    )

    top_df = pd.DataFrame(global_top).sort_values("val_net_rmse").head(100)
    top_df.to_csv(
        outdir / "top100_joint_configs.csv",
        index=False, encoding="utf-8-sig",
    )

    top10_df = pd.DataFrame(top10_by_W)
    top10_df["rank"] = top10_df.groupby("W")["val_net_rmse"].rank(
        method="first"
    ).astype(int)
    top10_df = top10_df.sort_values(["W", "rank"])
    top10_df.to_csv(
        outdir / "top10_per_W.csv",
        index=False, encoding="utf-8-sig",
    )

    prof_rows = list(profile.values())
    prof_df = pd.DataFrame(prof_rows)
    global_best = top_df.iloc[0]
    prof_df["increase_pct"] = (
        (prof_df["val_net_rmse"] / float(global_best["val_net_rmse"]) - 1.0)
        * 100.0
    )
    # Match the paper-friendly schema used by the final analysis.
    prof_df = prof_df.rename(columns={"val_net_rmse": "rmse"})
    cols = [
        "parameter", "fixed_value", "rmse", "increase_pct",
        "W", "KL", "KP", "hL", "hP", "calL", "calP",
        "alphaL", "alphaP",
    ]
    prof_df[cols].to_csv(
        outdir / "parameter_profile_results.csv",
        index=False, encoding="utf-8-sig",
    )

    pd.DataFrame([global_best]).to_csv(
        outdir / "global_best_validation.csv",
        index=False, encoding="utf-8-sig",
    )

    manifest = {
        "validation_period": [str(VAL_START.date()), str(VAL_END.date())],
        "W": W_grid,
        "KL": KL_GRID if not smoke else sorted(candidate_table("load", True)["K"].unique().tolist()),
        "KP": KP_GRID if not smoke else sorted(candidate_table("pv", True)["K"].unique().tolist()),
        "history": H_GRID if not smoke else "smoke subset",
        "calL": CAL_L_GRID if not smoke else "smoke subset",
        "calP": CAL_P_GRID if not smoke else "smoke subset",
        "alpha": ALPHA_GRID if not smoke else "smoke subset",
        "search_type": "joint search with per-W re-optimization",
        "objective": "pooled validation Net Load RMSE",
        "smoke": bool(smoke),
    }
    (outdir / "search_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nGlobal best:")
    print(pd.DataFrame([global_best]).to_string(index=False))
    return global_best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--actual",
        type=Path,
        default=SCRIPT_DIR / "input" / "q2_actual_load_pv.csv",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "joint_search_results",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small subset only, for environment/code verification.",
    )
    args = ap.parse_args()
    if not args.actual.exists():
        raise FileNotFoundError(args.actual)
    run_search(args.actual, args.outdir, smoke=args.smoke)

if __name__ == "__main__":
    main()
