"""Generate final Q3 paper figures from submit/q3.py's exact model path."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submit"
OUT = ROOT / "paper" / "figures" / "a3"
TMP = ROOT / ".tmp" / "q3_paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SUBMIT))
import q3


def configure_style() -> None:
    cache = ROOT / ".tmp" / "mplcache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    font = next((x for x in candidates if x in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 9.2,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2, "axes.linewidth": 0.8,
        "lines.linewidth": 1.7, "figure.dpi": 140,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.9)
    ax.grid(axis="x", visible=False)


def save_png(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def run_models():
    dates, load, pv, prices, forecasts = q3.read_inputs(
        SUBMIT / "Data" / "附件2.xlsx",
        SUBMIT / "Data" / "附件1.xlsx",
        SUBMIT / "Data" / "附件3.xlsx",
    )
    pred_load, first = q3.q2_forecast(dates, load, pv)
    strategies = {
        "Baseline 1": "baseline1_fixed",
        "Baseline 2": "baseline2_6h",
        "Baseline 3": "baseline3_6_12h",
        "Baseline 4": "baseline4_6_12h",
        "主模型": "probabilistic",
    }
    # q3.py uses baseline4_6_12_18h; keep the label and exact kind explicit.
    strategies["Baseline 4"] = "baseline4_6_12_18h"
    results = {}
    for label, kind in strategies.items():
        results[label] = q3.run_strategy(
            label, kind, dates, load, pv, prices, forecasts, pred_load, first
        )
    return dates, load, pv, forecasts, results


def figure_strategy_cost_risk(results) -> None:
    rows = []
    for label, (daily, detail, decisions) in results.items():
        x = daily[daily["date"] >= str(q3.EVAL_START)]
        rows.append({
            "strategy": label,
            "planned": x["planned_cost"].sum() / 1e4,
            "adjustment": x["adjustment_cost"].sum() / 1e4,
            "emergency": x["emergency_cost"].sum() / 1e4,
            "total": x["total_cost"].sum() / 1e4,
            "emergency_kwh": x["emergency_kwh"].sum() / 1e3,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TMP / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    pos = np.arange(len(summary))
    colors = ["#B8C9D6", "#7F9DB2", "#A9B7C6"]
    bottom = np.zeros(len(summary))
    for col, color, label in zip(["planned", "adjustment", "emergency"], colors, ["原计划费用", "调整费用", "紧急购电费用"]):
        ax.bar(pos, summary[col], bottom=bottom, width=0.62, color=color, edgecolor="white", linewidth=0.7, label=label)
        bottom += summary[col].to_numpy()
    ax2 = ax.twinx()
    ax2.plot(pos, summary["emergency_kwh"], "o-", color="#8C4A4A", markerfacecolor="white", markeredgewidth=1.2, label="紧急购电量")
    ax.set_xticks(pos, summary["strategy"])
    ax.set_ylabel("全年费用 / 万元")
    ax2.set_ylabel("紧急购电量 / MWh", color="#8C4A4A")
    ax2.tick_params(axis="y", colors="#8C4A4A")
    clean_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#8C4A4A")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=2, loc="upper right")
    for i, value in enumerate(summary["total"]):
        ax.text(i, value + 3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    save_png(fig, "q3_fig01_strategy_cost_risk.png")


def figure_time_decomposition(results) -> None:
    base = results["Baseline 4"][1]
    main = results["主模型"][1]
    a = base[base["date"] >= str(q3.EVAL_START)].copy()
    b = main[main["date"] >= str(q3.EVAL_START)].copy()
    key = ["date", "slot"]
    merged = a.merge(b, on=key, suffixes=("_b4", "_main"))
    merged["period"] = pd.cut(merged["slot"], [0, 36, 72, 108, 144], labels=["00–06", "06–12", "12–18", "18–24"])
    # 正值定义为主模型相对 baseline4 的节省。
    for col in ["planned_cost", "adjustment_cost", "emergency_cost"]:
        merged[f"saving_{col}"] = merged[f"{col}_b4"] - merged[f"{col}_main"]
    table = merged.groupby("period", observed=False)[["saving_planned_cost", "saving_adjustment_cost", "saving_emergency_cost"]].sum()
    table["net_saving"] = table.sum(axis=1)
    table.to_csv(TMP / "time_decomposition.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    pos = np.arange(len(table))
    width = 0.22
    series = [("saving_planned_cost", "计划费节省", "#B8C9D6"), ("saving_adjustment_cost", "调整费节省", "#7F9DB2"), ("saving_emergency_cost", "紧急费用节省", "#8C4A4A")]
    for j, (col, label, color) in enumerate(series):
        ax.bar(pos + (j - 1) * width, table[col] / 1e4, width, color=color, label=label, edgecolor="white", linewidth=0.6)
    ax.plot(pos, table["net_saving"] / 1e4, "o-", color="#1F4E79", linewidth=2, label="净节省")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(pos, [f"{x}时段" for x in table.index])
    ax.set_ylabel("主模型相对 baseline 4 的节省 / 万元")
    clean_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="best")
    label_offsets = [(0, 16), (0, -16), (0, -18), (7, 13)]
    for i, value in enumerate(table["net_saving"] / 1e4):
        dx, dy = label_offsets[i]
        ax.annotate(
            f"{value:.2f}",
            xy=(i, value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center" if dx == 0 else "left",
            va="center",
            fontsize=8,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=6,
        )
    save_png(fig, "q3_fig02_time_decomposition.png")


def figure_forecast_error_update(dates, pv, forecasts, results) -> None:
    eval_idx = np.where(dates >= pd.Timestamp(q3.EVAL_START))[0]
    intervals = [(6, 12), (12, 18), (18, 24)]
    metric_rows = []
    for issue_index, issue_hour in enumerate(q3.ISSUE_HOURS):
        for h0, h1 in intervals:
            start, end = h0 * 6, h1 * 6
            if issue_hour >= h1:
                continue
            pred = forecasts[eval_idx, issue_index, start:end]
            actual = pv[eval_idx, start:end]
            error = pred - actual
            metric_rows.append({
                "发布时刻": f"{issue_hour:02d}:00",
                "目标时段": f"{h0:02d}–{h1:02d}",
                "MAE": np.mean(np.abs(error)),
                "RMSE": np.sqrt(np.mean(error ** 2)),
                "Bias": np.mean(error),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(TMP / "forecast_error_update.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    for target, ax in zip(["18–24", "12–18"], axes):
        sub = metrics[metrics["目标时段"] == target].copy()
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["MAE"], 0.36, color="#7F9DB2", label="MAE")
        ax.bar(x + 0.18, sub["RMSE"], 0.36, color="#B8C9D6", label="RMSE")
        ax.set_xticks(x, sub["发布时刻"])
        ax.set_xlabel(f"目标时段 {target}时")
        ax.set_ylabel("光伏预测误差 / kW")
        ax.legend(frameon=False, loc="upper left")
        clean_axes(ax)
        for i, bias in enumerate(sub["Bias"]):
            ax.text(i, max(sub["MAE"].max(), sub["RMSE"].max()) * 1.03, f"偏差 {bias:.1f}", ha="center", fontsize=7.5)
    save_png(fig, "q3_fig03_forecast_error_by_update.png")


def main() -> None:
    configure_style()
    dates, load, pv, forecasts, results = run_models()
    figure_strategy_cost_risk(results)
    figure_time_decomposition(results)
    figure_forecast_error_update(dates, pv, forecasts, results)
    print(json.dumps({"status": "PASS", "output": str(OUT), "figures": sorted(p.name for p in OUT.glob("q3_fig*.png"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
