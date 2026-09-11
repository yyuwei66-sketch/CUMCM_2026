#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化白天光伏预测误差随预测提前量的变化。

默认读取 Data_preprocessed/processed 下的逐小时光伏预测和实际光伏，
按目标时刻精确对齐，只统计 06:00--18:00 的白天记录。
输出 q3/figures/04_pv_forecast_error_by_horizon.{csv,png,pdf}。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECAST = ROOT / "Data_preprocessed" / "processed" / "pv_forecast_hourly.csv"
DEFAULT_ACTUALS = ROOT / "Data_preprocessed" / "processed" / "actuals_10min.csv"
DEFAULT_OUTDIR = ROOT / "q3" / "figures"
OUTPUT_STEM = "04_pv_forecast_error_by_horizon"

BLUE = "#105A90"
ORANGE = "#F28C00"
RED = "#FF3028"
SKY = "#DCEEF9"
CREAM = "#FCF3DD"
INK = "#333333"


def configure_style() -> None:
    """Use available Chinese fonts, matching the existing Q3 figure style."""
    for directory in [Path.home() / ".local/share/fonts", Path("/System/Library/Fonts"), Path("/Library/Fonts")]:
        if not directory.exists():
            continue
        for path in directory.glob("*"):
            if path.suffix.lower() in {".ttf", ".ttc", ".otf"}:
                try:
                    font_manager.fontManager.addfont(str(path))
                except (OSError, RuntimeError, TypeError):
                    pass

    names = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "SimSun",
        "STSong",
        "Songti SC",
        "Noto Serif CJK SC",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "SimHei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    selected = [name for name in candidates if name in names]
    family = selected + ["DejaVu Serif"] if selected else ["DejaVu Serif"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": family,
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": "#454545",
            "axes.linewidth": 1.0,
            "axes.axisbelow": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.38,
            "grid.color": "#B8C1C8",
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#B5B5B5",
            "legend.framealpha": 0.97,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_aligned_daytime_errors(forecast_path: Path, actuals_path: Path, daytime_start: int, daytime_end: int) -> tuple[pd.DataFrame, float]:
    forecast = pd.read_csv(forecast_path, parse_dates=["issue_time", "target_time"])
    actuals = pd.read_csv(actuals_path, parse_dates=["interval_end"], usecols=["interval_end", "pv_kw"])

    required = {"target_time", "lead_hours", "pv_forecast_kw", "target_in_actual_coverage"}
    missing = required.difference(forecast.columns)
    if missing:
        raise ValueError(f"预测文件缺少列：{sorted(missing)}")

    data = forecast.merge(actuals, left_on="target_time", right_on="interval_end", how="left")
    data = data[data["target_in_actual_coverage"] & data["pv_kw"].notna()].copy()
    data = data[data["target_time"].dt.hour.between(daytime_start, daytime_end)].copy()

    if data.empty:
        raise ValueError("白天口径下没有可比较的预测与实际数据。")

    capacity_kw = float(max(data["pv_kw"].max(), data["pv_forecast_kw"].max()))
    data["error_kw"] = data["pv_forecast_kw"] - data["pv_kw"]
    data["abs_error_kw"] = data["error_kw"].abs()
    data["squared_error_kw"] = data["error_kw"] ** 2
    return data, capacity_kw


def summarize_by_horizon(data: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    summary = (
        data.groupby("lead_hours", as_index=False)
        .agg(
            n=("abs_error_kw", "size"),
            MAE_kw=("abs_error_kw", "mean"),
            RMSE_kw=("squared_error_kw", lambda values: float(np.sqrt(np.mean(values)))),
            Bias_kw=("error_kw", "mean"),
        )
        .sort_values("lead_hours")
    )
    summary["nMAE_capacity_pct"] = 100 * summary["MAE_kw"] / capacity_kw

    expected = set(range(1, 25))
    actual = set(summary["lead_hours"].astype(int))
    missing = sorted(expected.difference(actual))
    if missing:
        raise ValueError(f"缺少预测提前量：{missing}")
    if (summary["n"] <= 0).any():
        raise ValueError("存在样本数为 0 的预测提前量。")
    if (summary["MAE_kw"] < 0).any():
        raise ValueError("MAE 不应为负。")
    return summary


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(CREAM)
    ax.grid(True)
    ax.tick_params(direction="out", length=4, width=0.8)


def plot_summary(summary: pd.DataFrame, outdir: Path, daytime_start: int, daytime_end: int) -> list[Path]:
    configure_style()
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.7, 6.6))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.91)
    style_axes(ax)

    ax.axvspan(1, 6, color=SKY, alpha=0.78, zorder=0)
    ax.axvspan(6, 12, color=CREAM, alpha=0.85, zorder=0)
    ax.axvspan(12, 18, color=SKY, alpha=0.55, zorder=0)
    ax.axvspan(18, 24, color="#F7E5E2", alpha=0.72, zorder=0)

    ax.plot(summary["lead_hours"], summary["MAE_kw"], color=BLUE, lw=2.3, marker="o", ms=5.2, label="平均绝对误差")
    ax.plot(summary["lead_hours"], summary["RMSE_kw"], color=ORANGE, lw=1.8, ls="--", marker="s", ms=4.2, label="均方根误差")

    for x, label in [(6, "6h"), (12, "12h"), (18, "18h"), (24, "24h")]:
        ax.axvline(x, color="#7A8288", ls="--", lw=0.95, zorder=1)
        ax.text(x, ax.get_ylim()[1] * 0.94, label, ha="center", va="top", fontsize=10, color=INK)

    ax.set_xlim(1, 24)
    ax.set_xticks(np.arange(1, 25, 1))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("预测尺度／提前量（小时）")
    ax.set_ylabel("白天平均预测误差（kW）")
    ax.set_title(f"光伏预测误差随预测尺度的变化（白天 {daytime_start:02d}:00--{daytime_end:02d}:00）", pad=12)
    ax.legend(loc="upper left")

    mae_12 = float(summary.loc[summary["lead_hours"].eq(12), "MAE_kw"].iloc[0])
    mae_18 = float(summary.loc[summary["lead_hours"].eq(18), "MAE_kw"].iloc[0])
    mae_24 = float(summary.loc[summary["lead_hours"].eq(24), "MAE_kw"].iloc[0])
    ax.text(
        0.98,
        0.06,
        f"12h MAE：{mae_12:.1f} kW\n18h MAE：{mae_18:.1f} kW\n24h MAE：{mae_24:.1f} kW",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=.45", fc="white", ec="#B5B5B5"),
    )

    paths = []
    for ext in ["png", "pdf"]:
        path = outdir / f"{OUTPUT_STEM}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        paths.append(path)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制白天光伏预测误差随预测提前量的变化。")
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST, help="逐小时光伏预测 CSV 路径。")
    parser.add_argument("--actuals", type=Path, default=DEFAULT_ACTUALS, help="实际光伏 10 分钟 CSV 路径。")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="输出目录。")
    parser.add_argument("--daytime-start", type=int, default=6, help="白天起始小时，默认 6。")
    parser.add_argument("--daytime-end", type=int, default=18, help="白天结束小时，默认 18。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, capacity_kw = load_aligned_daytime_errors(args.forecast, args.actuals, args.daytime_start, args.daytime_end)
    summary = summarize_by_horizon(data, capacity_kw)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / f"{OUTPUT_STEM}.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    figure_paths = plot_summary(summary, args.outdir, args.daytime_start, args.daytime_end)

    print(f"白天样本数：{len(data)}")
    print(f"容量归一化基准：{capacity_kw:.4f} kW")
    print(f"明细表：{csv_path}")
    for path in figure_paths:
        print(f"图像：{path}")


if __name__ == "__main__":
    main()
