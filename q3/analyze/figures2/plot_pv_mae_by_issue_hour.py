#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制四个发布时刻（00:00/06:00/12:00/18:00）的光伏预报 MAE 随预测提前量的变化。

对每一个发布时刻 r 与提前量 h（h=1..24），把该发布版本预报的目标时刻光伏功率
与附件2 的实测光伏功率精确对齐，计算平均绝对误差

    MAE(r, h) = mean_t | P_forecast(r, t+h) - P_actual(t+h) |

只保留实测覆盖率标记为 True 且实测值非空的记录，不做任何插值或外推。
输出 q3/figures2/05_pv_mae_by_issue_hour.{csv,png,pdf}。
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST = ROOT / "Data_preprocessed" / "processed" / "pv_forecast_hourly.csv"
DEFAULT_ACTUALS = ROOT / "Data_preprocessed" / "processed" / "actuals_10min.csv"
DEFAULT_OUTDIR = ROOT / "q3" / "figures2"
OUTPUT_STEM = "05_pv_mae_by_issue_hour"

# 四个发布时刻的固定顺序与配色（保持与 Q3 其他插图一致的蓝-橙-红系）。
ISSUE_HOURS = (0, 6, 12, 18)
SERIES_STYLE = {
    0: {"color": "#105A90", "marker": "o", "label": "00:00 发布预报"},
    6: {"color": "#F28C00", "marker": "s", "label": "06:00 发布预报"},
    12: {"color": "#2E9E5B", "marker": "^", "label": "12:00 发布预报"},
    18: {"color": "#FF3028", "marker": "D", "label": "18:00 发布预报"},
}

SKY = "#DCEEF9"
CREAM = "#FCF3DD"
ROSE = "#F7E5E2"
INK = "#333333"


def configure_style() -> None:
    """沿用 Q3 已有插图的字体与 rcParams 设置。"""
    for directory in [
        Path.home() / ".local/share/fonts",
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]:
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


def load_aligned_errors(forecast_path: Path, actuals_path: Path) -> tuple[pd.DataFrame, float]:
    """把逐小时预报与 10 分钟实测精确对齐，返回逐样本绝对误差与容量基准。"""
    forecast = pd.read_csv(forecast_path, parse_dates=["issue_time", "target_time"])
    actuals = pd.read_csv(
        actuals_path, parse_dates=["interval_end"], usecols=["interval_end", "pv_kw"]
    )

    required = {"issue_time", "target_time", "lead_hours", "pv_forecast_kw", "target_in_actual_coverage"}
    missing = required.difference(forecast.columns)
    if missing:
        raise ValueError(f"预测文件缺少列：{sorted(missing)}")

    data = forecast.merge(actuals, left_on="target_time", right_on="interval_end", how="left")
    data = data[data["target_in_actual_coverage"] & data["pv_kw"].notna()].copy()
    if data.empty:
        raise ValueError("没有可比较的预报与实测数据。")

    capacity_kw = float(max(data["pv_kw"].max(), data["pv_forecast_kw"].max()))
    data["issue_hour"] = data["issue_time"].dt.hour
    data["abs_error_kw"] = (data["pv_forecast_kw"] - data["pv_kw"]).abs()
    return data, capacity_kw


def summarize(data: pd.DataFrame, capacity_kw: float) -> pd.DataFrame:
    """按（发布时刻，提前量）聚合 MAE，并校验完整性。"""
    summary = (
        data.groupby(["issue_hour", "lead_hours"], as_index=False)
        .agg(n=("abs_error_kw", "size"), MAE_kw=("abs_error_kw", "mean"))
        .sort_values(["issue_hour", "lead_hours"])
    )
    summary["nMAE_capacity_pct"] = 100 * summary["MAE_kw"] / capacity_kw

    available = sorted(summary["issue_hour"].unique().tolist())
    missing_hours = sorted(set(ISSUE_HOURS).difference(available))
    if missing_hours:
        raise ValueError(f"缺少发布时刻：{missing_hours}")

    for hour in ISSUE_HOURS:
        block = summary[summary["issue_hour"].eq(hour)]
        missing_leads = sorted(set(range(1, 25)).difference(block["lead_hours"].astype(int)))
        if missing_leads:
            raise ValueError(f"发布时刻 {hour:02d}:00 缺少提前量：{missing_leads}")
        if (block["n"] <= 0).any():
            raise ValueError(f"发布时刻 {hour:02d}:00 存在样本数为 0 的提前量。")
    if (summary["MAE_kw"] < 0).any():
        raise ValueError("MAE 不应为负。")
    return summary


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(CREAM)
    ax.grid(True)
    ax.tick_params(direction="out", length=4, width=0.8)


def plot_summary(summary: pd.DataFrame, outdir: Path) -> list[Path]:
    configure_style()
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.7, 6.6))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.88)
    style_axes(ax)

    # 用四段底色区分四个 6 小时提前量区段，与发布时刻的间隔保持一致。
    for lo, hi, color in [(1, 6, SKY), (6, 12, CREAM), (12, 18, SKY), (18, 24, ROSE)]:
        ax.axvspan(lo, hi, color=color, alpha=0.75 if color == SKY else 0.85, zorder=0)

    for hour in ISSUE_HOURS:
        block = summary[summary["issue_hour"].eq(hour)]
        style = SERIES_STYLE[hour]
        ax.plot(
            block["lead_hours"],
            block["MAE_kw"],
            color=style["color"],
            lw=2.3,
            marker=style["marker"],
            ms=5.2,
            label=style["label"],
            zorder=3,
        )

    for x, label in [(6, "6h"), (12, "12h"), (18, "18h"), (24, "24h")]:
        ax.axvline(x, color="#7A8288", ls="--", lw=0.95, zorder=1)
        ax.text(x, ax.get_ylim()[1] * 0.94, label, ha="center", va="top", fontsize=10, color=INK)

    ax.set_xlim(1, 24)
    ax.set_xticks(np.arange(1, 25, 1))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.set_xlabel("预测提前量（小时）")
    ax.set_ylabel("光伏发电功率预测 MAE（kW）")
    ax.set_title("不同发布时刻的光伏预报误差随预测提前量的变化（2025 全年全天）", pad=12)
    ax.legend(loc="upper left", ncol=2, title="预报发布时刻")

    lines = []
    for hour in ISSUE_HOURS:
        block = summary[summary["issue_hour"].eq(hour)]
        mean_mae = float(block["MAE_kw"].mean())
        lines.append(f"{hour:02d}:00 全天平均 MAE：{mean_mae:6.1f} kW")
    ax.text(
        0.985,
        0.05,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
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
    parser = argparse.ArgumentParser(
        description="绘制四个发布时刻的光伏预报 MAE 随预测提前量的变化。"
    )
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST, help="逐小时光伏预报 CSV 路径。")
    parser.add_argument("--actuals", type=Path, default=DEFAULT_ACTUALS, help="实测 10 分钟光伏 CSV 路径。")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="输出目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, capacity_kw = load_aligned_errors(args.forecast, args.actuals)
    summary = summarize(data, capacity_kw)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / f"{OUTPUT_STEM}.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    figure_paths = plot_summary(summary, args.outdir)

    print(f"对齐样本数：{len(data)}")
    print(f"容量归一化基准：{capacity_kw:.4f} kW")
    print(f"明细表：{csv_path}")
    for path in figure_paths:
        print(f"图像：{path}")


if __name__ == "__main__":
    main()
