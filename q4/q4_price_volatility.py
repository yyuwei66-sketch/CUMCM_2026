#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四问：实时电价波动可视化。

输入：Data_preprocessed/processed/actuals_10min.csv（附件 4 实际电价，10 分钟粒度，2025 全年）
输出：q4/figures/
    01_24小时电价箱线图.png       —— 24 小时各整点的电价分布箱线图
    02_每日同一时刻电价年内变化.png —— 8 个时刻（4 小时一间隔）电价随日期变化，越晚配色越深
    02_每日同一时刻电价年内变化_data.csv —— 绘图所用宽表数据

画图时间口径说明：
  原始采样时间 0:00+1 表示次日 00:00，归入来源日的第 144 个 10 分钟区间。
  绘图时把「次日 00:00」记为 24 时点以保持日曲线完整，共 145 个时点；
  但在按小时聚合（箱线图、以及时刻序列）时，24:00 与同日 00:00 归并为同一小时，
  即每小时 6 个 10 分钟样本、每天 24 小时 × 6 = 144 个样本。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV = PROJECT_ROOT / "Data_preprocessed" / "processed" / "actuals_10min.csv"
FIG_DIR = SCRIPT_DIR / "figures"

PRICE_COL = "price_actual_yuan_per_kwh"
TIME_COL = "sample_time"

# 4 小时一间隔的 8 个时刻
DRAW_HOURS = [0, 4, 8, 12, 16, 20]
# 用户明确要求 00:00,4:00,...,20:00；24:00 与 00:00 同属一天但分属两端，
# 若需要把日末 24:00 也单列，可把 24 加入下表。默认不含，避免与 00:00 重复。


def configure_style() -> None:
    """中文字体 + 论文级全局样式。"""
    cache_dir = PROJECT_ROOT / ".tmp" / "mplcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    print(f"[font] 使用中文字体: {chosen}")

    plt.rcParams.update(
        {
            "font.sans-serif": [chosen, "DejaVu Sans"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.edgecolor": "#4d4d4d",
            "axes.linewidth": 0.9,
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_price() -> pd.DataFrame:
    """读取电价序列并构造绘图所需时间字段。"""
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV, usecols=["source_date", TIME_COL, PRICE_COL])
    df["source_date"] = pd.to_datetime(df["source_date"])
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.dropna(subset=[PRICE_COL]).copy()

    # 10 分钟粒度的时间序列索引（用于按日透视、按小时聚合）
    df["hour"] = df[TIME_COL].dt.hour
    df["minute"] = df[TIME_COL].dt.minute
    # 时刻标签 hour:minute（0:00, 0:10, ..., 23:50）
    df["clock"] = df["hour"] * 60 + df["minute"]
    # 位于来源日内的第几个 10 分钟槽（1..144）
    df["slot_index"] = (df["clock"] - 10) // 10 + 1
    df.loc[df["clock"] == 0, "slot_index"] = 144  # 次日 00:00 归为当日末槽

    df["month"] = df["source_date"].dt.month
    df["day_of_year"] = df["source_date"].dt.dayofyear

    print(
        f"[data] {len(df)} 条 10 分钟电价记录，"
        f"覆盖 {df['source_date'].min():%Y-%m-%d} ~ {df['source_date'].max():%Y-%m-%d}，"
        f"共 {df['source_date'].dt.date.nunique()} 天"
    )
    print(
        f"[data] 电价范围 {df[PRICE_COL].min():.4f} ~ {df[PRICE_COL].max():.4f} 元/kWh，"
        f"均值 {df[PRICE_COL].mean():.4f}，标准差 {df[PRICE_COL].std():.4f}"
    )
    return df


def figure1_boxplot(df: pd.DataFrame) -> Path:
    """图 1：24 小时电价的箱线图（每小时 6 个 10 分钟样本，全年汇总）。"""
    groups = [df.loc[df["hour"] == h, PRICE_COL].values for h in range(24)]

    # 图例放在绘图区之外的上方，避免遮挡箱体与标注
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    fig.subplots_adjust(top=0.86)

    bp = ax.boxplot(
        groups,
        positions=np.arange(24),
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        medianprops=dict(color="#B22222", linewidth=1.9),
        meanprops=dict(marker="D", markerfacecolor="#FFFFFF",
                       markeredgecolor="#1F4E79", markersize=4.4),
        flierprops=dict(marker="o", markersize=2.2, markerfacecolor="none",
                        markeredgecolor="#8C8C8C", markeredgewidth=0.55, alpha=0.55),
        boxprops=dict(linewidth=0.85, edgecolor="#2F2F2F"),
        whiskerprops=dict(linewidth=0.85, color="#2F2F2F"),
        capprops=dict(linewidth=0.85, color="#2F2F2F"),
        showmeans=True,
    )

    # 按电价水平着色：低谷偏蓝、高峰偏暖
    medians = np.array([np.median(g) for g in groups])
    norm = (medians - medians.min()) / (medians.max() - medians.min())
    cmap = plt.get_cmap("RdYlBu_r")
    for patch, t in zip(bp["boxes"], norm):
        patch.set_facecolor(cmap(0.15 + 0.7 * t))
        patch.set_alpha(0.88)

    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
    ax.set_xlabel("时刻（小时）", labelpad=8)
    ax.set_ylabel("实时电价（元/kWh）", labelpad=8)
    ax.set_title("图 1  24 小时实时电价箱线图（2025 年全年，每小时 6 个 10 分钟采样点）",
                 pad=34)

    overall_mean = df[PRICE_COL].mean()
    ax.axhline(overall_mean, color="#1F4E79", linestyle=":", linewidth=1.3,
               label=f"全年均价 {overall_mean:.4f} 元/kWh")

    handles = [
        plt.Line2D([], [], color="#B22222", linewidth=1.9, label="中位数"),
        plt.Line2D([], [], marker="D", color="none", markerfacecolor="#FFFFFF",
                   markeredgecolor="#1F4E79", markersize=4.4, label="均值"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="#8C8C8C", markersize=3.4, label="异常值（1.5×IQR）"),
        plt.Line2D([], [], color="#1F4E79", linestyle=":", linewidth=1.3,
                   label=f"全年均价 {overall_mean:.4f} 元/kWh"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    ax.set_xlim(-0.7, 23.7)
    # 上方留白给峰标注与图例，下方留白给谷标注
    y_lo = min(g.min() for g in groups)
    y_hi = max(g.max() for g in groups)
    pad_lo = 0.30 * (y_hi - y_lo)
    pad_hi = 0.34 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad_lo, y_hi + pad_hi)

    # 标注峰谷：文字锚点落在箱体之外的空白区，箭头指向中位数，避免遮挡数据
    peak_h, valley_h = int(np.argmax(medians)), int(np.argmin(medians))
    ax.annotate(
        f"电价高峰 {peak_h:02d}:00\n中位数 {medians[peak_h]:.4f} 元/kWh",
        xy=(peak_h, medians[peak_h]),
        xytext=(peak_h + 0.4, y_hi + pad_hi * 0.52),
        ha="center", va="center", fontsize=9.5, color="#8B0000",
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9,
                        connectionstyle="arc3,rad=0.12"),
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#D9A0A0", alpha=0.95),
        zorder=30,
    )
    ax.annotate(
        f"电价低谷 {valley_h:02d}:00\n中位数 {medians[valley_h]:.4f} 元/kWh",
        xy=(valley_h, medians[valley_h]),
        xytext=(valley_h - 1.6, y_lo - pad_lo * 0.60),
        ha="center", va="center", fontsize=9.5, color="#104E8B",
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9,
                        connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#A0B8D9", alpha=0.95),
        zorder=30,
    )

    out = FIG_DIR / "01_24小时电价箱线图.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] 已保存 {out}")

    # 文字摘要
    stats = pd.DataFrame(
        {
            "hour": range(24),
            "mean": [g.mean() for g in groups],
            "median": medians,
            "std": [g.std(ddof=1) for g in groups],
            "q1": [np.percentile(g, 25) for g in groups],
            "q3": [np.percentile(g, 75) for g in groups],
            "min": [g.min() for g in groups],
            "max": [g.max() for g in groups],
        }
    )
    stats.to_csv(FIG_DIR / "01_24小时电价箱线图_stats.csv", index=False)
    print("[stats] 峰谷时刻：")
    print(
        stats.sort_values("median", ascending=False).head(4)[["hour", "median", "mean", "std"]]
        .to_string(index=False)
    )
    print(
        stats.sort_values("median").head(4)[["hour", "median", "mean", "std"]]
        .to_string(index=False)
    )
    return out


def _deep_colors(n: int) -> list:
    """按「时刻越晚配色越深」生成 n 个颜色。

    采用多色相渐变而非单一蓝色：色相沿 蓝 -> 青 -> 绿 -> 橄榄 -> 橙 -> 酒红 过渡，
    同时亮度严格单调递减（0.62 -> 0.16），因此既满足「越晚越深」，
    又保证相邻时刻色相差异明显、便于辨认。
    """
    palette = ["#40B2E7", "#259CAC", "#268062", "#6F560D", "#88320D", "#771124"]
    if n <= len(palette):
        return palette[:n]
    cmap = LinearSegmentedColormap.from_list("time_of_day", palette, N=n)
    return [cmap(i / (n - 1)) for i in range(n)]


def figure2_intra_year(df: pd.DataFrame) -> Path:
    """图 2：每日同一时刻电价在一年中的变化（6 个时刻，4 小时一间隔，越晚越深）。"""
    hours = DRAW_HOURS
    colors = _deep_colors(len(hours))

    # 按 (日期, 小时) 求均值：每小时 6 个 10 分钟采样
    hourly = (
        df.groupby(["source_date", "hour"], as_index=False)[PRICE_COL].mean()
        .rename(columns={PRICE_COL: "price_mean"})
    )
    wide = hourly.pivot(index="source_date", columns="hour", values="price_mean")
    wide = wide.reindex(columns=hours)

    wide.to_csv(FIG_DIR / "02_每日同一时刻电价年内变化_data.csv",
                encoding="utf-8-sig", float_format="%.6f")

    # 图例移到绘图区之外的上方，横向排列，既不遮挡曲线也不压住标题
    fig, ax = plt.subplots(figsize=(14.6, 7.4))
    fig.subplots_adjust(top=0.855)

    x = wide.index
    for i, h in enumerate(hours):
        ax.plot(
            x,
            wide[h].values,
            color=colors[i],
            linewidth=1.55,
            alpha=0.95,
            zorder=2 + i,
            label=f"{h:02d}:00",
        )

    # 时间轴：按月份打刻度
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(x.min(), x.max())

    ax.set_xlabel("日期（2025 年）", labelpad=8)
    ax.set_ylabel("实时电价（元/kWh）", labelpad=8)
    ax.set_title(
        "图 2  每日同一时刻电价在一年中的变化（4 小时一间隔，时刻越晚配色越深）",
        pad=46,
    )

    # 图例：按时刻从早到晚横向排列，配色由浅到深
    leg = ax.legend(
        title="每日时刻",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(hours),
        frameon=False,
        handlelength=2.6,
        columnspacing=2.2,
    )
    leg.get_title().set_fontsize(10.5)
    leg.get_title().set_fontweight("bold")

    # 逐时刻年度中位数参考线（细虚线，同色系）
    for i, h in enumerate(hours):
        ax.axhline(
            float(np.nanmedian(wide[h].values)),
            color=colors[i],
            linewidth=0.8,
            linestyle=(0, (6, 4)),
            alpha=0.45,
            zorder=1,
        )

    # 月度刻度分隔线
    for m in range(2, 13):
        ax.axvline(
            pd.Timestamp(year=2025, month=m, day=1),
            color="#BBBBBB", linewidth=0.7, linestyle="-.", alpha=0.65, zorder=0,
        )

    # 标注全年最高/最低点
    col_max = wide.max(axis=1)
    col_min = wide.min(axis=1)
    hi_day, lo_day = col_max.idxmax(), col_min.idxmin()
    hi_h = int(wide.loc[hi_day].idxmax())
    lo_h = int(wide.loc[lo_day].idxmin())
    ci, cj = hours.index(hi_h), hours.index(lo_h)

    # 上下各留一段空白，专门给极值标注使用，避免文字压在曲线或图例上
    y_lo = float(np.nanmin(wide.values))
    y_hi = float(np.nanmax(wide.values))
    span = y_hi - y_lo
    ax.set_ylim(y_lo - 0.30 * span, y_hi + 0.26 * span)

    ax.scatter([hi_day], [col_max.max()], s=52, facecolor="white",
               edgecolor=colors[ci], linewidth=1.7, zorder=20)
    ax.annotate(
        f"全年最高 {hi_day:%Y-%m-%d} {hi_h:02d}:00  {col_max.max():.4f} 元/kWh",
        xy=(hi_day, col_max.max()),
        xytext=(hi_day, y_hi + 0.20 * span),
        ha="center", va="center", fontsize=9.5, color="#7A1F1F",
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9,
                        connectionstyle="arc3,rad=0.10"),
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#D9A0A0", alpha=0.95),
        zorder=21,
    )
    ax.scatter([lo_day], [col_min.min()], s=52, facecolor="white",
               edgecolor=colors[cj], linewidth=1.7, zorder=20)
    ax.annotate(
        f"全年最低 {lo_day:%Y-%m-%d} {lo_h:02d}:00  {col_min.min():.4f} 元/kWh",
        xy=(lo_day, col_min.min()),
        xytext=(lo_day, y_lo - 0.20 * span),
        ha="center", va="center", fontsize=9.5, color="#0F4C81",
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9,
                        connectionstyle="arc3,rad=-0.10"),
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#A0B8D9", alpha=0.95),
        zorder=21,
    )

    out = FIG_DIR / "02_每日同一时刻电价年内变化.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] 已保存 {out}")

    # 摘要统计
    summary = pd.DataFrame(
        {
            "hour": hours,
            "annual_mean": [float(np.nanmean(wide[h].values)) for h in hours],
            "annual_median": [float(np.nanmedian(wide[h].values)) for h in hours],
            "std": [float(np.nanstd(wide[h].values, ddof=1)) for h in hours],
            "cv": [
                float(np.nanstd(wide[h].values, ddof=1) / np.nanmean(wide[h].values))
                for h in hours
            ],
            "min": [float(np.nanmin(wide[h].values)) for h in hours],
            "max": [float(np.nanmax(wide[h].values)) for h in hours],
        }
    )
    summary.to_csv(FIG_DIR / "02_每日同一时刻电价年内变化_summary.csv",
                   index=False, float_format="%.6f")
    print("[stats] 各时刻年度汇总：")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df = load_price()
    figure1_boxplot(df)
    figure2_intra_year(df)
    print("\n[done] 全部图形输出至", FIG_DIR)


if __name__ == "__main__":
    main()
