#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 国赛论文风格绘图脚本（适配当前项目结构）

当前项目结构：
CUMCM_2026/
└─ q1/
   ├─ input/
   │  └─ q1_typical_day.csv
   ├─ q1.py
   └─ results/
      ├─ result1.xlsx
      └─ figures/
         └─ draw_award_style.py   <- 本文件建议放这里

关键设计：
- 不依赖 intermediate 文件夹。
- 直接动态导入 q1/q1.py，并调用它自己的 solve_from_input()。
- 因此绘图数据与 Q1 最终优化模型完全同源，不重新定义优化逻辑。
- 图片下方不添加解释段落。
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np


# =============================================================================
# 路径
# =============================================================================
HERE = Path(__file__).resolve().parent          # q1/results/figures
Q1 = HERE.parents[1]                           # q1
Q1_PY = Q1 / "q1.py"
INPUT_CSV = Q1 / "input" / "q1_typical_day.csv"

if not Q1_PY.exists():
    raise SystemExit(f"找不到 q1.py：{Q1_PY}")
if not INPUT_CSV.exists():
    raise SystemExit(f"找不到输入文件：{INPUT_CSV}")


# =============================================================================
# 动态导入 q1.py —— 直接复用正式求解逻辑
# =============================================================================
spec = importlib.util.spec_from_file_location("q1_model", Q1_PY)
q1_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(q1_model)


# =============================================================================
# 中文字体
# =============================================================================
_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "Source Han Sans SC", "PingFang SC", "Hiragino Sans GB",
    "WenQuanYi Zen Hei", "Arial Unicode MS",
]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_cjk = [name for name in _CJK_CANDIDATES if name in _available]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _cjk + ["DejaVu Sans"],
    "axes.unicode_minus": False,

    "figure.dpi": 120,
    "savefig.dpi": 360,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#30343B",
    "axes.linewidth": 0.95,

    "axes.labelsize": 12.5,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,

    "grid.color": "#B8C0C8",
    "grid.alpha": 0.36,
    "grid.linestyle": "--",
    "grid.linewidth": 0.65,
    "axes.axisbelow": True,

    "legend.fontsize": 10.2,
    "legend.frameon": True,
    "legend.framealpha": 0.94,
    "legend.edgecolor": "#B8BCC3",
})


# =============================================================================
# 配色
# =============================================================================
C_NAVY = "#0B4F8A"
C_BLUE = "#1E6FAE"
C_GOLD = "#F29A18"
C_RED = "#C43B2F"
C_GREEN = "#1E7A5C"
C_COST = "#C96B5C"
C_GREY = "#7F8B96"

BG_NIGHT = "#EAF4FF"
BG_DAY = "#FFF5DD"
BG_SURPLUS = "#F6D58A"

DT_H = 1.0 / 6.0
N = 144


# =============================================================================
# 数据：直接来自 q1.py 的 source + solution
# =============================================================================
def load_model_data():
    cfg, source, solution, checks = q1_model.solve_from_input(INPUT_CSV, "auto")

    if len(source) != N:
        raise RuntimeError(f"Q1 输入应为 144 个 10 分钟区间，实际 {len(source)}。")

    # source 中是 kW；q1.py 内部求解时把每个 10 min 区间转换为 kWh
    price = source["price_yuan_per_kwh"].to_numpy(dtype=float)
    load_kw = source["load_kw"].to_numpy(dtype=float)
    pv_kw = source["pv_forecast_kw"].to_numpy(dtype=float)

    t_start = source["interval_start_minute"].to_numpy(dtype=float) / 60.0
    t_end = source["interval_end_minute"].to_numpy(dtype=float) / 60.0
    t_mid = (t_start + t_end) / 2.0

    grid_kwh = np.asarray(solution["grid"], dtype=float)
    charge_kwh = np.asarray(solution["charge"], dtype=float)
    discharge_kwh = np.asarray(solution["discharge"], dtype=float)
    storage_kwh = np.asarray(solution["storage"], dtype=float)

    # 功率：10 min 能量 / (1/6 h)
    grid_kw = grid_kwh / DT_H
    charge_kw = charge_kwh / DT_H
    discharge_kw = discharge_kwh / DT_H

    interval_cost = price * grid_kwh
    cum_cost = np.cumsum(interval_cost)

    return {
        "cfg": cfg,
        "checks": checks,
        "source": source,
        "solution": solution,

        "t_start": t_start,
        "t_end": t_end,
        "t_mid": t_mid,

        "price": price,
        "load_kw": load_kw,
        "pv_kw": pv_kw,

        "grid_kwh": grid_kwh,
        "grid_kw": grid_kw,
        "charge_kw": charge_kw,
        "discharge_kw": discharge_kw,

        "storage_t": np.arange(N + 1) * DT_H,
        "storage_kwh": storage_kwh,

        "interval_cost": interval_cost,
        "cum_cost": cum_cost,
        "total_cost": float(interval_cost.sum()),
        "total_load_kwh": float(load_kw.sum() * DT_H),
        "total_pv_kwh": float(pv_kw.sum() * DT_H),
    }


# =============================================================================
# 工具函数
# =============================================================================
def step_xy(t_start, values):
    """前向保持阶梯；最后补到 24:00。"""
    x = np.r_[np.asarray(t_start, dtype=float), 24.0]
    y = np.r_[np.asarray(values, dtype=float), float(values[-1])]
    return x, y


def fmt_hhmm(hour: float) -> str:
    total_min = int(round(hour * 60))
    if total_min >= 1440:
        return "24:00"
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


def interval_text(i: int) -> str:
    return f"{fmt_hhmm(i * DT_H)}–{fmt_hhmm((i + 1) * DT_H)}"


def contiguous_true_intervals(mask):
    mask = list(bool(v) for v in mask)
    out = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start * DT_H, i * DT_H))
            start = None
    if start is not None:
        out.append((start * DT_H, len(mask) * DT_H))
    return out


def longest_interval(intervals):
    return max(intervals, key=lambda z: z[1] - z[0]) if intervals else None


def setup_hour_axis(ax):
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_day_night(ax, labels=True):
    # 分界线和背景严格使用同一个 x=6 / 18
    ax.axvspan(0, 6, color=BG_NIGHT, alpha=0.80, zorder=-20)
    ax.axvspan(6, 18, color=BG_DAY, alpha=0.75, zorder=-20)
    ax.axvspan(18, 24, color=BG_NIGHT, alpha=0.80, zorder=-20)

    ax.axvline(6, color=C_NAVY, lw=1.0, ls="--", alpha=0.75, zorder=-5)
    ax.axvline(18, color=C_NAVY, lw=1.0, ls="--", alpha=0.75, zorder=-5)

    if labels:
        trans = ax.get_xaxis_transform()
        ax.text(
            3, 0.987, "夜间  0:00–6:00",
            transform=trans, ha="center", va="top",
            color=C_NAVY, fontsize=10.3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.23", fc=BG_NIGHT, ec="none", alpha=0.9),
        )
        ax.text(
            12, 0.987, "白天  6:00–18:00",
            transform=trans, ha="center", va="top",
            color="#8C5A00", fontsize=10.3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.23", fc=BG_DAY, ec="none", alpha=0.9),
        )
        ax.text(
            21, 0.987, "夜间  18:00–24:00",
            transform=trans, ha="center", va="top",
            color=C_NAVY, fontsize=10.3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.23", fc=BG_NIGHT, ec="none", alpha=0.9),
        )


def legend_box(ax, handles, labels, loc="upper left", ncol=1):
    leg = ax.legend(
        handles, labels, loc=loc, ncol=ncol,
        fancybox=True, borderpad=0.55,
        handlelength=2.8, labelspacing=0.42,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.94)
    leg.get_frame().set_edgecolor("#B7BDC5")
    return leg


def save_figure(fig, path):
    fig.savefig(path, dpi=360, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# =============================================================================
# 图1：负荷 + 光伏 + 电价
# =============================================================================
def fig1_input(d, out):
    fig, ax = plt.subplots(figsize=(12.2, 6.35))

    x_load, y_load = step_xy(d["t_start"], d["load_kw"])
    x_pv, y_pv = step_xy(d["t_start"], d["pv_kw"])
    x_price, y_price = step_xy(d["t_start"], d["price"])

    ymax = max(float(d["load_kw"].max()), float(d["pv_kw"].max()))
    ax.set_ylim(0, ymax * 1.22)

    add_day_night(ax, labels=True)

    l_load, = ax.step(
        x_load, y_load, where="post",
        color=C_NAVY, lw=2.35, label="小区负荷功率", zorder=8,
    )
    l_pv, = ax.step(
        x_pv, y_pv, where="post",
        color=C_GOLD, lw=2.45, label="光伏预测功率", zorder=9,
    )

    ax.fill_between(
        x_load, 0, y_load, step="post",
        color=C_NAVY, alpha=0.06, zorder=1,
    )
    ax.fill_between(
        x_pv, 0, y_pv, step="post",
        color=C_GOLD, alpha=0.11, zorder=2,
    )

    # 光伏超过负荷区间：严格由 10 min 数据判定
    mask = d["pv_kw"] > d["load_kw"]
    surplus_intervals = contiguous_true_intervals(mask)
    for a, b in surplus_intervals:
        ax.axvspan(a, b, color=BG_SURPLUS, alpha=0.20, zorder=0)

    main_surplus = longest_interval(surplus_intervals)
    if main_surplus:
        a, b = main_surplus
        mid = (a + b) / 2
        ax.annotate(
            f"光伏出力超过负荷\n{fmt_hhmm(a)}–{fmt_hhmm(b)}",
            xy=(mid, ymax * 0.70),
            xytext=(mid, ymax * 1.11),
            ha="center", va="center",
            fontsize=10.2, color="#8C5A00",
            arrowprops=dict(arrowstyle="-|>", color="#9A6500", lw=1.05),
            bbox=dict(
                boxstyle="round,pad=0.34",
                fc="#FFF9EC", ec=C_GOLD, alpha=0.96,
            ),
            zorder=20,
        )

    # PV 峰值
    i_pv = int(np.argmax(d["pv_kw"]))
    pv_t = float(d["t_mid"][i_pv])
    pv_peak = float(d["pv_kw"][i_pv])

    ax.scatter(
        pv_t, pv_peak, s=60, color="#D6550D",
        ec="white", lw=0.8, zorder=25,
    )
    ax.annotate(
        f"光伏峰值约 {pv_peak:,.0f} kW\n{interval_text(i_pv)}",
        xy=(pv_t, pv_peak),
        xytext=(min(pv_t + 1.15, 19.7), pv_peak * 1.075),
        ha="left", va="bottom",
        fontsize=10.0, fontweight="bold", color="#B64016",
        arrowprops=dict(arrowstyle="-|>", color="#B64016", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.27", fc="white", ec="#D98C70", alpha=0.93),
    )

    # 负荷峰值
    i_load = int(np.argmax(d["load_kw"]))
    load_t = float(d["t_mid"][i_load])
    load_peak = float(d["load_kw"][i_load])

    ax.scatter(
        load_t, load_peak, s=52, color=C_NAVY,
        ec="white", lw=0.8, zorder=25,
    )
    ax.annotate(
        f"负荷峰值约 {load_peak:,.0f} kW\n{interval_text(i_load)}",
        xy=(load_t, load_peak),
        xytext=(max(load_t - 1.2, 1.0), load_peak * 1.10),
        ha="right", va="bottom",
        fontsize=9.8, fontweight="bold", color=C_NAVY,
        arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#8AB2D4", alpha=0.93),
    )

    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("功率 / kW")
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    setup_hour_axis(ax)

    ax2 = ax.twinx()
    l_price, = ax2.step(
        x_price, y_price, where="post",
        color=C_RED, lw=1.72, ls="-.",
        label="购电电价", zorder=10,
    )
    ax2.set_ylabel("购电电价 / (元·kWh$^{-1}$)", color=C_RED)
    ax2.tick_params(axis="y", colors=C_RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(C_RED)
    ax2.set_ylim(0, float(d["price"].max()) * 1.42)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax2.grid(False)

    legend_box(
        ax,
        [l_load, l_pv, l_price],
        ["小区负荷功率", "光伏预测功率", "购电电价"],
        loc="upper left",
    )

    # 信息放图内，不放图片下方
    ax.text(
        0.985, 0.035,
        f"全天负荷电量  {d['total_load_kwh']:,.0f} kWh\n"
        f"全天光伏电量  {d['total_pv_kwh']:,.0f} kWh",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9.3, color="#5F6871",
        bbox=dict(
            boxstyle="round,pad=0.33",
            fc="white", ec="#C9CED4", alpha=0.90,
        ),
        zorder=30,
    )

    ax.set_title("典型日输入：购电电价、小区负荷与光伏预测功率", pad=11)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.10)
    save_figure(fig, out)


# =============================================================================
# 图2：储能电量 + 电价
# =============================================================================
def fig2_storage_price(d, out):
    cfg = d["cfg"]
    lo = float(cfg["storage_min_kwh"])
    hi = float(cfg["storage_max_kwh"])
    init = float(cfg["initial_storage_kwh"])

    t = d["storage_t"]
    e = d["storage_kwh"]
    yrange = hi - lo

    fig, ax = plt.subplots(figsize=(12.2, 6.25))
    ax.set_ylim(max(0, lo - 0.12 * yrange), hi + 0.18 * yrange)

    add_day_night(ax, labels=True)

    ax.axhspan(lo, hi, color=C_GREEN, alpha=0.075, zorder=0)
    ax.axhline(hi, color=C_GREY, ls=":", lw=1.35, zorder=2)
    ax.axhline(lo, color=C_GREY, ls=":", lw=1.35, zorder=2)
    ax.axhline(init, color="#8C8C8C", ls="--", lw=1.05, alpha=0.82, zorder=2)

    l_energy, = ax.plot(
        t, e, color=C_GREEN, lw=2.65,
        label="储能电量", zorder=8,
    )
    ax.fill_between(
        t, lo, e, where=e >= lo,
        color=C_GREEN, alpha=0.15, zorder=1,
    )

    # 严格标注容量边界
    ax.text(
        23.65, hi, f"容量上限  {hi:,.0f} kWh",
        ha="right", va="center",
        fontsize=9.4, color="#5F6871",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D0D3D7", alpha=0.90),
    )
    ax.text(
        23.65, lo, f"容量下限  {lo:,.0f} kWh",
        ha="right", va="center",
        fontsize=9.4, color="#5F6871",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D0D3D7", alpha=0.90),
    )

    # 最低点
    i_min = int(np.argmin(e))
    t_min = float(t[i_min])
    e_min = float(e[i_min])

    ax.scatter(
        t_min, e_min, s=58, color=C_NAVY,
        ec="white", lw=0.8, zorder=22,
    )
    ax.annotate(
        f"最低电量 {e_min:,.0f} kWh\n{fmt_hhmm(t_min)}",
        xy=(t_min, e_min),
        xytext=(max(t_min - 1.0, 0.8), e_min + yrange * 0.18),
        ha="right", va="bottom",
        fontsize=9.8, fontweight="bold", color=C_NAVY,
        arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#8AB2D4", alpha=0.94),
    )

    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("储能电量 / kWh")
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    setup_hour_axis(ax)

    # 电价副轴
    x_price, y_price = step_xy(d["t_start"], d["price"])
    ax2 = ax.twinx()
    l_price, = ax2.step(
        x_price, y_price, where="post",
        color=C_RED, lw=1.7, ls="-.",
        label="购电电价", zorder=10,
    )
    ax2.set_ylabel("购电电价 / (元·kWh$^{-1}$)", color=C_RED)
    ax2.tick_params(axis="y", colors=C_RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(C_RED)
    ax2.set_ylim(0, float(d["price"].max()) * 1.45)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax2.grid(False)

    init_line = Line2D(
        [], [], color="#8C8C8C", ls="--", lw=1.05,
        label=f"初始电量 {init:,.0f} kWh",
    )
    band_patch = Patch(
        facecolor=C_GREEN, alpha=0.10, edgecolor="none",
        label=f"允许储能区间 {lo:,.0f}–{hi:,.0f} kWh",
    )

    legend_box(
        ax,
        [l_energy, l_price, init_line, band_patch],
        [
            "储能电量",
            "购电电价",
            f"初始电量 {init:,.0f} kWh",
            f"允许储能区间 {lo:,.0f}–{hi:,.0f} kWh",
        ],
        loc="upper left",
    )

    ax.set_title("储能电量与分时购电电价对照", pad=11)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.10)
    save_figure(fig, out)


# =============================================================================
# 图3：逐10分钟购电花费 + 累计花费
# =============================================================================
def fig3_cost(d, out):
    fig, ax = plt.subplots(figsize=(12.2, 6.25))

    peak = float(d["interval_cost"].max())
    ax.set_ylim(0, peak * 1.24)

    add_day_night(ax, labels=True)

    bars = ax.bar(
        d["t_mid"], d["interval_cost"],
        width=DT_H * 0.88,
        color=C_COST, alpha=0.72,
        edgecolor="white", linewidth=0.25,
        label="逐10分钟购电花费",
        zorder=5,
    )

    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("逐10分钟购电花费 / 元")
    setup_hour_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    # 累计费用画在区间结束时刻
    ax2 = ax.twinx()
    l_cum, = ax2.plot(
        np.r_[0.0, d["t_end"]],
        np.r_[0.0, d["cum_cost"]],
        color=C_NAVY, lw=2.65,
        label="当日累计购电花费",
        zorder=9,
    )
    ax2.set_ylabel("当日累计购电花费 / 元", color=C_NAVY)
    ax2.tick_params(axis="y", colors=C_NAVY)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(C_NAVY)
    ax2.set_ylim(0, d["total_cost"] * 1.22)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.grid(False)

    # 单时段最高费用
    i_peak = int(np.argmax(d["interval_cost"]))
    xp = float(d["t_mid"][i_peak])
    yp = float(d["interval_cost"][i_peak])

    ax.scatter(
        xp, yp, s=48, color="#A94F43",
        ec="white", lw=0.75, zorder=21,
    )
    ax.annotate(
        f"单时段最高 {yp:,.0f} 元\n{interval_text(i_peak)}",
        xy=(xp, yp),
        xytext=(min(xp + 1.0, 19.0), yp * 1.10),
        ha="left", va="bottom",
        fontsize=9.7, fontweight="bold", color="#9D4439",
        arrowprops=dict(arrowstyle="-|>", color="#9D4439", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#D3A19A", alpha=0.94),
    )

    # 总额终点严格 x=24:00
    ax2.scatter(
        24.0, d["total_cost"],
        s=64, color=C_NAVY,
        ec="white", lw=0.8,
        zorder=22, clip_on=False,
    )
    ax2.annotate(
        f"全天合计\n{d['total_cost']:,.0f} 元",
        xy=(24.0, d["total_cost"]),
        xytext=(21.9, d["total_cost"] * 1.12),
        ha="center", va="center",
        fontsize=10.3, fontweight="bold", color=C_NAVY,
        arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.32", fc="#EDF5FC", ec="#8AB2D4", alpha=0.96),
    )

    legend_box(
        ax,
        [bars, l_cum],
        ["逐10分钟购电花费", "当日累计购电花费"],
        loc="upper left",
    )

    ax.set_title("逐10分钟购电花费与当日累计购电花费", pad=11)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.10)
    save_figure(fig, out)


# =============================================================================
# 主程序
# =============================================================================
def main():
    d = load_model_data()

    outputs = [
        ("fig1_input_power_price_award.png", fig1_input),
        ("fig2_battery_energy_price_award.png", fig2_storage_price),
        ("fig3_purchase_cost_award.png", fig3_cost),
    ]

    print(f"[Q1] 使用正式模型：{Q1_PY}")
    print(f"[Q1] 使用输入：{INPUT_CSV}")
    print(f"[输出目录] {HERE}")

    for filename, fn in outputs:
        target = HERE / filename
        fn(d, target)
        print(f"  已生成：{filename} + {target.with_suffix('.pdf').name}")

    print("-" * 60)
    print(f"全天负荷电量：{d['total_load_kwh']:,.3f} kWh")
    print(f"全天光伏电量：{d['total_pv_kwh']:,.3f} kWh")
    print(f"全天购电量：{d['grid_kwh'].sum():,.3f} kWh")
    print(f"全天购电费用：{d['total_cost']:,.3f} 元")
    print(
        f"储能电量范围："
        f"{d['storage_kwh'].min():,.3f}–{d['storage_kwh'].max():,.3f} kWh"
    )
    print("-" * 60)


if __name__ == "__main__":
    main()
