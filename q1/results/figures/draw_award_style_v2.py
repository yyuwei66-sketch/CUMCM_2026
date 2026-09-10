#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 论文风格绘图脚本（V2：保留队友的信息量，但整体风格改为更像国赛获奖论文）

放置位置建议：
    q1/results/figures/draw_award_style_v2.py

运行方式：
    python .\q1\results\figures\draw_award_style_v2.py

特点
----
1. 数据严格来自 q1.py 的正式求解函数 solve_from_input()；
2. 不依赖 intermediate 文件夹；
3. 保留三张图的信息量：
   - 图1：负荷 + 光伏 + 购电电价 + 光伏超过负荷区间 + 峰值标注
   - 图2：储能电量 + 购电电价 + 储能上下限 + 初始电量 + 极值标注
   - 图3：逐10分钟购电花费 + 累计购电花费 + 最高单时段 + 全天总额
4. 风格上避免默认 Matplotlib 灰底，改为：
   - 白底
   - 浅蓝 / 浅米黄昼夜分区
   - 更克制的学术配色
   - 更像论文插图，而不是程序默认图
5. 图片底部不再添加解释段落。
"""

from __future__ import annotations

import importlib.util
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
HERE = Path(__file__).resolve().parent            # q1/results/figures
Q1 = HERE.parents[1]                             # q1
Q1_PY = Q1 / "q1.py"
INPUT_CSV = Q1 / "input" / "q1_typical_day.csv"

if not Q1_PY.exists():
    raise SystemExit(f"找不到 q1.py：{Q1_PY}")
if not INPUT_CSV.exists():
    raise SystemExit(f"找不到输入文件：{INPUT_CSV}")


# =============================================================================
# 动态导入 q1.py，确保绘图与正式求解结果同源
# =============================================================================
spec = importlib.util.spec_from_file_location("q1_model", Q1_PY)
q1_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(q1_model)


# =============================================================================
# 中文字体
# =============================================================================
_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Hiragino Sans GB",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
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
    "savefig.pad_inches": 0.10,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.00,

    "axes.labelsize": 13,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "grid.color": "#BFC6CE",
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
    "grid.linewidth": 0.70,
    "axes.axisbelow": True,

    "legend.fontsize": 10.5,
    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "#BFC6CE",
})


# =============================================================================
# 配色
# =============================================================================
C_NAVY = "#0D4D8B"
C_NAVY_2 = "#1F5FA4"
C_GOLD = "#F39A17"
C_GOLD_DARK = "#B87106"
C_RED = "#C43D2B"
C_GREEN = "#177A5A"
C_SALMON = "#CC7A6B"
C_GREY = "#7B8794"

BG_NIGHT = "#EAF3FB"
BG_DAY = "#FFF7E6"
BG_LIMIT = "#EAF7F1"

BOX_BLUE = "#EAF2FC"
BOX_ORANGE = "#FFF5E8"
BOX_RED = "#FDF0EE"

DT_H = 1.0 / 6.0
N = 144


# =============================================================================
# 数据读取：直接调用 q1.py
# =============================================================================
def load_model_data():
    cfg, source, solution, checks = q1_model.solve_from_input(INPUT_CSV, "auto")

    if len(source) != N:
        raise RuntimeError(f"Q1 输入应为 144 个 10 分钟区间，实际 {len(source)} 个。")

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
    x = np.r_[np.asarray(t_start, dtype=float), 24.0]
    y = np.r_[np.asarray(values, dtype=float), float(values[-1])]
    return x, y


def fmt_hhmm(hour: float) -> str:
    minute = int(round(hour * 60))
    if minute >= 24 * 60:
        return "24:00"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def interval_text(i: int) -> str:
    return f"{fmt_hhmm(i * DT_H)}–{fmt_hhmm((i + 1) * DT_H)}"


def contiguous_true_intervals(mask):
    mask = list(bool(v) for v in mask)
    intervals = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append((start * DT_H, i * DT_H))
            start = None
    if start is not None:
        intervals.append((start * DT_H, len(mask) * DT_H))
    return intervals


def longest_interval(intervals):
    return max(intervals, key=lambda z: z[1] - z[0]) if intervals else None


def setup_axes(ax):
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", length=5, width=0.9, color="#333333")
    ax.tick_params(axis="x", which="minor", length=2.8, width=0.7, color="#666666")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_day_night_background(ax):
    """
    背景色与分界线严格对齐在 0–6 / 6–18 / 18–24 h。
    """
    ax.axvspan(0, 6, color=BG_NIGHT, alpha=0.95, zorder=-30)
    ax.axvspan(6, 18, color=BG_DAY, alpha=0.95, zorder=-30)
    ax.axvspan(18, 24, color=BG_NIGHT, alpha=0.95, zorder=-30)

    ax.axvline(6, color=C_NAVY_2, ls="--", lw=1.15, alpha=0.80, zorder=-5)
    ax.axvline(18, color=C_NAVY_2, ls="--", lw=1.15, alpha=0.80, zorder=-5)


def add_period_header(ax):
    """
    顶部时段标题 + 双向箭头，更贴近你选中的那类风格。
    """
    trans = ax.get_xaxis_transform()
    y = 0.998

    def span_arrow(x0, x1, label, color):
        ax.annotate(
            "",
            xy=(x0, y), xytext=(x1, y),
            xycoords=trans, textcoords=trans,
            arrowprops=dict(arrowstyle="<|-|>", lw=1.1, color=color, shrinkA=0, shrinkB=0),
            annotation_clip=False,
        )
        ax.text(
            (x0 + x1) / 2, y + 0.018, label,
            transform=trans, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
        )

    span_arrow(0, 6, "夜间时段\n(0:00–6:00)", C_NAVY)
    span_arrow(6, 18, "白天时段\n(6:00–18:00)", "#7E5200")
    span_arrow(18, 24, "夜间时段\n(18:00–24:00)", C_NAVY)


def legend_box(ax, handles, labels, loc="upper left", ncol=1):
    leg = ax.legend(
        handles, labels, loc=loc, ncol=ncol,
        fancybox=True, borderpad=0.58,
        handlelength=2.9, labelspacing=0.42,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.96)
    leg.get_frame().set_edgecolor("#BDC5CD")
    return leg


def save_figure(fig, path):
    fig.savefig(path, dpi=360)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# =============================================================================
# 图1：输入——负荷 + 光伏 + 电价
# =============================================================================
def fig1_input(d, out):
    fig, ax = plt.subplots(figsize=(12.8, 6.9))

    x_load, y_load = step_xy(d["t_start"], d["load_kw"])
    x_pv, y_pv = step_xy(d["t_start"], d["pv_kw"])
    x_price, y_price = step_xy(d["t_start"], d["price"])

    ymax = max(float(d["load_kw"].max()), float(d["pv_kw"].max()))
    ax.set_ylim(0, ymax * 1.18)

    add_day_night_background(ax)
    add_period_header(ax)

    # 光伏 > 负荷区间
    surplus_mask = d["pv_kw"] > d["load_kw"]
    surplus_intervals = contiguous_true_intervals(surplus_mask)
    for a, b in surplus_intervals:
        ax.axvspan(a, b, color="#F7DFA3", alpha=0.35, zorder=-10)

    # 曲线
    l_load, = ax.step(
        x_load, y_load, where="post",
        color=C_NAVY, lw=2.35, label="系统负荷", zorder=8
    )
    l_pv, = ax.step(
        x_pv, y_pv, where="post",
        color=C_GOLD, lw=2.40, label="光伏出力预测", zorder=9
    )

    ax.fill_between(x_load, 0, y_load, step="post", color=C_NAVY, alpha=0.05, zorder=1)
    ax.fill_between(x_pv, 0, y_pv, step="post", color=C_GOLD, alpha=0.10, zorder=2)

    # 电价
    ax2 = ax.twinx()
    l_price, = ax2.step(
        x_price, y_price, where="post",
        color=C_RED, lw=1.75, ls="-.", label="购电电价", zorder=10
    )
    ax2.set_ylabel("购电电价 / (元·kWh$^{-1}$)", color=C_RED, fontsize=13)
    ax2.tick_params(axis="y", colors=C_RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(C_RED)
    ax2.set_ylim(0, float(d["price"].max()) * 1.38)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax2.grid(False)

    # 峰值标注
    i_load = int(np.argmax(d["load_kw"]))
    t_load = float(d["t_mid"][i_load])
    v_load = float(d["load_kw"][i_load])
    ax.scatter([t_load], [v_load], s=56, color=C_NAVY, edgecolor="white", linewidth=0.8, zorder=20)
    ax.annotate(
        f"负荷峰值\n约 {v_load:,.0f} kW\n({interval_text(i_load)})",
        xy=(t_load, v_load),
        xytext=(max(t_load - 1.0, 1.0), v_load * 1.09),
        ha="right", va="bottom",
        fontsize=10.2, fontweight="bold", color=C_NAVY,
        arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.28", fc=BOX_BLUE, ec="#8DB6DA", alpha=0.97),
        zorder=20,
    )

    i_pv = int(np.argmax(d["pv_kw"]))
    t_pv = float(d["t_mid"][i_pv])
    v_pv = float(d["pv_kw"][i_pv])
    ax.scatter([t_pv], [v_pv], s=62, color="#D75A0E", edgecolor="white", linewidth=0.8, zorder=21)
    ax.annotate(
        f"光伏峰值\n约 {v_pv:,.0f} kW\n({interval_text(i_pv)})",
        xy=(t_pv, v_pv),
        xytext=(min(t_pv + 1.15, 19.3), v_pv * 1.02),
        ha="left", va="bottom",
        fontsize=10.3, fontweight="bold", color="#B44912",
        arrowprops=dict(arrowstyle="-|>", color="#B44912", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.28", fc=BOX_RED, ec="#E09A7D", alpha=0.97),
        zorder=21,
    )

    # 主盈余区间标注
    longest = longest_interval(surplus_intervals)
    if longest is not None:
        a, b = longest
        mid = (a + b) / 2
        ax.annotate(
            f"光伏出力高于负荷\n{fmt_hhmm(a)}–{fmt_hhmm(b)}",
            xy=(mid, ymax * 0.58),
            xytext=(mid, ymax * 1.02),
            ha="center", va="center",
            fontsize=10.0, color="#8C5A00",
            arrowprops=dict(arrowstyle="-|>", color="#9A6800", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.32", fc=BOX_ORANGE, ec=C_GOLD, alpha=0.97),
            zorder=20,
        )

    # 图例和角标信息
    legend_box(
        ax,
        [l_load, l_pv, l_price],
        ["系统负荷", "光伏出力预测", "购电电价"],
        loc="upper left",
    )

    ax.text(
        0.985, 0.055,
        f"全天负荷电量：{d['total_load_kwh']:,.0f} kWh\n"
        f"全天光伏电量：{d['total_pv_kwh']:,.0f} kWh",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9.8, color="#5F6871",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#C7CFD6", alpha=0.94),
    )

    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("功率 / kW")
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    setup_axes(ax)
    ax.set_title("图1  典型日输入：购电电价、小区负荷与光伏预测功率", pad=14)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.10)
    save_figure(fig, out)


# =============================================================================
# 图2：储能电量 + 电价
# =============================================================================
def fig2_storage(d, out):
    lo = float(d["cfg"]["storage_min_kwh"])
    hi = float(d["cfg"]["storage_max_kwh"])
    init = float(d["cfg"]["initial_storage_kwh"])

    t = d["storage_t"]
    e = d["storage_kwh"]
    yrange = hi - lo

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.set_ylim(max(0.0, lo - 0.10 * yrange), hi + 0.18 * yrange)

    add_day_night_background(ax)
    add_period_header(ax)

    # 允许区间
    ax.axhspan(lo, hi, color=BG_LIMIT, alpha=0.90, zorder=-15)
    ax.axhline(hi, color=C_GREY, ls=":", lw=1.35, zorder=2)
    ax.axhline(lo, color=C_GREY, ls=":", lw=1.35, zorder=2)
    ax.axhline(init, color="#8D8D8D", ls="--", lw=1.08, alpha=0.9, zorder=2)

    l_energy, = ax.plot(t, e, color=C_GREEN, lw=2.8, label="储能电量", zorder=8)
    ax.fill_between(t, lo, e, where=e >= lo, color=C_GREEN, alpha=0.13, interpolate=True, zorder=1)

    # 电价副轴
    x_price, y_price = step_xy(d["t_start"], d["price"])
    ax2 = ax.twinx()
    l_price, = ax2.step(
        x_price, y_price, where="post",
        color=C_RED, lw=1.75, ls="-.", label="购电电价", zorder=10
    )
    ax2.set_ylabel("购电电价 / (元·kWh$^{-1}$)", color=C_RED, fontsize=13)
    ax2.tick_params(axis="y", colors=C_RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(C_RED)
    ax2.set_ylim(0, float(d["price"].max()) * 1.40)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax2.grid(False)

    # 极值标注
    i_min = int(np.argmin(e))
    t_min = float(t[i_min])
    e_min = float(e[i_min])
    ax.scatter([t_min], [e_min], s=58, color=C_NAVY, edgecolor="white", linewidth=0.8, zorder=20)
    ax.annotate(
        f"最低电量\n{e_min:,.0f} kWh\n({fmt_hhmm(t_min)})",
        xy=(t_min, e_min),
        xytext=(max(t_min - 1.0, 1.0), e_min + yrange * 0.18),
        ha="right", va="bottom",
        fontsize=10.1, fontweight="bold", color=C_NAVY,
        arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.28", fc=BOX_BLUE, ec="#8DB6DA", alpha=0.97),
    )

    i_max = int(np.argmax(e))
    t_max = float(t[i_max])
    e_max = float(e[i_max])
    ax.scatter([t_max], [e_max], s=46, color=C_GREEN, edgecolor="white", linewidth=0.8, zorder=20)
    ax.annotate(
        f"最高电量\n{e_max:,.0f} kWh\n({fmt_hhmm(t_max)})",
        xy=(t_max, e_max),
        xytext=(min(t_max + 0.95, 19.2), min(e_max + yrange * 0.06, hi + yrange * 0.12)),
        ha="left", va="bottom",
        fontsize=9.7, fontweight="bold", color=C_GREEN,
        arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="#EDF9F4", ec="#8EC3B0", alpha=0.97),
    )

    # 上下限说明放右侧图内
    ax.text(
        23.78, hi, f"容量上限  {hi:,.0f} kWh",
        ha="right", va="center", fontsize=9.6, color="#5F6871",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D0D6DB", alpha=0.95),
    )
    ax.text(
        23.78, lo, f"容量下限  {lo:,.0f} kWh",
        ha="right", va="center", fontsize=9.6, color="#5F6871",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D0D6DB", alpha=0.95),
    )

    init_line = Line2D([], [], color="#8D8D8D", ls="--", lw=1.08, label=f"初始电量 {init:,.0f} kWh")
    range_patch = Patch(facecolor=BG_LIMIT, edgecolor="none", label=f"允许储能区间 {lo:,.0f}–{hi:,.0f} kWh")

    legend_box(
        ax,
        [l_energy, l_price, init_line, range_patch],
        ["储能电量", "购电电价", f"初始电量 {init:,.0f} kWh", f"允许储能区间 {lo:,.0f}–{hi:,.0f} kWh"],
        loc="upper left"
    )

    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("储能电量 / kWh")
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    setup_axes(ax)
    ax.set_title("图2  储能电量与分时购电电价对照", pad=14)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.10)
    save_figure(fig, out)


# =============================================================================
# 图3：逐10分钟购电花费 + 累计购电花费
# =============================================================================
def fig3_cost(d, out):
    fig, ax = plt.subplots(figsize=(12.8, 6.8))

    peak_cost = float(d["interval_cost"].max())
    ax.set_ylim(0, peak_cost * 1.23)

    add_day_night_background(ax)
    add_period_header(ax)

    bars = ax.bar(
        d["t_mid"], d["interval_cost"],
        width=DT_H * 0.88,
        color=C_SALMON, alpha=0.72,
        edgecolor="white", linewidth=0.25,
        label="逐10分钟购电花费", zorder=5
    )

    ax2 = ax.twinx()
    l_cum, = ax2.plot(
        np.r_[0.0, d["t_end"]],
        np.r_[0.0, d["cum_cost"]],
        color=C_NAVY, lw=2.95,
        label="当日累计购电花费", zorder=9
    )
    ax2.set_ylabel("当日累计购电花费 / 元", color=C_NAVY, fontsize=13)
    ax2.tick_params(axis="y", colors=C_NAVY)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(C_NAVY)
    ax2.set_ylim(0, float(d["total_cost"]) * 1.18)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.grid(False)

    # 最高单时段费用
    i_peak = int(np.argmax(d["interval_cost"]))
    x_peak = float(d["t_mid"][i_peak])
    y_peak = float(d["interval_cost"][i_peak])

    ax.scatter([x_peak], [y_peak], s=52, color="#B15549", edgecolor="white", linewidth=0.8, zorder=20)
    ax.annotate(
        f"单时段最高\n{y_peak:,.0f} 元\n({interval_text(i_peak)})",
        xy=(x_peak, y_peak),
        xytext=(min(x_peak + 0.9, 20.0), y_peak * 1.04),
        ha="left", va="bottom",
        fontsize=10.0, fontweight="bold", color="#A3473B",
        arrowprops=dict(arrowstyle="-|>", color="#A3473B", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.28", fc=BOX_RED, ec="#D5A39B", alpha=0.97),
    )

    # 全天总额：严格落在 24:00
    ax2.scatter([24.0], [d["total_cost"]], s=64, color=C_NAVY, edgecolor="white", linewidth=0.8, zorder=21, clip_on=False)
    ax2.annotate(
        f"全天合计\n{d['total_cost']:,.0f} 元",
        xy=(24.0, d["total_cost"]),
        xytext=(22.0, d["total_cost"] * 1.08),
        ha="center", va="center",
        fontsize=10.9, fontweight="bold", color=C_NAVY,
        arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.34", fc=BOX_BLUE, ec="#8DB6DA", alpha=0.97),
    )

    legend_box(
        ax,
        [bars, l_cum],
        ["逐10分钟购电花费", "当日累计购电花费"],
        loc="upper left",
    )

    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("逐10分钟购电花费 / 元")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    setup_axes(ax)
    ax.set_title("图3  逐10分钟购电花费与当日累计购电花费", pad=14)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.10)
    save_figure(fig, out)


# =============================================================================
# 主程序
# =============================================================================
def main():
    d = load_model_data()

    tasks = [
        ("fig1_input_power_price_award_v2.png", fig1_input),
        ("fig2_storage_energy_price_award_v2.png", fig2_storage),
        ("fig3_purchase_cost_award_v2.png", fig3_cost),
    ]

    print(f"[Q1] 使用正式模型：{Q1_PY}")
    print(f"[Q1] 使用输入数据：{INPUT_CSV}")
    print(f"[输出目录] {HERE}")

    for name, fn in tasks:
        target = HERE / name
        fn(d, target)
        print(f"  已生成：{name} 以及 {target.with_suffix('.pdf').name}")

    print("-" * 60)
    print(f"全天负荷电量：{d['total_load_kwh']:,.3f} kWh")
    print(f"全天光伏电量：{d['total_pv_kwh']:,.3f} kWh")
    print(f"全天购电量：{d['grid_kwh'].sum():,.3f} kWh")
    print(f"全天购电费用：{d['total_cost']:,.3f} 元")
    print(f"储能电量范围：{d['storage_kwh'].min():,.3f}–{d['storage_kwh'].max():,.3f} kWh")
    print("-" * 60)


if __name__ == "__main__":
    main()
