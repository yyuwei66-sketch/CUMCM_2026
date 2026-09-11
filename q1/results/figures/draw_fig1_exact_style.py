#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 图1：按已确认的视觉样式，用真实数据重绘。

建议放置：
    q1/results/figures/draw_fig1_exact_style.py

运行：
    python q1/results/figures/draw_fig1_exact_style.py

输出：
    q1/results/figures/fig1_exact_style.png
    q1/results/figures/fig1_exact_style.pdf

说明：
- 不使用手工伪造曲线；
- 负荷、光伏、电价、总电量均直接来自 q1.py 正式数据源；
- “光伏出力超过负荷”的时间区间由真实 10 min 数据自动计算；
- 不在图顶部添加标题，也不在图底部添加说明文字。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np


# =============================================================================
# 路径
# =============================================================================
HERE = Path(__file__).resolve().parent          # q1/results/figures
Q1 = HERE.parents[1]                           # q1
Q1_PY = Q1 / "q1.py"
INPUT_CSV = Q1 / "input" / "q1_typical_day.csv"

OUT_PNG = HERE / "fig1_exact_style.png"
OUT_PDF = HERE / "fig1_exact_style.pdf"

if not Q1_PY.exists():
    raise SystemExit(f"找不到 q1.py：{Q1_PY}")
if not INPUT_CSV.exists():
    raise SystemExit(f"找不到输入文件：{INPUT_CSV}")


# =============================================================================
# 直接复用 q1.py，确保数据与正式 Q1 结果同源
# =============================================================================
spec = importlib.util.spec_from_file_location("q1_model", Q1_PY)
q1_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(q1_model)

cfg, source, solution, checks = q1_model.solve_from_input(INPUT_CSV, "auto")

if len(source) != 144:
    raise RuntimeError(f"预期 144 个 10 分钟区间，实际得到 {len(source)} 个。")


# =============================================================================
# 数据
# =============================================================================
DT_H = 1.0 / 6.0

t_start = source["interval_start_minute"].to_numpy(dtype=float) / 60.0
t_end = source["interval_end_minute"].to_numpy(dtype=float) / 60.0

load_kw = source["load_kw"].to_numpy(dtype=float)
pv_kw = source["pv_forecast_kw"].to_numpy(dtype=float)
price = source["price_yuan_per_kwh"].to_numpy(dtype=float)

total_load_kwh = float(np.sum(load_kw) * DT_H)
total_pv_kwh = float(np.sum(pv_kw) * DT_H)


def step_xy(t, values):
    """10 min 区间值按前向保持阶梯画至 24:00。"""
    x = np.r_[t, 24.0]
    y = np.r_[values, values[-1]]
    return x, y


def fmt_hhmm(hour: float) -> str:
    minute = int(round(hour * 60))
    minute = max(0, min(1440, minute))
    if minute == 1440:
        return "24:00"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def contiguous_intervals(mask):
    """把 True 的 10 min 区间合并为连续时间段。"""
    intervals = []
    start_idx = None

    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            intervals.append((start_idx * DT_H, i * DT_H))
            start_idx = None

    if start_idx is not None:
        intervals.append((start_idx * DT_H, len(mask) * DT_H))

    return intervals


# =============================================================================
# 字体：优先宋体，让整体更接近论文插图
# =============================================================================
font_candidates = [
    "SimSun",
    "Songti SC",
    "STSong",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Microsoft YaHei",
    "SimHei",
]
available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
chosen = next((f for f in font_candidates if f in available_fonts), "DejaVu Sans")

plt.rcParams.update({
    "font.family": chosen,
    "axes.unicode_minus": False,

    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.05,

    "axes.labelsize": 15,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,

    "legend.fontsize": 12.5,

    "savefig.facecolor": "white",
})


# =============================================================================
# 精确复刻的视觉参数
# =============================================================================
C_LOAD = "#0B4F8A"       # 深蓝
C_PV = "#F28A00"         # 橙
C_PRICE = "#F21C12"      # 红
C_NOTE = "#8B5A00"       # 棕金

BG_NIGHT = "#DBEFFE"     # 浅蓝
BG_DAY = "#FDF3DE"       # 浅米黄
BG_SURPLUS = "#F8DFA7"   # 光伏盈余区间更深一点

GRID = "#B9C1C8"
BOUNDARY = "#707070"


# =============================================================================
# 绘图
# =============================================================================

# 4:3 画布，与确认的样图一致；plot 本体缩短并留较多白边
fig = plt.figure(figsize=(12, 9), dpi=160, facecolor="white")

# [left, bottom, width, height]
# 这组尺寸专门用于复刻你确认的版式比例
ax = fig.add_axes([0.080, 0.195, 0.810, 0.660])

# ---------------------------
# 昼夜背景
# ---------------------------
ax.axvspan(0, 6, facecolor=BG_NIGHT, alpha=1.0, zorder=-30)
ax.axvspan(6, 18, facecolor=BG_DAY, alpha=1.0, zorder=-30)
ax.axvspan(18, 24, facecolor=BG_NIGHT, alpha=1.0, zorder=-30)

# 6:00、18:00 分界严格落在真实 x 坐标
for x in (6, 18):
    ax.axvline(
        x,
        color=BOUNDARY,
        linestyle="--",
        linewidth=1.0,
        alpha=0.95,
        zorder=2,
    )

# ---------------------------
# 光伏 > 负荷的真实区间
# ---------------------------
surplus_mask = pv_kw > load_kw
surplus_intervals = contiguous_intervals(surplus_mask)

# 若有多个区间，取持续时间最长的一个作为主标注区间
main_surplus = None
if surplus_intervals:
    main_surplus = max(
        surplus_intervals,
        key=lambda ab: ab[1] - ab[0]
    )
    a, b = main_surplus

    ax.axvspan(
        a, b,
        facecolor=BG_SURPLUS,
        alpha=0.30,
        zorder=-20,
    )

# ---------------------------
# 三条真实曲线
# ---------------------------
x_load, y_load = step_xy(t_start, load_kw)
x_pv, y_pv = step_xy(t_start, pv_kw)
x_price, y_price = step_xy(t_start, price)

line_load, = ax.step(
    x_load,
    y_load,
    where="post",
    color=C_LOAD,
    linewidth=2.45,
    label="系统负荷",
    zorder=8,
)

line_pv, = ax.step(
    x_pv,
    y_pv,
    where="post",
    color=C_PV,
    linewidth=2.55,
    label="光伏出力预测",
    zorder=9,
)

# ---------------------------
# 左轴
# ---------------------------
left_max_data = max(float(load_kw.max()), float(pv_kw.max()))

# 真实数据约 7.6 MW 时，保持和样图一样：顶部 9000 刻度，上方再留少量空间
y_top = max(
    9400.0,
    np.ceil(left_max_data / 1000.0) * 1000.0 + 1000.0
)

ax.set_xlim(0, 24)
ax.set_ylim(0, y_top)

ax.set_xlabel("时刻 (h)", labelpad=9)
ax.set_ylabel("功率 (kW)", labelpad=9)

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1000))

ax.tick_params(axis="both", which="major", length=5, width=0.9)
ax.tick_params(axis="x", which="minor", length=3, width=0.7)

ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.65,
    color=GRID,
    alpha=0.48,
)
ax.set_axisbelow(True)

# ---------------------------
# 右轴：电价
# ---------------------------
ax2 = ax.twinx()

line_price, = ax2.step(
    x_price,
    y_price,
    where="post",
    color=C_PRICE,
    linewidth=1.75,
    linestyle="-.",
    label="购电电价",
    zorder=10,
)

ax2.set_ylabel(
    "购电电价 (元/kWh)",
    color=C_PRICE,
    labelpad=13,
)
ax2.tick_params(
    axis="y",
    colors=C_PRICE,
    length=4,
    width=0.9,
)

# 样图固定显示到约 2.00；如果未来价格超过 2，则自动扩展
price_top = max(
    2.08,
    np.ceil(float(price.max()) / 0.25) * 0.25 + 0.08,
)
ax2.set_ylim(0, price_top)
ax2.yaxis.set_major_locator(MultipleLocator(0.25))
ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

ax2.spines["right"].set_color(C_PRICE)
ax2.spines["right"].set_linewidth(1.05)
ax2.spines["top"].set_visible(False)

# ---------------------------
# 图例：左上角
# ---------------------------
legend = ax.legend(
    [line_load, line_pv, line_price],
    ["系统负荷", "光伏出力预测", "购电电价"],
    loc="upper left",
    bbox_to_anchor=(0.012, 0.975),
    borderaxespad=0.0,
    frameon=True,
    fancybox=True,
    framealpha=0.96,
    facecolor="white",
    edgecolor="#B5B5B5",
    handlelength=2.6,
    borderpad=0.62,
    labelspacing=0.45,
)
legend.get_frame().set_linewidth(0.9)

# ---------------------------
# 顶部真实盈余区间标注
# ---------------------------
if main_surplus is not None:
    a, b = main_surplus
    mid = (a + b) / 2.0

    # 箭头指向区间中部对应的光伏曲线，而不是手写虚假位置
    idx_mid = int(np.argmin(np.abs((t_start + DT_H / 2.0) - mid)))
    arrow_y = float(pv_kw[idx_mid]) + 80.0

    label = (
        f"光伏出力超过负荷"
        f"（{fmt_hhmm(a)}–{fmt_hhmm(b)}）"
    )

    ax.annotate(
        label,
        xy=(mid, arrow_y),
        xytext=(mid, y_top * 0.948),
        ha="center",
        va="center",
        fontsize=12.0,
        color="#3B2B10",
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_NOTE,
            linewidth=1.25,
            shrinkA=3,
            shrinkB=4,
        ),
        bbox=dict(
            boxstyle="round,pad=0.32",
            facecolor="#FFF8EB",
            edgecolor=C_PV,
            linewidth=1.0,
            alpha=0.98,
        ),
        zorder=30,
    )

# ---------------------------
# 右下角真实日电量信息
# ---------------------------
info = (
    f"全天负载电量 {total_load_kwh:,.0f} kWh"
    f" | 光伏电量 {total_pv_kwh:,.0f} kWh"
)

ax.text(
    23.55,
    y_top * 0.047,
    info,
    ha="right",
    va="center",
    fontsize=11.0,
    color="#222222",
    bbox=dict(
        boxstyle="round,pad=0.42",
        facecolor="white",
        edgecolor="#A8A8A8",
        linewidth=0.9,
        alpha=0.96,
    ),
    zorder=30,
)

# 不加标题；不加图片下方说明段落

# =============================================================================
# 保存
# =============================================================================
fig.savefig(
    OUT_PNG,
    dpi=360,
    facecolor="white",
    bbox_inches=None,          # 保留固定 4:3 白边与尺寸
)

fig.savefig(
    OUT_PDF,
    facecolor="white",
    bbox_inches=None,
)

plt.close(fig)

print(f"[完成] PNG：{OUT_PNG}")
print(f"[完成] PDF：{OUT_PDF}")
print(f"[核对] 全天负载电量：{total_load_kwh:,.3f} kWh")
print(f"[核对] 全天光伏电量：{total_pv_kwh:,.3f} kWh")

if main_surplus is not None:
    print(
        "[核对] 最长光伏出力超过负荷区间："
        f"{fmt_hhmm(main_surplus[0])}–{fmt_hhmm(main_surplus[1])}"
    )
else:
    print("[核对] 当日不存在光伏出力超过负荷的连续区间。")
