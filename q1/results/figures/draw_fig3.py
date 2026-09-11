#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 图3：逐时段购电花费与当日累计购电花费（完美交付终稿版）
------------------------------------------------------------
本次微调：
1. 修正“全天合计”箭头指向，使其精确指向曲线终点散点，且不压边框；
2. 调整左下角说明文字左对齐基准，与最左侧 Y 轴标签边缘垂直对齐。
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import numpy as np
except ModuleNotFoundError as e:
    missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
    raise SystemExit(
        f"缺少依赖：{missing}\n"
        f"请先安装：pip install matplotlib numpy"
    )


# =============================================================================
# 路径设置
# =============================================================================
HERE = Path(__file__).resolve().parent
Q1_DIR = HERE.parents[1]
Q1_PY = Q1_DIR / "q1.py"
INPUT_CSV = Q1_DIR / "input" / "q1_typical_day.csv"

OUT_PNG = HERE / "fig3_purchase_cost_final_accepted.png"
OUT_PDF = HERE / "fig3_purchase_cost_final_accepted.pdf"

if not Q1_PY.exists():
    raise SystemExit(f"找不到 q1.py：{Q1_PY}")
if not INPUT_CSV.exists():
    raise SystemExit(f"找不到输入文件：{INPUT_CSV}")


# =============================================================================
# 导入正式模型并求解
# =============================================================================
spec = importlib.util.spec_from_file_location("q1_model", Q1_PY)
q1_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(q1_model)

cfg, source, solution, checks = q1_model.solve_from_input(INPUT_CSV, "auto")

DT_H = 1.0 / 6.0  # 10 分钟 (h)
t_start = source["interval_start_minute"].to_numpy(dtype=float) / 60.0
price = source["price_yuan_per_kwh"].to_numpy(dtype=float)

# 获取购电功率并计算花费
grid_power = np.asarray(solution["grid"], dtype=float)
grid_import = np.maximum(grid_power, 0.0)

raw_step_cost = grid_import * price
total_cost_model = float(solution.get("cost", 35126.95))

if abs(np.sum(raw_step_cost) - total_cost_model) < abs(np.sum(raw_step_cost * DT_H) - total_cost_model):
    interval_cost = raw_step_cost
else:
    interval_cost = raw_step_cost * DT_H

cum_cost = np.r_[0.0, np.cumsum(interval_cost)]
cum_time = np.arange(len(cum_cost), dtype=float) * DT_H
total_cost = float(cum_cost[-1])


# =============================================================================
# 字体与全局样式设置
# =============================================================================
font_candidates = [
    "STSong",
    "SimSun",
    "Songti SC",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Microsoft YaHei",
    "SimHei",
]

available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
chosen_font = next((f for f in font_candidates if f in available_fonts), "DejaVu Sans")

plt.rcParams.update({
    "font.family": chosen_font,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#262626",
    "axes.linewidth": 1.12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 21,
    "xtick.labelsize": 16.5,
    "ytick.labelsize": 16.5,
    "legend.fontsize": 14.5,
})


# =============================================================================
# 配色定义
# =============================================================================
C_BAR = "#C87B72"           # 逐时段花费条形柔红
C_LINE = "#1C4E78"          # 累计折线深海蓝
BG_NIGHT = "#D7EDFD"        # 夜间浅蓝
BG_DAY = "#FFF7E6"          # 白天浅米黄

C_BOUNDARY = "#959595"
C_GRID = "#C8D0D3"
TEXT_GRAY = "#555555"


# =============================================================================
# 画布排版
# =============================================================================
fig = plt.figure(figsize=(12.0, 7.5), dpi=260, facecolor="white")
ax = fig.add_axes([0.100, 0.155, 0.775, 0.730])


# =============================================================================
# 背景色块
# =============================================================================
ax.axvspan(0, 6, facecolor=BG_NIGHT, alpha=1.0, zorder=-30)
ax.axvspan(6, 18, facecolor=BG_DAY, alpha=1.0, zorder=-30)
ax.axvspan(18, 24, facecolor=BG_NIGHT, alpha=1.0, zorder=-30)

for x in (6, 18):
    ax.axvline(
        x,
        color=C_BOUNDARY,
        linestyle="--",
        linewidth=1.00,
        alpha=0.55,
        zorder=2,
    )


# =============================================================================
# 左轴：逐时段购电花费（柱状图）
# =============================================================================
bar_width = DT_H * 0.88
bars = ax.bar(
    t_start + DT_H / 2.0,
    interval_cost,
    width=bar_width,
    color=C_BAR,
    edgecolor="none",
    alpha=0.90,
    zorder=5,
    label="逐时段购电花费",
)


# =============================================================================
# 右轴：累计购电花费（折线）
# =============================================================================
ax2 = ax.twinx()

line_cum, = ax2.plot(
    cum_time,
    cum_cost,
    color=C_LINE,
    linewidth=3.2,
    solid_capstyle="round",
    solid_joinstyle="round",
    zorder=12,
    label="当日累计购电花费",
)

ax2.scatter(
    [cum_time[-1]],
    [total_cost],
    color=C_LINE,
    s=44,
    zorder=15,
)


# =============================================================================
# 坐标轴刻度与标签
# =============================================================================
ax.set_xlim(0, 24)
ax.set_ylim(0, 1150)

ax2.set_xlim(0, 24)
ax2.set_ylim(0, 48000)

ax.set_xlabel("时刻（h）", labelpad=12, fontsize=21)
ax.set_ylabel("逐时段购电花费（元）", labelpad=14, fontsize=21)
ax2.set_ylabel("当日累计购电花费（元）", color=C_LINE, labelpad=18, fontsize=21)

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(200))
ax2.yaxis.set_major_locator(MultipleLocator(10000))
ax2.yaxis.set_major_formatter(FormatStrFormatter("%d"))

ax.tick_params(axis="both", which="major", length=5.2, width=0.95, labelsize=16.5)
ax.tick_params(axis="x", which="minor", length=3.2, width=0.70)
ax2.tick_params(axis="y", which="major", colors=C_LINE, length=5.2, width=0.95, labelsize=16.5)

ax.grid(True, which="major", linestyle="--", linewidth=0.56, color=C_GRID, alpha=0.46)
ax.set_axisbelow(True)

ax2.spines["right"].set_color(C_LINE)
ax2.spines["right"].set_linewidth(1.10)
ax2.spines["top"].set_visible(False)


# =============================================================================
# 全天合计标注框（精准指引末端散点，无遮挡与出界）
# =============================================================================
ax2.annotate(
    f"全天合计 {total_cost:,.0f} 元",
    xy=(23.90, total_cost),          # 精准指向终点圆点边缘
    xytext=(20.6, 43200),           # 适度调高文本框，形成优雅的俯指角度
    ha="center",
    va="center",
    fontsize=15.5,
    color=C_LINE,
    bbox=dict(
        boxstyle="round,pad=0.38",
        facecolor="#F2F7FC",
        edgecolor="#9BBEE0",
        alpha=0.98,
        linewidth=0.8,
    ),
    arrowprops=dict(
        arrowstyle="->",
        color=C_LINE,
        lw=1.2,
        shrinkA=3,
        shrinkB=4,                  # 箭头自然停在小圆点边缘
    ),
    zorder=22,
)


# =============================================================================
# 图例
# =============================================================================
legend = ax.legend(
    [bars, line_cum],
    ["逐时段购电花费", "当日累计购电花费"],
    loc="upper left",
    bbox_to_anchor=(0.02, 0.965),
    ncol=1,
    frameon=True,
    fancybox=False,
    framealpha=0.92,
    facecolor="white",
    edgecolor="#C8C8C8",
    handlelength=1.8,
    borderpad=0.40,
    labelspacing=0.32,
    handletextpad=0.50,
)
legend.get_frame().set_linewidth(0.8)
legend.set_zorder(25)


# =============================================================================
# 左下角说明文本（对齐最左侧 Y 轴标签外边缘）
# =============================================================================
fig.text(
    0.055,
    0.048,
    f"优化方案全天购电费 {total_cost:,.2f} 元",
    ha="left",
    va="center",
    fontsize=15.0,
    color=TEXT_GRAY,
)


# =============================================================================
# 保存
# =============================================================================
fig.savefig(OUT_PNG, dpi=700, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
plt.close(fig)

print(f"[Q1 图3] 优化方案全天购电费：{total_cost:,.2f} 元")
print(f"[输出 PNG] {OUT_PNG}")
print(f"[输出 PDF] {OUT_PDF}")