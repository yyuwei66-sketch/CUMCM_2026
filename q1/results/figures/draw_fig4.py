#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 图4：单时段充放电量分布图（无干涉终稿）
------------------------------------------------
修复：
1. 图例与标注框边界彻底避开 x=6 与 x=18 两条分界虚线；
2. 保持 Y 轴上限 1250 的通透感与统一排版规范。
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

OUT_PNG = HERE / "fig4_charge_discharge_final_accepted.png"
OUT_PDF = HERE / "fig4_charge_discharge_final_accepted.pdf"

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

raw_charge = np.asarray(solution["charge"], dtype=float)
raw_discharge = np.asarray(solution["discharge"], dtype=float)

if np.max(raw_charge) > 1500 or np.max(raw_discharge) > 1500:
    charge_kwh = raw_charge * DT_H
    discharge_kwh = raw_discharge * DT_H
else:
    charge_kwh = raw_charge
    discharge_kwh = raw_discharge

simultaneous_count = int(np.sum((charge_kwh > 1e-4) & (discharge_kwh > 1e-4)))
total_intervals = len(charge_kwh)


# =============================================================================
# 字体与排版设置
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
    "legend.fontsize": 13.5,
})


# =============================================================================
# 配色定义
# =============================================================================
C_CHARGE = "#1C4E78"        # 充电：深青蓝
C_DISCHARGE = "#C87B72"     # 放电：砖粉柔红
BG_NIGHT = "#D7EDFD"        # 夜间底色
BG_DAY = "#FFF7E6"          # 白天底色

C_BOUNDARY = "#959595"
C_GRID = "#C8D0D3"


# =============================================================================
# 画布排版
# =============================================================================
fig = plt.figure(figsize=(12.0, 7.4), dpi=260, facecolor="white")
ax = fig.add_axes([0.100, 0.145, 0.775, 0.730])


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
# 绘制充放电柱状图
# =============================================================================
bar_width = DT_H * 0.88

bar_charge = ax.bar(
    t_start + DT_H / 2.0,
    charge_kwh,
    width=bar_width,
    color=C_CHARGE,
    edgecolor="none",
    alpha=0.92,
    zorder=5,
    label=r"充电 $c_t$",
)

bar_discharge = ax.bar(
    t_start + DT_H / 2.0,
    -discharge_kwh,
    width=bar_width,
    color=C_DISCHARGE,
    edgecolor="none",
    alpha=0.92,
    zorder=5,
    label=r"放电 $d_t$（负向显示）",
)

# y=0 基准分界线
ax.axhline(0, color="#333333", linewidth=1.10, zorder=6)


# =============================================================================
# 坐标轴刻度与范围配置
# =============================================================================
ax.set_xlim(0, 24)
ax.set_ylim(-980, 1250)

ax.set_xlabel("时刻（h）", labelpad=12, fontsize=21)
ax.set_ylabel("单时段充放电量（kWh）", labelpad=14, fontsize=21)

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(250))

ax.tick_params(axis="both", which="major", length=5.2, width=0.95, labelsize=16.5)
ax.tick_params(axis="x", which="minor", length=3.2, width=0.70)

ax.grid(True, which="major", linestyle="--", linewidth=0.56, color=C_GRID, alpha=0.46)
ax.set_axisbelow(True)


# =============================================================================
# 右上角标注框：右移收窄，离开 18h 虚线
# =============================================================================
ax.text(
    23.75,
    1100,
    f"同时充放电时段：{simultaneous_count} / {total_intervals}",
    ha="right",
    va="center",
    fontsize=14.0,
    color="#1C4E78",
    bbox=dict(
        boxstyle="round,pad=0.30",
        facecolor="#F2F7FC",
        edgecolor="#9BBEE0",
        alpha=0.98,
        linewidth=0.8,
    ),
    zorder=20,
)


# =============================================================================
# 图例：收窄并微调至 (0.015, 0.97)，右边界收在 x=5.5h 内，不触碰 6h 虚线
# =============================================================================
legend = ax.legend(
    [bar_charge, bar_discharge],
    [r"充电 $c_t$", r"放电 $d_t$（负向显示）"],
    loc="upper left",
    bbox_to_anchor=(0.015, 0.97),
    ncol=2,
    frameon=True,
    fancybox=False,
    framealpha=0.96,
    facecolor="white",
    edgecolor="#C8C8C8",
    handlelength=1.1,
    columnspacing=0.8,
    borderpad=0.25,
    labelspacing=0.20,
    handletextpad=0.35,
)
legend.get_frame().set_linewidth(0.8)
legend.set_zorder(25)


# =============================================================================
# 保存
# =============================================================================
fig.savefig(OUT_PNG, dpi=700, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
plt.close(fig)

print(f"[Q1 图4] 同时充放电时段：{simultaneous_count} / {total_intervals}")
print(f"[输出 PNG] {OUT_PNG}")
print(f"[输出 PDF] {OUT_PDF}")
print(f"[字体] 使用字体：{chosen_font}")