#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 图2最终版代码：电池电量与购电电价对照图
------------------------------------------------
修改点：
- 严格将图例框定位于 [Y=0, Y=1800] 的空白带内；
- 上边缘不碰绿色曲线波谷，下边缘完全收在坐标轴框内；
- 紧凑图例尺寸，左右不遮挡右下角说明文字。
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

OUT_PNG = HERE / "fig2_battery_energy_price_final_accepted.png"
OUT_PDF = HERE / "fig2_battery_energy_price_final_accepted.pdf"

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

DT_H = 1.0 / 6.0  # 10 分钟
interval_start_h = source["interval_start_minute"].to_numpy(dtype=float) / 60.0
price = source["price_yuan_per_kwh"].to_numpy(dtype=float)
storage = np.asarray(solution["storage"], dtype=float)
time_storage = np.arange(len(storage), dtype=float) * DT_H

storage_min = float(cfg["storage_min_kwh"])
storage_max = float(cfg["storage_max_kwh"])
initial_storage = float(cfg["initial_storage_kwh"])


def step_xy(t: np.ndarray, values: np.ndarray):
    """把 144 段数据拓展成 post-step 画法需要的 x/y。"""
    x = np.r_[t, 24.0]
    y = np.r_[values, values[-1]]
    return x, y


# =============================================================================
# 字体选择
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
# 配色
# =============================================================================
C_STORAGE = "#0A9352"       # 主绿线
C_STORAGE_FILL = "#BFE8C3"  # 储能区浅绿
C_PRICE = "#FF1A12"         # 电价红线

BG_NIGHT = "#D7EDFD"        # 夜间浅蓝
BG_DAY = "#FFF7E6"          # 白天浅米黄

C_BOUNDARY = "#959595"
C_GRID = "#C8D0D3"
C_LIMIT = "#9BB09F"
C_INIT = "#B3B3B3"
TEXT_GRAY = "#595959"


# =============================================================================
# 画布
# =============================================================================
fig = plt.figure(figsize=(12.0, 8.4), dpi=260, facecolor="white")
ax = fig.add_axes([0.100, 0.120, 0.780, 0.720])


# =============================================================================
# 背景时段
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
# 参考线
# =============================================================================
ax.axhline(storage_max, color=C_LIMIT, linestyle=":", linewidth=0.95, alpha=0.82, zorder=2)
ax.axhline(storage_min, color=C_LIMIT, linestyle=":", linewidth=0.95, alpha=0.82, zorder=2)
ax.axhline(initial_storage, color=C_INIT, linestyle="--", linewidth=1.00, alpha=0.70, zorder=2)


# =============================================================================
# 左轴：储能电量
# =============================================================================
line_storage, = ax.plot(
    time_storage,
    storage,
    color=C_STORAGE,
    linewidth=3.15,
    solid_capstyle="round",
    solid_joinstyle="round",
    antialiased=True,
    zorder=10,
    label="电池电量（左轴）",
)

ax.fill_between(
    time_storage,
    storage_min,
    storage,
    where=(storage >= storage_min),
    color=C_STORAGE_FILL,
    alpha=0.34,
    linewidth=0,
    interpolate=True,
    zorder=3,
)


# =============================================================================
# 右轴：电价
# =============================================================================
x_price, y_price = step_xy(interval_start_h, price)
ax2 = ax.twinx()

line_price, = ax2.step(
    x_price,
    y_price,
    where="post",
    color=C_PRICE,
    linewidth=2.05,
    linestyle="-.",
    antialiased=True,
    zorder=11,
    label="购电电价（右轴）",
)


# =============================================================================
# 坐标轴设置
# =============================================================================
ax.set_xlim(0, 24)
ax.set_ylim(0, 12100)
ax2.set_ylim(0.0, 2.03)

ax.set_xlabel("时刻（h）", labelpad=12, fontsize=21)
ax.set_ylabel("电池电量（kWh）", labelpad=6, fontsize=21)
ax2.set_ylabel("购电电价（元/kWh）", color=C_PRICE, labelpad=20, fontsize=21)

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(2000))
ax2.yaxis.set_major_locator(MultipleLocator(0.5))
ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

ax.tick_params(axis="both", which="major", length=5.2, width=0.95, labelsize=16.5)
ax.tick_params(axis="x", which="minor", length=3.2, width=0.70)
ax2.tick_params(axis="y", which="major", colors=C_PRICE, length=5.2, width=0.95, labelsize=16.5)

ax.grid(True, which="major", linestyle="--", linewidth=0.56, color=C_GRID, alpha=0.46)
ax.set_axisbelow(True)

ax2.spines["right"].set_color(C_PRICE)
ax2.spines["right"].set_linewidth(1.10)
ax2.spines["top"].set_visible(False)


# =============================================================================
# 上下限说明文字
# =============================================================================
ax.text(
    23.72,
    storage_max + 350,
    f"容量上限 {storage_max:,.0f} kWh",
    ha="right",
    va="center",
    fontsize=16.5,
    color=TEXT_GRAY,
    zorder=20,
)

ax.text(
    23.72,
    storage_min - 350,
    f"容量下限 {storage_min:,.0f} kWh",
    ha="right",
    va="center",
    fontsize=16.5,
    color=TEXT_GRAY,
    zorder=20,
)


# =============================================================================
# 图例（精确控制：不出底边线、不碰绿色波谷、不压右侧说明）
# =============================================================================
legend = ax.legend(
    [line_storage, line_price],
    ["电池电量（左轴）", "购电电价（右轴）"],
    loc="lower center",
    bbox_to_anchor=(0.52, 0.025),
    ncol=2,
    frameon=True,
    fancybox=False,
    framealpha=0.92,
    facecolor="white",
    edgecolor="#C8C8C8",
    handlelength=1.8,
    columnspacing=1.2,
    borderpad=0.22,
    labelspacing=0.10,
    handletextpad=0.45,
)
legend.get_frame().set_linewidth(0.7)
legend.set_zorder(25)


# =============================================================================
# 保存
# =============================================================================
fig.savefig(OUT_PNG, dpi=700, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
plt.close(fig)

print(f"[Q1 图2] 使用正式模型：{Q1_PY}")
print(f"[Q1 图2] 使用输入数据：{INPUT_CSV}")
print(f"[输出 PNG] {OUT_PNG}")
print(f"[输出 PDF] {OUT_PDF}")
print(f"[字体] 使用字体：{chosen_font}")
print("[完成] 图例已微调至表格底部内侧黄金空白区。")

