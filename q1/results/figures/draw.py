#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C题第一问 输入/输出可视化绘图脚本。

读取 q1 目录下的中间结果 CSV，生成四张论文用图，全部平铺输出到本脚本所在
目录（q1/.figures/）：

    fig1_input_power_price.png      输入：电价 / 负载 / 光伏功率随时间变化
    fig2_battery_energy_price.png   输出：电池电量随时间变化（叠加电价副轴）
    fig3_purchase_cost.png          输出：逐时段购电花费与当日累计购电花费

数据来源（均由 q1 流程生成，本脚本不修改任何输入）：
    results/intermediate/dispatch_10min.csv   144 时段调度结果（含电价/负载/光伏/购电量/费用）
    results/intermediate/storage_states.csv   145 个边界时刻的储能电量
    results/intermediate/parameters.json      效率、容量上下限等模型参数

运行方式：
    python q1/.figures/draw.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境（服务器/CI）下也能出图
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # q1/.figures
Q1 = HERE.parent                                # q1
INTERIM = Q1 / "results" / "intermediate"

DISPATCH_CSV = INTERIM / "dispatch_10min.csv"
STORAGE_CSV = INTERIM / "storage_states.csv"
PARAMS_JSON = INTERIM / "parameters.json"

# 数据缺失时的兜底参数（与 parameters.json 一致）
DEFAULT_MIN_KWH, DEFAULT_MAX_KWH = 1200.0, 10800.0
DEFAULT_INIT_KWH = 6000.0

# --------------------------------------------------------------------------
# 中文字体：按可用性依次尝试，确保中文与负号都能正常显示
# --------------------------------------------------------------------------
_CJK_CANDIDATES = [
    "PingFang SC", "Heiti SC", "Heiti TC", "Hiragino Sans GB",
    "Songti SC", "STHeiti", "Arial Unicode MS", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei",
]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_cjk = [name for name in _CJK_CANDIDATES if name in _available]

plt.rcParams.update({
    "font.sans-serif": _cjk + ["DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,   # 中文字体缺少 U+2212，必须用 ASCII 减号
    "figure.dpi": 120,
    "savefig.dpi": 300,            # 论文用清晰度
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#BBBBBB",
    "lines.linewidth": 2.0,
    "figure.facecolor": "white",
})

# 配色（打印友好）
C_PRICE = "#C0392B"   # 电价
C_LOAD = "#1F4E79"    # 负载
C_PV = "#E8A33D"      # 光伏
C_ENERGY = "#1F7A5C"  # 电池电量
C_COST = "#B03A2E"    # 逐时段花费
C_CUM = "#1F4E79"     # 累计花费

if not _cjk:
    print("[警告] 未找到可用中文字体，中文可能显示为方块。")
else:
    print(f"[字体] 使用中文字体：{', '.join(_cjk[:3])}")


# --------------------------------------------------------------------------
# 读取数据
# --------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"缺少数据文件：{path}\n请先运行 q1 计算流程生成中间结果。")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_params() -> dict:
    if PARAMS_JSON.exists():
        return json.loads(PARAMS_JSON.read_text(encoding="utf-8-sig"))
    return {}


def _clock_minutes(label: str) -> int:
    """把 'HH:MM' 时刻标签转换为当日分钟数。"""
    hh, mm = label.strip().split(":")
    return int(hh) * 60 + int(mm)


def load_dispatch() -> dict:
    """读出 144 个 10 分钟时段的电价、功率与购电花费。"""
    rows = _read_csv(DISPATCH_CSV)
    if len(rows) != 144:
        raise SystemExit(f"调度结果应为 144 行，实际 {len(rows)} 行。")

    dt = 1.0 / 6.0  # 小时
    t, price, load_kw, pv_kw, cost = [], [], [], [], []
    for row in rows:
        t.append(_clock_minutes(row["interval_start"]) // 10)  # 时段索引 0..143
        price.append(float(row["price_yuan_per_kwh"]))
        load_kw.append(float(row["load_kwh"]) / dt)         # kWh -> kW（区间平均功率）
        pv_kw.append(float(row["pv_kwh"]) / dt)
        cost.append(float(row["purchase_cost_yuan"]))

    return {
        "t": t,
        "t_dense": [i / 6.0 for i in range(144)],           # 用于阶梯曲线的小时刻度
        "price": price,
        "load_kw": load_kw,
        "pv_kw": pv_kw,
        "cost": cost,
        "cum_cost": _cumsum(cost),
        "total_cost": math.fsum(cost),
    }


def load_storage() -> dict:
    """读出 145 个边界时刻的储能电量。"""
    rows = _read_csv(STORAGE_CSV)
    t_dense, energy = [], []
    for row in rows:
        t_dense.append(int(row["minute"]) / 60.0)           # 分钟 -> 小时
        energy.append(float(row["stored_energy_kwh"]))
    if len(energy) != 145:
        raise SystemExit(f"储能状态应为 145 行，实际 {len(energy)} 行。")
    return {"t_dense": t_dense, "energy": energy}


def _cumsum(values: list[float]) -> list[float]:
    out, run = [], 0.0
    for v in values:
        run += v
        out.append(run)
    return out


def _step_xy(t_dense: list[float], values: list[float]):
    """构造阶梯（前向保持）曲线的坐标。"""
    x, y = [t_dense[0]], [values[0]]
    for i in range(1, len(values)):
        x.append(t_dense[i])
        y.append(values[i - 1])
        x.append(t_dense[i])
        y.append(values[i])
    return x, y


def _hour_axis(ax, xmax: float = 24.0) -> None:
    """统一横轴：0-24 时，每 2 小时一个主刻度。"""
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))


def _tag(ax, text: str) -> None:
    """把说明文字放在坐标区之外的下方，避免压住曲线。"""
    ax.text(0.0, -0.155, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color="#555555")


def _headroom(ax, ymin: float, ymax: float, ratio: float = 0.28) -> None:
    """在数据上方预留空白，使图内标注不与曲线重叠。"""
    ax.set_ylim(ymin, ymax + (ymax - ymin) * ratio)


# --------------------------------------------------------------------------
# 图1：输入 —— 电价、负载、光伏功率随时间变化
# --------------------------------------------------------------------------
def fig_input(d, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    # 左轴：功率（负载 / 光伏）
    x_load, y_load = _step_xy(d["t_dense"], d["load_kw"])
    x_pv, y_pv = _step_xy(d["t_dense"], d["pv_kw"])
    ax.plot(x_load, y_load, color=C_LOAD, lw=2.2, label="小区负载功率")
    ax.plot(x_pv, y_pv, color=C_PV, lw=2.2, label="光伏预测功率")
    ax.fill_between(x_pv, 0, y_pv, color=C_PV, alpha=0.22)
    ax.fill_between(x_load, 0, y_load, color=C_LOAD, alpha=0.08)
    ax.set_xlabel("时间（时）")
    ax.set_ylabel("功率（kW）")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    _hour_axis(ax)

    # 电价曲线在右上区域，图例置于左下角空白处，避免遮挡负载/光伏曲线
    ax2 = ax.twinx()
    x_p, y_p = _step_xy(d["t_dense"], d["price"])
    ax2.plot(x_p, y_p, color=C_PRICE, lw=1.7, ls="-.", alpha=0.95, label="购电电价")
    ax2.set_ylabel("购电电价（元/kWh）", color=C_PRICE)
    ax2.tick_params(axis="y", colors=C_PRICE)
    ax2.set_ylim(0, max(d["price"]) * 1.55)
    ax2.grid(False)

    # 在曲线上方预留空白，供区间标注使用
    _headroom(ax, 0, max(max(d["load_kw"]), max(d["pv_kw"])), 0.30)
    top = ax.get_ylim()[1]

    # 标注光伏超过负载的时段（可充电的剩余电量来源）
    surplus = [i for i in range(144) if d["pv_kw"][i] > d["load_kw"][i]]
    if surplus:
        i0, i1 = surplus[0] / 6.0, (surplus[-1] + 1) / 6.0
        mid = (i0 + i1) / 2
        ax.axvspan(i0, i1, color=C_PV, alpha=0.09, zorder=0)
        # 说明文字放在预留的顶部空白里，箭头指向阴影带本身，不压曲线
        ax.annotate(
            f"光伏超过负载（{i0:.0f}:00-{i1:.0f}:00），剩余电量可供储能充电",
            xy=(mid, max(d["pv_kw"]) * 1.02),
            xytext=(mid, top * 0.995),
            ha="center", va="top", fontsize=10.5, color="#8A5A00",
            arrowprops=dict(arrowstyle="->", color="#8A5A00", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFF7E6", ec="#E8A33D", alpha=0.95),
        )

    # 合并图例：放在左下角，该区域无曲线经过
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", ncol=1, fontsize=10.5)

    ax.set_title("图1  典型日输入：电价、小区负载与光伏预测功率")
    _tag(ax, f"全天负载电量 {math.fsum(d['load_kw'])/6:,.0f} kWh"
             f"｜光伏电量 {math.fsum(d['pv_kw'])/6:,.0f} kWh")
    fig.savefig(path)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# 电池电量曲线（图2：叠加电价副轴）
# --------------------------------------------------------------------------
def _draw_energy(ax, st, params) -> None:
    lo = float(params.get("storage_min_kwh", DEFAULT_MIN_KWH))
    hi = float(params.get("storage_max_kwh", DEFAULT_MAX_KWH))
    init = float(params.get("initial_storage_kwh", DEFAULT_INIT_KWH))
    e = st["energy"]
    t = st["t_dense"]

    ax.axhspan(lo, hi, color="#1F7A5C", alpha=0.055, zorder=0)
    ax.axhline(hi, color="#999999", ls=":", lw=1.3)
    ax.axhline(lo, color="#999999", ls=":", lw=1.3)
    ax.axhline(init, color="#888888", ls="--", lw=1.1, alpha=0.8)

    ax.plot(t, e, color=C_ENERGY, lw=2.5, zorder=5)
    ax.fill_between(t, e, lo, color=C_ENERGY, alpha=0.16, zorder=1)

    # 上下限说明放在坐标区之外的右侧留白，避免压住电量曲线
    ax.text(1.012, hi, f"容量上限 {hi:,.0f} kWh", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=9.5, color="#666666")
    ax.text(1.012, lo, f"容量下限 {lo:,.0f} kWh", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=9.5, color="#666666")

    ax.set_xlabel("时间（时）")
    ax.set_ylabel("电池电量（kWh）")
    ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.20)
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))


# --------------------------------------------------------------------------
# 图2：输出 —— 电池电量随时间变化（叠加电价副轴）
# --------------------------------------------------------------------------
def fig_energy_price(d, st, params, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    _draw_energy(ax, st, params)

    ax2 = ax.twinx()
    x_p, y_p = _step_xy(d["t_dense"], d["price"])
    ax2.plot(x_p, y_p, color=C_PRICE, lw=1.6, ls="-.", alpha=0.95, label="购电电价")
    ax2.set_ylabel("购电电价（元/kWh）", color=C_PRICE)
    ax2.tick_params(axis="y", colors=C_PRICE)
    # 电价轴留出较大上限，使电价曲线整体低于电量曲线，两条线少交叉
    ax2.set_ylim(0, max(d["price"]) * 1.75)
    ax2.grid(False)

    # 两条曲线会占满右上角；图例放在正上方、曲线之上，不压数据
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=C_ENERGY, lw=2.5, label="电池电量（左轴）"),
        Line2D([], [], color=C_PRICE, lw=1.6, ls="-.", label="购电电价（右轴）"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=10.5)

    ax.set_title("图2  电池电量与购电电价对照（低价时段充电、高价时段放电）")
    _tag(ax, "电量在低价时段抬升，在早、晚高价时段回落")
    # 右侧留白用于容量上下限文字，并容纳电价轴标签
    fig.subplots_adjust(right=0.845)
    fig.savefig(path)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# 图4：输出 —— 逐时段购电花费与当日累计购电花费
# --------------------------------------------------------------------------
def fig_cost(d, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.4))

    bars = ax.bar(d["t_dense"], d["cost"], width=1.0 / 6.0 * 0.92,
                  color=C_COST, alpha=0.62, label="逐时段购电花费", zorder=3)
    ax.set_xlabel("时间（时）")
    ax.set_ylabel("逐时段购电花费（元）")
    ax.set_ylim(bottom=0)
    _hour_axis(ax)

    # 在柱子上方预留空白，供图例使用
    peak = max(d["cost"])
    _headroom(ax, 0, peak, 0.34)

    ax2 = ax.twinx()
    ax2.plot(d["t_dense"], d["cum_cost"], color=C_CUM, lw=2.6, label="当日累计购电花费")
    ax2.set_ylabel("当日累计购电花费（元）", color=C_CUM)
    ax2.tick_params(axis="y", colors=C_CUM)
    ax2.set_ylim(0, d["total_cost"] * 1.32)
    ax2.grid(False)

    # 终点标注全天总额：文字放在右上角留白，箭头指向累计曲线终点
    ax2.plot([24.0], [d["total_cost"]], "o", ms=6.5, color=C_CUM, zorder=6)
    ax2.annotate(f"全天合计 {d['total_cost']:,.0f} 元", xy=(24.0, d["total_cost"]),
                 xytext=(23.4, d["total_cost"] * 1.27), ha="right", va="top",
                 fontsize=10.5, color=C_CUM,
                 bbox=dict(boxstyle="round,pad=0.32", fc="#EAF0F7", ec="#9DB8D6", alpha=0.96),
                 arrowprops=dict(arrowstyle="->", color=C_CUM, lw=1.0, shrinkB=3))

    # 图例放在左下角：该处柱子较低，且累计曲线尚未升起
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10.5)

    ax.set_title("图3  逐时段购电花费与当日累计购电花费")
    _tag(ax, f"优化方案全天购电费 {d['total_cost']:,.2f} 元")
    fig.subplots_adjust(right=0.88)
    fig.savefig(path)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------
def main() -> None:
    params = _load_params()
    d = load_dispatch()
    st = load_storage()

    outputs = [
        ("fig1_input_power_price.png", lambda p: fig_input(d, p)),
        ("fig2_battery_energy_price.png", lambda p: fig_energy_price(d, st, params, p)),
        ("fig3_purchase_cost.png", lambda p: fig_cost(d, p)),
    ]

    print(f"[输出目录] {HERE}")
    for name, draw in outputs:
        target = HERE / name
        draw(target)
        print(f"  已生成 {name}  ({target.stat().st_size / 1024:.0f} KB, 300 dpi)")

    print(f"[核对] 全天购电费 {d['total_cost']:,.4f} 元，"
          f"储能电量区间 {min(st['energy']):,.0f}-{max(st['energy']):,.0f} kWh")


if __name__ == "__main__":
    main()
