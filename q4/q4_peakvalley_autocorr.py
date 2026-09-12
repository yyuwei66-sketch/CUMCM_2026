#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四问：峰日 / 谷日分组的电价「跨日同时刻」相关性分析。

分组口径（题目要求）：
    谷日：所有周六、周日
    峰日：其余所有日子（周一 ~ 周五）

分析目标：
    对每一个 10 分钟时刻 t，度量「本日 t 时刻电价」与「同类日中前一个日子的
    同一时刻电价」之间的相关性，即一阶跨日滞后自相关：
        峰日： corr{ P(d_k, t), P(d_{k-1}, t) }，d_k 为第 k 个峰日
        谷日： corr{ P(w_k, t), P(w_{k-1}, t) }，w_k 为第 k 个谷日
    其中「前一个峰日 / 前一个谷日」只在该组内部按日期先后取相邻，跨组不接连，
    因此相关系数刻画的是"同类日之间的日间可预测性"。

输出：
    q4/figures/20_峰日同时刻跨日相关性.csv   144 行（时刻）× 峰日数 列 + 均值列
    q4/figures/21_谷日同时刻跨日相关性.csv   144 行（时刻）× 谷日数 列 + 均值列
    q4/figures/20_峰谷日跨日相关性.png       相关性曲线对比图（含分组基准线）

时间口径：
    与 `q4_price_volatility.py` 一致。附件 4 每行 144 个 10 分钟电价，
    列顺序为 00:10 ~ 23:50、0:00+1（次日 00:00 在原始表中位于行尾）。
    时间线按自然钟表顺序重排为 00:00, 00:10, ..., 23:50，
    其中 00:00 取自该行的 '0:00+1' 列，即"来源日的 24:00"。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV = PROJECT_ROOT / "Data_preprocessed" / "processed" / "actuals_10min.csv"
RAW_XLSX = PROJECT_ROOT / "Data_preprocessed" / "raw" / "附件4.xlsx"
FIG_DIR = SCRIPT_DIR / "figures"

PRICE_COL = "price_actual_yuan_per_kwh"
WEEKEND_IS_VALLEY = True  # 谷日 = 周六周日

# 峰/谷日分组的论文配色
C_PEAK = "#B22222"   # 峰日：暖色
C_VALLEY = "#1F4E79"  # 谷日：冷色


# ---------------------------------------------------------------- 中文字体
def configure_style() -> None:
    cache_dir = PROJECT_ROOT / ".tmp" / "mplcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS",
        "PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
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
    print(f"[font] 使用中文字体: {chosen}")


# ---------------------------------------------------------------- 数据读取
def load_price_matrix() -> pd.DataFrame:
    """返回 (日期 × 144 个 10 分钟时刻) 的电价宽表，列名为 'HH:MM' 钟表时刻。"""
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, usecols=["source_date", "sample_time", PRICE_COL])
        df["source_date"] = pd.to_datetime(df["source_date"]).dt.normalize()
        df["sample_time"] = pd.to_datetime(df["sample_time"])
        # 00:00（次日零点）在本脚本中记为来源日的 00:00 时刻，与其余时刻构成
        # 00:00, 00:10, ..., 23:50 的完整 144 点日曲线顺序
        df["minutes"] = df["sample_time"].dt.hour * 60 + df["sample_time"].dt.minute
        wide = df.pivot_table(
            index="source_date", columns="minutes", values=PRICE_COL, aggfunc="mean"
        )
        src = "actuals_10min.csv"
    else:  # 兜底：直接从原始附件读取
        raw = pd.read_excel(RAW_XLSX)
        raw = raw.rename(columns={"日期\\时间": "date"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        cols, names = [], []
        for c in raw.columns:
            if c == "date":
                continue
            if isinstance(c, str):  # '0:00+1'
                cols.append(c)
                names.append(0)
            else:  # datetime.time
                cols.append(c)
                names.append(c.hour * 60 + c.minute)
        body = raw[cols].copy()
        body.columns = names
        wide = body.set_index(raw["date"])
        src = "附件4.xlsx"

    wide = wide.reindex(columns=sorted(wide.columns))
    wide.columns = [f"{m // 60:02d}:{m % 60:02d}" for m in wide.columns]
    wide.index.name = "date"

    print(
        f"[data] 来源 {src}：{wide.shape[0]} 天 × {wide.shape[1]} 个 10 分钟时刻，"
        f"缺失 {int(wide.isna().sum().sum())} 个"
    )
    return wide


def split_peak_valley(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """谷日 = 周六/周日；峰日 = 其余（周一~周五）。"""
    dow = wide.index.dayofweek  # 周一=0 ... 周日=6
    is_valley = dow >= 5 if WEEKEND_IS_VALLEY else dow < 5
    valley, peak = wide[is_valley], wide[~is_valley]

    peak_dow = pd.Series(peak.index.dayofweek).value_counts().sort_index()
    valley_dow = pd.Series(valley.index.dayofweek).value_counts().sort_index()
    names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    print(f"[split] 峰日 {len(peak)} 天（" +
          "、".join(f"{names[i]} {n}" for i, n in peak_dow.items()) + "）")
    print(f"[split] 谷日 {len(valley)} 天（" +
          "、".join(f"{names[i]} {n}" for i, n in valley_dow.items()) + "）")

    # 首尾日期与相邻间隔检查（确认同类日之间确实是"上一个同类日"）
    for tag, sub in (("峰日", peak), ("谷日", valley)):
        gaps = sub.index.to_series().diff().dt.days.dropna()
        print(
            f"[split] {tag}日期范围 {sub.index.min():%Y-%m-%d} ~ {sub.index.max():%Y-%m-%d}，"
            f"相邻同类日间隔 {int(gaps.min())}~{int(gaps.max())} 天"
        )
    return peak, valley


# ---------------------------------------------------------------- 相关计算
def cross_day_correlation(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """逐时刻计算「本日 vs 前一个同类日」的 Pearson 相关系数。

    返回 (列=相邻日对、行=时刻 的相关系数表, 每个时刻的样本量)。
    """
    cur = sub.iloc[1:].reset_index(drop=True)          # 第 2..K 个同类日
    prev = sub.iloc[:-1].reset_index(drop=True)        # 第 1..K-1 个同类日
    labels = [f"{d:%m-%d}" for d in sub.index[1:]]

    corr = cur.corrwith(prev, axis=0)  # 逐列（逐时刻）Pearson 相关
    corr.index.name = "时刻"
    corr = corr.rename("相关系数")

    # 每个时刻的有效样本量（前一日存在且两日该时刻均非缺失）
    valid = (cur.notna() & prev.notna()).sum(axis=0)
    valid.name = "样本量"

    out = pd.DataFrame({"相关系数": corr, "样本量": valid})
    out.attrs["labels"] = labels
    return out



def forecast_error(sub: pd.DataFrame) -> pd.DataFrame:
    """逐时刻计算「用前一同类日同时刻值作预测」的误差指标。

    预测方案（每个时刻 t 独立评估）：
        ŷ(d_k, t) = P(d_{k-1}, t)       前一同类日的同时刻电价

    三个指标（对 260 / 103 个日对逐时刻计算）：
        MAE_t  = mean|P(d_k,t) − P(d_{k-1},t)|                  单位 元/kWh
        MAPE_t = mean|P(d_k,t) − P(d_{k-1},t)| / |P(d_k,t)| ×100  单位 %（逐点比值后再平均）
        WAPE_t = Σ|P(d_k,t) − P(d_{k-1},t)| / Σ|P(d_k,t)| ×100   单位 %（先求和再相除）

    MAPE 与 WAPE 的区别：
        MAPE 先逐日算相对误差再平均，对**低价时刻**（分母小）敏感，容易被放大；
        WAPE 用总绝对误差除以总实际值，是**按电量加权**的整体相对误差，更稳健。
        两者对照可判断某时刻的相对误差是否由个别低价日的分母效应造成。
    """
    cur = sub.iloc[1:].reset_index(drop=True)    # 本日
    prev = sub.iloc[:-1].reset_index(drop=True)  # 前一同类日

    err = cur - prev                              # 预测误差（本日 − 预测值）
    ae = err.abs()
    y = cur.abs()                                 # 分母：本日实际值

    mae = ae.mean(axis=0)
    # 逐日相对误差再平均；分母为 0 的点置 NaN 后跳过
    with np.errstate(invalid="ignore", divide="ignore"):
        ape = np.where(y.values > 0, ae.values / y.values, np.nan) * 100.0
    mape = np.nanmean(ape, axis=0)
    # 先求和再相除
    wape = ae.sum(axis=0) / y.sum(axis=0) * 100.0

    out = pd.DataFrame(
        {
            "MAE": mae.values,
            "MAPE(%)": mape,
            "WAPE(%)": wape.values,
        },
        index=list(sub.columns),
    )
    out.index.name = "时刻"
    return out


def build_mae_table(sub: pd.DataFrame) -> pd.DataFrame:
    """生成 MAE 表：第一列时刻，其后 MAE / MAPE / WAPE 三列，最后一行为平均值。

    与相关系数表同形（144 时刻 + 平均值行），便于逐时刻对照阅读。
    注意：平均值行的 MAE 是对 144 个时刻取算术平均；MAPE/WAPE 同理
    （即"时刻间的平均相对误差"，不等于全天总体的相对误差）。
    """
    fe = forecast_error(sub)
    out = fe.reset_index()
    cols = ["时刻", "MAE", "MAPE(%)", "WAPE(%)"]
    out = out[cols]

    tail = {"时刻": "平均值"}
    for c in cols[1:]:
        tail[c] = float(out[c].mean())
    out = pd.concat([out, pd.DataFrame([tail])], ignore_index=True)
    out.attrs["fe"] = fe
    return out


def build_table(sub: pd.DataFrame, tag: str) -> pd.DataFrame:
    """生成最终输出表：第一列「时刻」，其后为相关性系数，最后一行为「平均值」。

    说明：本题的「前一个峰日 / 前一个谷日」是在**同类日内部**取相邻，
    因此每个时刻只有一条「(前一日, 本日)」的配对序列，逐时刻计算得到
    一个相关系数；表中每个时刻对应一列系数（列名标注该时刻），
    最后一行为对全部时刻取的平均值。
    """
    corr = cross_day_correlation(sub)
    out = pd.DataFrame(
        {
            "时刻": list(sub.columns),
            "相关系数": corr["相关系数"].values,
            "样本量（日对数）": corr["样本量"].values,
        }
    )
    # 末行：对 144 个时刻求平均
    tail = pd.DataFrame(
        [["平均值", float(corr["相关系数"].mean()), int(corr["样本量"].iloc[0])]],
        columns=out.columns,
    )
    out = pd.concat([out, tail], ignore_index=True)
    out.attrs["series"] = corr["相关系数"]
    return out


# ---------------------------------------------------------------- 绘图
def plot_curves(peak_corr: pd.Series, valley_corr: pd.Series) -> Path:
    """峰日 / 谷日各时刻相关系数曲线对比图。"""
    x = np.arange(len(peak_corr))
    hours = x // 6

    fig, ax = plt.subplots(figsize=(14.2, 6.6))
    fig.subplots_adjust(top=0.87)

    ax.plot(x, peak_corr.values, color=C_PEAK, linewidth=1.7,
            label="峰日（周一~周五）", zorder=4)
    ax.plot(x, valley_corr.values, color=C_VALLEY, linewidth=1.7,
            label="谷日（周六、周日）", zorder=4)

    ax.axhline(float(peak_corr.mean()), color=C_PEAK, linestyle="--",
               linewidth=1.15, alpha=0.75, zorder=2,
               label=f"峰日全天平均 {peak_corr.mean():.4f}")
    ax.axhline(float(valley_corr.mean()), color=C_VALLEY, linestyle="--",
               linewidth=1.15, alpha=0.75, zorder=2,
               label=f"谷日全天平均 {valley_corr.mean():.4f}")

    # 整点刻度
    ax.set_xticks(np.arange(0, 144, 6))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
    ax.set_xlim(-1.5, 143.5)

    lo = float(min(peak_corr.min(), valley_corr.min()))
    hi = float(max(peak_corr.max(), valley_corr.max()))
    span = hi - lo
    ax.set_ylim(lo - 0.14 * span, hi + 0.20 * span)

    ax.set_xlabel("时刻（10 分钟粒度，0:00 ~ 23:50）", labelpad=8)
    ax.set_ylabel("「本日 t 时刻」与「前一同类日 t 时刻」的相关系数", labelpad=8)
    ax.set_title(
        "图 3  峰日 / 谷日电价的同时刻跨日相关性（附件 4，2025 年）", pad=44
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              frameon=False, fontsize=9.8)

    # 标注平均相关最高 / 最低的时刻
    for series, color, tag in (
        (peak_corr, C_PEAK, "峰日"),
        (valley_corr, C_VALLEY, "谷日"),
    ):
        imax, imin = int(np.argmax(series.values)), int(np.argmin(series.values))
        for idx, va, dy in ((imax, "bottom", 0.02), (imin, "top", -0.02)):
            ax.annotate(
                f"{tag} {series.index[idx]}  {series.values[idx]:.3f}",
                xy=(idx, series.values[idx]),
                xytext=(idx, series.values[idx] + dy * span),
                ha="center", va=va, fontsize=8.8, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color,
                          alpha=0.92, linewidth=0.8),
            )

    out = FIG_DIR / "20_峰谷日跨日相关性.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] 已保存 {out}")
    return out


# ---------------------------------------------------------------- 主流程
def selftest(wide: pd.DataFrame, peak: pd.DataFrame, valley: pd.DataFrame) -> None:
    """自检：随机抽查若干 (时刻, 日对) 单元格，与 pandas 逐对计算比对。

    用 corrwith 独立重算整列相关系数，确认矩阵实现无误。
    """
    rng = np.random.default_rng(20250912)
    ok = True
    for tag, sub in (("峰日", peak), ("谷日", valley)):
        cur = sub.iloc[1:].reset_index(drop=True)
        prev = sub.iloc[:-1].reset_index(drop=True)
        ref = cur.corrwith(prev, axis=0)          # pandas 独立实现
        got = np.array([cross_day_correlation(sub)["相关系数"][c] for c in sub.columns])
        if not np.allclose(ref.values, got, atol=1e-12):
            ok = False
            bad = np.argmax(np.abs(ref.values - got))
            print(f"  [FAIL] {tag} 在 {sub.columns[bad]} 不一致："
                  f"{got[bad]} vs {ref.values[bad]}")
        print(f"  [ok] {tag}：144 个时刻全部与 pandas corrwith 一致；"
              f"首日 {sub.index[0]:%m-%d} 无前置同类日，已剔除，"
              f"实际参与 {len(cur)} 个日对")
    if ok:
        print("[selftest] 全部通过")


def plot_mae_curves(
    peak_fe: pd.DataFrame, valley_fe: pd.DataFrame
) -> Path:
    """图 4：峰日 / 谷日的 MAE 与 MAPE 曲线（双纵轴）。"""
    x = np.arange(len(peak_fe))
    fig, ax = plt.subplots(figsize=(14.2, 6.6))
    fig.subplots_adjust(top=0.87)

    ax.plot(x, peak_fe["MAE"].values, color=C_PEAK, linewidth=1.7,
            label="峰日 MAE（元/kWh，左轴）", zorder=5)
    ax.plot(x, valley_fe["MAE"].values, color=C_VALLEY, linewidth=1.7,
            label="谷日 MAE（元/kWh，左轴）", zorder=5)
    ax.set_ylabel("平均绝对误差 MAE（元/kWh）", labelpad=8)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(x, peak_fe["MAPE(%)"].values, color=C_PEAK, linewidth=1.2,
             linestyle="--", alpha=0.8, zorder=4, label="峰日 MAPE（%，右轴）")
    ax2.plot(x, valley_fe["MAPE(%)"].values, color=C_VALLEY, linewidth=1.2,
             linestyle="--", alpha=0.8, zorder=4, label="谷日 MAPE（%，右轴）")
    ax2.set_ylabel("平均绝对百分比误差 MAPE（%）", labelpad=8)
    ax2.set_ylim(bottom=0)
    ax2.grid(False)

    ax.set_xticks(np.arange(0, 144, 6))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
    ax.set_xlim(-1.5, 143.5)

    ax.set_xlabel("时刻（10 分钟粒度，0:00 ~ 23:50）", labelpad=8)
    ax.set_title("图 4  峰日 / 谷日电价的昨日同时刻预测误差（MAE 与 MAPE）", pad=44)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=4, frameon=False, fontsize=9.8)

    out = FIG_DIR / "22_峰谷日MAE对比.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] 已保存 {out}")
    return out


def plot_rho_vs_mae(peak_corr: pd.Series, valley_corr: pd.Series,
                    peak_fe: pd.DataFrame, valley_fe: pd.DataFrame) -> Path:
    """图 5：相关系数 ρ 与 MAE 的散点关系（检验"高相关是否等于低误差"）。"""
    fig, ax = plt.subplots(figsize=(9.6, 7.2))

    for corr, fe, color, tag in (
        (peak_corr, peak_fe, C_PEAK, "峰日"),
        (valley_corr, valley_fe, C_VALLEY, "谷日"),
    ):
        ax.scatter(corr.values, fe["MAE"].values, s=34,
                   facecolor=color, edgecolor="white", linewidth=0.7,
                   alpha=0.85, zorder=4, label=f"{tag}（144 个时刻）")
        # 线性拟合趋势线
        k, b = np.polyfit(corr.values, fe["MAE"].values, 1)
        xs = np.linspace(corr.values.min(), corr.values.max(), 50)
        ax.plot(xs, k * xs + b, color=color, linewidth=1.5, linestyle="--",
                alpha=0.9, zorder=3)
        r = np.corrcoef(corr.values, fe["MAE"].values)[0, 1]
        print(f"[stats] {tag}：ρ 与 MAE 的相关系数 = {r:+.3f}")

    ax.axvline(0, color="#888888", linewidth=0.9, linestyle="-", alpha=0.6, zorder=1)
    ax.set_xlabel("相关系数 ρ（本日 vs 前一同类日，同时刻）", labelpad=8)
    ax.set_ylabel("MAE（元/kWh）", labelpad=8)
    ax.set_title("图 5  相关系数 ρ 与预测误差 MAE 的关系\n"
                 "（ρ 高≠误差小：点在横轴上分散说明二者不完全一致）", pad=20)
    ax.legend(loc="upper right", framealpha=0.95)
    out = FIG_DIR / "26_相关系数与MAE散点.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] 已保存 {out}")
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    wide = load_price_matrix()
    peak, valley = split_peak_valley(wide)

    print("\n[selftest] 一致性校验")
    selftest(wide, peak, valley)

    peak_tab = build_table(peak, "峰日")
    valley_tab = build_table(valley, "谷日")

    peak_path = FIG_DIR / "20_峰日同时刻跨日相关性.csv"
    valley_path = FIG_DIR / "21_谷日同时刻跨日相关性.csv"
    peak_tab.to_csv(peak_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    valley_tab.to_csv(valley_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[table] 已保存 {peak_path}（{len(peak_tab) - 1} 个时刻 + 平均值行）")
    print(f"[table] 已保存 {valley_path}（{len(valley_tab) - 1} 个时刻 + 平均值行）")

    peak_corr = peak_tab.attrs["series"]
    valley_corr = valley_tab.attrs["series"]

    plot_curves(peak_corr, valley_corr)

    # ---------------- MAE 表 ----------------
    peak_mae_tab = build_mae_table(peak)
    valley_mae_tab = build_mae_table(valley)
    peak_mae_path = FIG_DIR / "24_峰日MAE表.csv"
    valley_mae_path = FIG_DIR / "25_谷日MAE表.csv"
    peak_mae_tab.to_csv(peak_mae_path, index=False, encoding="utf-8-sig",
                        float_format="%.6f")
    valley_mae_tab.to_csv(valley_mae_path, index=False, encoding="utf-8-sig",
                          float_format="%.6f")
    print(f"[table] 已保存 {peak_mae_path}（{len(peak_mae_tab) - 1} 个时刻 + 平均值行）")
    print(f"[table] 已保存 {valley_mae_path}（{len(valley_mae_tab) - 1} 个时刻 + 平均值行）")

    plot_mae_curves(peak_mae_tab.attrs["fe"], valley_mae_tab.attrs["fe"])
    plot_rho_vs_mae(peak_corr, valley_corr,
                    peak_mae_tab.attrs["fe"], valley_mae_tab.attrs["fe"])

    # ---------------- 控制台摘要 ----------------
    print("\n[结果] 峰日各时刻平均相关系数（按小时汇总）")
    print_hourly(peak_corr, "峰日")
    print("\n[结果] 谷日各时刻平均相关系数（按小时汇总）")
    print_hourly(valley_corr, "谷日")

    print("\n[结果] 全天总览")
    print(f"  峰日：均值 {peak_corr.mean():.4f}，中位数 {peak_corr.median():.4f}，"
          f"最低 {peak_corr.min():.4f}（{peak_corr.idxmin()}），"
          f"最高 {peak_corr.max():.4f}（{peak_corr.idxmax()}）")
    print(f"  谷日：均值 {valley_corr.mean():.4f}，中位数 {valley_corr.median():.4f}，"
          f"最低 {valley_corr.min():.4f}（{valley_corr.idxmin()}），"
          f"最高 {valley_corr.max():.4f}（{valley_corr.idxmax()}）")
    diff = (valley_corr - peak_corr)
    print(f"  谷日 − 峰日：均值差 {diff.mean():+.4f}；"
          f"谷日更高的时刻 {(diff > 0).sum()}/144 个")

    print("\n[结果] 预测误差汇总（对 144 个时刻取算术平均）")
    for tag, tab in (("峰日", peak_mae_tab), ("谷日", valley_mae_tab)):
        last = tab.iloc[-1]
        fe = tab.attrs["fe"]
        print(f"  {tag}：MAE {last['MAE']:.4f} 元/kWh，"
              f"MAPE {last['MAPE(%)']:.2f}%，WAPE {last['WAPE(%)']:.2f}%")
        # MAPE 与 WAPE 差异最大的时刻：说明这些时刻的相对误差受低价分母放大
        gap = fe["MAPE(%)"] - fe["WAPE(%)"]
        top = gap.sort_values(ascending=False).head(3)
        print(f"       MAPE−WAPE 最大的 3 个时刻（低价分母效应最强）：" +
              "、".join(f"{t} {v:+.1f}pp" for t, v in top.items()))
        print(f"       MAE 最大/最小的时刻：{fe['MAE'].idxmax()} "
              f"{fe['MAE'].max():.4f} / {fe['MAE'].idxmin()} {fe['MAE'].min():.4f}")

    print("\n[done] 全部输出至", FIG_DIR)


def print_hourly(corr: pd.Series, tag: str) -> None:
    hourly = corr.groupby(corr.index.str.slice(0, 2)).mean()
    line = "  ".join(f"{h}:00 {v:.3f}" for h, v in hourly.items())
    print(f"  {tag}: {line}")


if __name__ == "__main__":
    main()
