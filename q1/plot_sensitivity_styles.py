"""
批量生成敏感性分析多种可视化风格
================================

运行示例：
    python plot_sensitivity_styles.py

默认读取：
    results/sensitivity/sensitivity_results.csv
    results/sensitivity/sensitivity_elasticity.csv

默认输出到：
    results/sensitivity/styles/

将一次性生成以下图片：
1. style_01_faceted_lines.png        分面折线图（购电费用归一化）
2. style_02_tornado.png             Tornado 敏感性图
3. style_03_lollipop.png            Lollipop 排名图
4. style_04_heatmap_cost.png        购电费用热力图
5. style_05_heatmap_savings.png     节省率变化热力图
6. style_06_dumbbell.png            哑铃图
7. style_07_faceted_savings.png     分面折线图（节省率）
8. style_08_slopegraph.png          基准-低值-高值坡度图

说明：
- 仅使用 matplotlib；
- 不改动原始 CSV；
- 适合从多种论文风格中快速挑选适合正文/附录的版本。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 基础设置
# ============================================================

ORDER = ["efficiency", "capacity_kwh", "power_kw", "price_spread"]

TITLES = {
    "efficiency": "（a）充放电效率",
    "capacity_kwh": "（b）储能容量",
    "power_kw": "（c）最大充放电功率",
    "price_spread": "（d）峰谷电价差",
}

DISPLAY_ORDER_CN = ["充放电效率", "储能容量", "最大充放电功率", "峰谷电价差"]


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def ensure_columns(frame: pd.DataFrame, columns, name: str):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def load_tables(results_path: Path, elasticity_path: Path):
    results = pd.read_csv(results_path)
    elasticity = pd.read_csv(elasticity_path)

    ensure_columns(
        results,
        [
            "experiment",
            "experiment_cn",
            "parameter_value",
            "baseline_value",
            "parameter_ratio",
            "cost_ratio",
            "cost_yuan",
            "savings_percent",
            "savings_change_pp",
        ],
        "sensitivity_results.csv",
    )

    ensure_columns(
        elasticity,
        [
            "experiment",
            "experiment_cn",
            "elasticity",
            "absolute_elasticity",
        ],
        "sensitivity_elasticity.csv",
    )

    return results, elasticity


def ordered_results(results: pd.DataFrame):
    frames = []
    for exp in ORDER:
        sub = results[results["experiment"] == exp].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("parameter_ratio")
        frames.append(sub)
    return frames


# ============================================================
# 风格 1：分面折线图（购电费用）
# ============================================================

def plot_faceted_lines(results, output_dir: Path):
    plt = configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()

    for ax, exp in zip(axes, ORDER):
        sub = results[results["experiment"] == exp].sort_values("parameter_ratio")
        ax.plot(
            sub["parameter_ratio"],
            sub["cost_ratio"],
            marker="o",
            linewidth=2,
        )
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.axvline(1.0, linestyle="--", linewidth=1)

        for _, row in sub.iterrows():
            ax.text(
                row["parameter_ratio"],
                row["cost_ratio"],
                f"{row['cost_ratio']:.3f}",
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax.set_title(TITLES[exp], fontsize=12)
        ax.set_xlabel("参数相对基准值")
        ax.set_ylabel("购电费用 / 基准购电费用")
        ax.grid(alpha=0.25)

    fig.suptitle("第一问敏感性分析（分面展示）", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "style_01_faceted_lines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 风格 2：Tornado 图
# ============================================================

def build_tornado_table(results):
    rows = []

    for exp, sub in results.groupby("experiment", sort=False):
        sub = sub.sort_values("parameter_value")
        x0 = sub["baseline_value"].iloc[0]
        base = sub[np.isclose(sub["parameter_value"], x0)].iloc[0]
        lower = sub[sub["parameter_value"] < x0].iloc[-1]
        upper = sub[sub["parameter_value"] > x0].iloc[0]

        rows.append(
            {
                "experiment_cn": base["experiment_cn"],
                "lower_change": float(lower["cost_ratio"] - 1.0),
                "upper_change": float(upper["cost_ratio"] - 1.0),
                "magnitude": float(max(abs(lower["cost_ratio"] - 1.0), abs(upper["cost_ratio"] - 1.0))),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("magnitude", ascending=True).reset_index(drop=True)
    return df


def plot_tornado(results, output_dir: Path):
    plt = configure_matplotlib()

    df = build_tornado_table(results)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.barh(y, df["lower_change"], height=0.35, label="参数减小")
    ax.barh(y, df["upper_change"], height=0.35, label="参数增大")
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["experiment_cn"])
    ax.set_xlabel("购电费用相对变化（相对于基准）")
    ax.set_title("第一问关键参数 Tornado 敏感性图")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)

    for i, v in enumerate(df["lower_change"]):
        ax.text(v, i, f"{v:+.3f}", va="center", ha="right" if v < 0 else "left", fontsize=9)

    for i, v in enumerate(df["upper_change"]):
        ax.text(v, i, f"{v:+.3f}", va="center", ha="right" if v < 0 else "left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "style_02_tornado.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 风格 3：Lollipop 排名图
# ============================================================

def plot_lollipop(elasticity, output_dir: Path):
    plt = configure_matplotlib()

    df = elasticity.sort_values("absolute_elasticity", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.hlines(y=y, xmin=0, xmax=df["absolute_elasticity"], linewidth=2)
    ax.plot(df["absolute_elasticity"], y, "o", markersize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["experiment_cn"])
    ax.set_xlabel(r"局部敏感性系数绝对值 $|S|$")
    ax.set_title("第一问参数局部敏感性排序")
    ax.grid(axis="x", alpha=0.25)

    for i, v in enumerate(df["absolute_elasticity"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "style_03_lollipop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 风格 4：购电费用热力图
# ============================================================

def _sorted_ratios(results):
    vals = sorted(results["parameter_ratio"].dropna().unique())
    return vals


def plot_heatmap_cost(results, output_dir: Path):
    plt = configure_matplotlib()

    ratios = _sorted_ratios(results)
    pivot = results.pivot(index="experiment_cn", columns="parameter_ratio", values="cost_ratio")
    pivot = pivot.reindex(index=DISPLAY_ORDER_CN, columns=ratios)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("参数相对基准值")
    ax.set_title("第一问购电费用归一化热力图")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("购电费用 / 基准购电费用")

    fig.tight_layout()
    fig.savefig(output_dir / "style_04_heatmap_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 风格 5：节省率变化热力图
# ============================================================

def plot_heatmap_savings(results, output_dir: Path):
    plt = configure_matplotlib()

    ratios = _sorted_ratios(results)
    pivot = results.pivot(index="experiment_cn", columns="parameter_ratio", values="savings_change_pp")
    pivot = pivot.reindex(index=DISPLAY_ORDER_CN, columns=ratios)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("参数相对基准值")
    ax.set_title("第一问节省率变化热力图（百分点）")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("节省率变化（百分点）")

    fig.tight_layout()
    fig.savefig(output_dir / "style_05_heatmap_savings.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 风格 6：哑铃图
# ============================================================

def plot_dumbbell(results, output_dir: Path):
    plt = configure_matplotlib()

    rows = []
    for exp, sub in results.groupby("experiment", sort=False):
        sub = sub.sort_values("parameter_value")
        x0 = sub["baseline_value"].iloc[0]
        base = sub[np.isclose(sub["parameter_value"], x0)].iloc[0]
        low = sub.iloc[0]
        high = sub.iloc[-1]

        rows.append(
            {
                "experiment_cn": base["experiment_cn"],
                "low": float(low["cost_ratio"]),
                "base": float(base["cost_ratio"]),
                "high": float(high["cost_ratio"]),
            }
        )

    df = pd.DataFrame(rows).sort_values("base", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for i, row in df.iterrows():
        ax.plot([row["low"], row["high"]], [i, i], linewidth=2)
        ax.plot(row["low"], i, "o", markersize=8)
        ax.plot(row["base"], i, "s", markersize=8)
        ax.plot(row["high"], i, "o", markersize=8)

        ax.text(row["low"], i + 0.10, f"{row['low']:.3f}", fontsize=9, ha="center")
        ax.text(row["base"], i + 0.10, f"{row['base']:.3f}", fontsize=9, ha="center")
        ax.text(row["high"], i + 0.10, f"{row['high']:.3f}", fontsize=9, ha="center")

    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["experiment_cn"])
    ax.set_xlabel("购电费用 / 基准购电费用")
    ax.set_title("第一问参数变化区间对购电费用的影响")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "style_06_dumbbell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 风格 7：分面折线图（节省率）
# ============================================================

def plot_faceted_savings(results, output_dir: Path):
    plt = configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()

    for ax, exp in zip(axes, ORDER):
        sub = results[results["experiment"] == exp].sort_values("parameter_ratio")
        ax.plot(
            sub["parameter_ratio"],
            sub["savings_percent"],
            marker="o",
            linewidth=2,
        )

        baseline_value = float(sub["baseline_value"].iloc[0])
        base_row = sub[np.isclose(sub["parameter_value"], baseline_value)].iloc[0]
        ax.axhline(base_row["savings_percent"], linestyle="--", linewidth=1)
        ax.axvline(1.0, linestyle="--", linewidth=1)

        for _, row in sub.iterrows():
            ax.text(
                row["parameter_ratio"],
                row["savings_percent"],
                f"{row['savings_percent']:.2f}",
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax.set_title(TITLES[exp], fontsize=12)
        ax.set_xlabel("参数相对基准值")
        ax.set_ylabel("节省率 / %")
        ax.grid(alpha=0.25)

    fig.suptitle("第一问敏感性分析（节省率视角）", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "style_07_faceted_savings.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 风格 8：Slope graph
# ============================================================

def plot_slopegraph(results, output_dir: Path):
    plt = configure_matplotlib()

    rows = []
    for exp, sub in results.groupby("experiment", sort=False):
        sub = sub.sort_values("parameter_value")
        x0 = sub["baseline_value"].iloc[0]
        base = sub[np.isclose(sub["parameter_value"], x0)].iloc[0]
        low = sub.iloc[0]
        high = sub.iloc[-1]

        rows.append(
            {
                "experiment_cn": base["experiment_cn"],
                "低值": float(low["cost_ratio"]),
                "基准": float(base["cost_ratio"]),
                "高值": float(high["cost_ratio"]),
            }
        )

    df = pd.DataFrame(rows).reset_index(drop=True)

    x_positions = [0, 1, 2]
    labels = ["低值", "基准", "高值"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for _, row in df.iterrows():
        yvals = [row["低值"], row["基准"], row["高值"]]
        ax.plot(x_positions, yvals, marker="o", linewidth=1.8)
        ax.text(x_positions[-1] + 0.03, yvals[-1], row["experiment_cn"], va="center", fontsize=10)

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("购电费用 / 基准购电费用")
    ax.set_title("第一问参数变化下的坡度图比较")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "style_08_slopegraph.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="批量生成敏感性分析多种图形风格")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/sensitivity/sensitivity_results.csv"),
        help="敏感性实验结果 CSV",
    )
    parser.add_argument(
        "--elasticity",
        type=Path,
        default=Path("results/sensitivity/sensitivity_elasticity.csv"),
        help="局部敏感性系数 CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sensitivity/styles"),
        help="图片输出目录",
    )

    args = parser.parse_args()

    results_path = args.results.resolve()
    elasticity_path = args.elasticity.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"找不到结果文件: {results_path}")
    if not elasticity_path.exists():
        raise FileNotFoundError(f"找不到弹性文件: {elasticity_path}")

    results, elasticity = load_tables(results_path, elasticity_path)

    plot_faceted_lines(results, output_dir)
    plot_tornado(results, output_dir)
    plot_lollipop(elasticity, output_dir)
    plot_heatmap_cost(results, output_dir)
    plot_heatmap_savings(results, output_dir)
    plot_dumbbell(results, output_dir)
    plot_faceted_savings(results, output_dir)
    plot_slopegraph(results, output_dir)

    print("=" * 72)
    print("敏感性分析多风格图片已生成")
    print(f"输出目录：{output_dir}")
    print("已生成文件：")
    for name in [
        "style_01_faceted_lines.png",
        "style_02_tornado.png",
        "style_03_lollipop.png",
        "style_04_heatmap_cost.png",
        "style_05_heatmap_savings.png",
        "style_06_dumbbell.png",
        "style_07_faceted_savings.png",
        "style_08_slopegraph.png",
    ]:
        print(" -", name)
    print("=" * 72)


if __name__ == "__main__":
    main()
