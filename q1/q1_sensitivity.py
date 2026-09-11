"""
Q1 敏感性分析
============

运行：
    python q1_sensitivity.py

默认分析四类关键参数：
1. 充/放电效率 eta：0.80, 0.85, 0.90, 0.95, 0.98
2. 名义储能容量 C：8000, 10000, 12000, 14000, 16000 kWh
   同步设置 E_min=0.1C, E_max=0.9C, E_0=E_T=0.5C
3. 最大充/放电功率 Pmax：3000, 4000, 5000, 6000, 7000 kW
4. 峰谷电价差系数 alpha：0.6, 0.8, 1.0, 1.2, 1.4
   p_t(alpha) = p_bar + alpha * (p_t - p_bar)

输出：
    results/sensitivity/sensitivity_results.csv
    results/sensitivity/sensitivity_elasticity.csv
    results/sensitivity/01_cost_sensitivity.png
    results/sensitivity/02_sensitivity_elasticity.png

说明：
- 不修改 q1.py；
- 不覆盖 result1.xlsx；
- 所有实验都重新调用 q1.py 中相同的优化模型与物理校验；
- 节省率始终与“同一组电价条件下的无储能方案”比较。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from q1 import CFG, Q1_DIR, solve_dispatch, check_solution


EFFICIENCIES = [0.80, 0.85, 0.90, 0.95, 0.98]
CAPACITIES_KWH = [8000, 10000, 12000, 14000, 16000]
POWERS_KW = [3000, 4000, 5000, 6000, 7000]
PRICE_SPREAD_FACTORS = [0.60, 0.80, 1.00, 1.20, 1.40]

BASELINE = {
    "efficiency": 0.90,
    "capacity_kwh": 12000.0,
    "power_kw": 5000.0,
    "price_spread": 1.00,
}

EXPERIMENT_CN = {
    "efficiency": "充放电效率",
    "capacity_kwh": "储能容量",
    "power_kw": "最大充放电功率",
    "price_spread": "峰谷电价差",
}


def load_q1_input(path: Path):
    source = pd.read_csv(path, float_precision="round_trip")

    required = {
        "slot",
        "interval_start_minute",
        "interval_end_minute",
        "price_yuan_per_kwh",
        "load_kw",
        "pv_forecast_kw",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Q1 输入缺少字段: {sorted(missing)}")

    if len(source) != 144:
        raise ValueError("Q1 敏感性分析要求完整典型日 144 个十分钟时段")

    if not np.array_equal(
        source["interval_start_minute"].to_numpy(),
        np.arange(0, 1440, 10),
    ):
        raise ValueError("Q1 输入时间索引异常")

    price = source["price_yuan_per_kwh"].to_numpy(dtype=float)
    load = source["load_kw"].to_numpy(dtype=float) / 6.0
    pv = source["pv_forecast_kw"].to_numpy(dtype=float) / 6.0

    return source, price, load, pv


def no_storage_cost(price, load, pv):
    grid = np.maximum(load - pv, 0.0)
    return float(price @ grid), float(grid.sum())


def run_one(price, load, pv, cfg, solver="auto"):
    solution = solve_dispatch(price, load, pv, cfg, solver=solver)
    check_solution(solution, price, load, pv, cfg)

    base_cost, base_grid = no_storage_cost(price, load, pv)

    cost = float(solution["cost"])
    savings = base_cost - cost
    savings_pct = 100.0 * savings / base_cost if base_cost > 0 else 0.0

    return {
        "cost_yuan": cost,
        "no_storage_cost_yuan": base_cost,
        "savings_yuan": savings,
        "savings_percent": savings_pct,
        "grid_kwh": float(solution["grid"].sum()),
        "no_storage_grid_kwh": base_grid,
        "charge_kwh": float(solution["charge"].sum()),
        "discharge_kwh": float(solution["discharge"].sum()),
        "curtailment_kwh": float(solution["curtailment"].sum()),
        "storage_min_actual_kwh": float(solution["storage"].min()),
        "storage_max_actual_kwh": float(solution["storage"].max()),
        "solver": solution["solver"],
    }


def experiment_row(
    experiment,
    value,
    baseline_value,
    price,
    load,
    pv,
    cfg,
    solver,
):
    metrics = run_one(price, load, pv, cfg, solver=solver)

    return {
        "experiment": experiment,
        "experiment_cn": EXPERIMENT_CN[experiment],
        "parameter_value": float(value),
        "baseline_value": float(baseline_value),
        "parameter_ratio": float(value / baseline_value),
        **metrics,
    }


def run_sensitivity(price, load, pv, solver="auto"):
    rows = []

    for eta in EFFICIENCIES:
        cfg = dict(CFG)
        cfg["charge_efficiency"] = eta
        cfg["discharge_efficiency"] = eta

        rows.append(
            experiment_row(
                "efficiency",
                eta,
                BASELINE["efficiency"],
                price,
                load,
                pv,
                cfg,
                solver,
            )
        )

    for capacity in CAPACITIES_KWH:
        cfg = dict(CFG)
        cfg["storage_min_kwh"] = 0.10 * capacity
        cfg["storage_max_kwh"] = 0.90 * capacity
        cfg["initial_storage_kwh"] = 0.50 * capacity
        cfg["terminal_storage_kwh"] = 0.50 * capacity

        rows.append(
            experiment_row(
                "capacity_kwh",
                capacity,
                BASELINE["capacity_kwh"],
                price,
                load,
                pv,
                cfg,
                solver,
            )
        )

    for power in POWERS_KW:
        cfg = dict(CFG)
        cfg["max_charge_power_kw"] = float(power)
        cfg["max_discharge_power_kw"] = float(power)

        rows.append(
            experiment_row(
                "power_kw",
                power,
                BASELINE["power_kw"],
                price,
                load,
                pv,
                cfg,
                solver,
            )
        )

    mean_price = float(np.mean(price))

    for alpha in PRICE_SPREAD_FACTORS:
        modified_price = mean_price + alpha * (price - mean_price)

        if np.any(modified_price <= 0):
            raise ValueError(
                f"alpha={alpha} 导致非正电价，不能用于当前优化模型"
            )

        cfg = dict(CFG)

        rows.append(
            experiment_row(
                "price_spread",
                alpha,
                BASELINE["price_spread"],
                modified_price,
                load,
                pv,
                cfg,
                solver,
            )
        )

    result = pd.DataFrame(rows)

    baseline_cost_map = {}
    baseline_savings_map = {}

    for experiment, group in result.groupby("experiment", sort=False):
        b = group.loc[
            np.isclose(
                group["parameter_value"],
                group["baseline_value"],
                rtol=0,
                atol=1e-12,
            )
        ]
        if len(b) != 1:
            raise ValueError(f"{experiment}: 找不到唯一基准点")

        baseline_cost_map[experiment] = float(b.iloc[0]["cost_yuan"])
        baseline_savings_map[experiment] = float(
            b.iloc[0]["savings_percent"]
        )

    result["cost_ratio"] = [
        row.cost_yuan / baseline_cost_map[row.experiment]
        for row in result.itertuples()
    ]
    result["cost_change_percent"] = 100.0 * (
        result["cost_ratio"] - 1.0
    )
    result["savings_change_pp"] = [
        row.savings_percent - baseline_savings_map[row.experiment]
        for row in result.itertuples()
    ]

    return result


def central_elasticity(group: pd.DataFrame):
    group = group.sort_values("parameter_value").reset_index(drop=True)

    x0 = float(group["baseline_value"].iloc[0])
    baseline = group[np.isclose(group.parameter_value, x0)]

    if len(baseline) != 1:
        raise ValueError("基准点不是唯一的")

    j0 = float(baseline.iloc[0].cost_yuan)

    lower = group[group.parameter_value < x0]
    upper = group[group.parameter_value > x0]

    if lower.empty or upper.empty:
        raise ValueError("基准点两侧必须都有实验点")

    low = lower.iloc[-1]
    high = upper.iloc[0]

    x_low = float(low.parameter_value)
    x_high = float(high.parameter_value)
    j_low = float(low.cost_yuan)
    j_high = float(high.cost_yuan)

    slope = (j_high - j_low) / (x_high - x_low)
    elasticity = slope * x0 / j0

    return {
        "experiment": str(group.experiment.iloc[0]),
        "experiment_cn": str(group.experiment_cn.iloc[0]),
        "baseline_value": x0,
        "low_value": x_low,
        "high_value": x_high,
        "baseline_cost_yuan": j0,
        "low_cost_yuan": j_low,
        "high_cost_yuan": j_high,
        "elasticity": float(elasticity),
        "absolute_elasticity": float(abs(elasticity)),
    }


def make_elasticity_table(results):
    rows = [
        central_elasticity(group)
        for _, group in results.groupby("experiment", sort=False)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("absolute_elasticity", ascending=False)
        .reset_index(drop=True)
    )


def configure_chinese_font():
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


def make_plots(results, elasticity, output_dir: Path):
    plt = configure_chinese_font()

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.4, 5.4))

    for experiment, group in results.groupby("experiment", sort=False):
        group = group.sort_values("parameter_ratio")
        ax.plot(
            group["parameter_ratio"],
            group["cost_ratio"],
            marker="o",
            linewidth=1.8,
            label=EXPERIMENT_CN[experiment],
        )

    ax.axhline(1.0, linewidth=0.9, linestyle="--")
    ax.axvline(1.0, linewidth=0.9, linestyle="--")
    ax.set_xlabel("参数相对基准值")
    ax.set_ylabel("购电费用 / 基准购电费用")
    ax.set_title("第一问关键参数敏感性分析")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        output_dir / "01_cost_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    e = elasticity.sort_values(
        "absolute_elasticity",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(e["experiment_cn"], e["absolute_elasticity"])
    ax.set_xlabel(r"局部敏感性系数绝对值 $|S|$")
    ax.set_title("第一问参数局部敏感性比较")
    ax.grid(axis="x", alpha=0.22)

    for y, value in enumerate(e["absolute_elasticity"]):
        ax.text(
            value,
            y,
            f"  {value:.3f}",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(
        output_dir / "02_sensitivity_elasticity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_summary(results, elasticity):
    baseline = results[
        (results.experiment == "efficiency")
        & np.isclose(
            results.parameter_value,
            BASELINE["efficiency"],
        )
    ].iloc[0]

    print("\n" + "=" * 72)
    print("Q1 SENSITIVITY ANALYSIS: PASS")
    print("=" * 72)
    print(
        f"基准购电费用：{baseline.cost_yuan:.4f} 元"
    )
    print(
        f"基准无储能费用：{baseline.no_storage_cost_yuan:.4f} 元"
    )
    print(
        f"基准节省率：{baseline.savings_percent:.4f}%"
    )

    print("\n局部敏感性排序（按 |S| 从大到小）：")
    for i, row in elasticity.iterrows():
        direction = (
            "参数增大 → 成本下降"
            if row.elasticity < 0
            else "参数增大 → 成本上升"
        )
        print(
            f"{i+1}. {row.experiment_cn}: "
            f"S={row.elasticity:.4f}, |S|={row.absolute_elasticity:.4f} "
            f"({direction})"
        )

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Q1 四因素敏感性分析"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Q1_DIR / "input" / "q1_typical_day.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Q1_DIR / "results" / "sensitivity",
    )
    parser.add_argument(
        "--solver",
        choices=["auto", "milp"],
        default="auto",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="只输出CSV，不生成敏感性图",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到 Q1 输入文件: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    _, price, load, pv = load_q1_input(input_path)

    results = run_sensitivity(
        price,
        load,
        pv,
        solver=args.solver,
    )
    elasticity = make_elasticity_table(results)

    results.to_csv(
        output_dir / "sensitivity_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    elasticity.to_csv(
        output_dir / "sensitivity_elasticity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not args.no_plots:
        make_plots(results, elasticity, output_dir)

    print_summary(results, elasticity)

    print("\n输出目录：", output_dir)
    print("  - sensitivity_results.csv")
    print("  - sensitivity_elasticity.csv")
    if not args.no_plots:
        print("  - 01_cost_sensitivity.png")
        print("  - 02_sensitivity_elasticity.png")


if __name__ == "__main__":
    main()
