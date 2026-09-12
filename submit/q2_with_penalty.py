"""Q2 variant with a soft penalty on deviation from the 6000 kWh target.

This script reuses q2's data, forecast and dispatch code.  It prints an
in-memory comparison and does not create an Excel file or intermediate files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import q2


TERMINAL_TARGET_KWH = 6000.0
TERMINAL_PENALTY_YUAN_PER_KWH = 0.45


def evaluate(terminal_target_kwh, terminal_penalty_yuan_per_kwh):
    root = Path(__file__).resolve().parent
    dates, load, pv, prices = q2.load_inputs(
        root / "Data" / "附件2.xlsx",
        root / "Data" / "附件1.xlsx",
    )
    _, _, scenarios = q2.forecast_scenarios(dates, load, pv)
    run_idx = np.where(
        (dates >= pd.Timestamp(q2.RUN_START))
        & (dates <= pd.Timestamp(q2.RUN_END))
    )[0]

    q_star, _ = q2.scan_q_star(
        dates, load, pv, prices, scenarios,
        terminal_target_kwh=terminal_target_kwh,
        terminal_penalty_yuan_per_kwh=terminal_penalty_yuan_per_kwh,
    )
    daily, slots = q2.run_q(
        dates, load, pv, prices, scenarios, run_idx, q_star, True,
        terminal_target_kwh, terminal_penalty_yuan_per_kwh,
    )
    ends = slots.groupby("date").planned_storage_end_kwh.last()
    summary = {
        "q_star": q_star,
        "planned_cost_yuan": float(daily.planned_cost_yuan.sum()),
        "emergency_cost_yuan": float(daily.emergency_cost_yuan.sum()),
        "total_cost_yuan": float(daily.total_cost_yuan.sum()),
        "emergency_kwh": float(daily.emergency_kwh.sum()),
        "terminal_mean_kwh": float(ends.mean()),
        "terminal_min_kwh": float(ends.min()),
        "terminal_max_kwh": float(ends.max()),
    }
    return summary, daily, slots


def main():
    current, _, _ = evaluate(None, 0.0)
    penalty, daily, slots = evaluate(
        TERMINAL_TARGET_KWH, TERMINAL_PENALTY_YUAN_PER_KWH
    )
    root = Path(__file__).resolve().parent
    output = root / "results" / "result2.xlsx"
    q2.export_result2(
        template_path=root / "Data" / "附件5" / "result2.xlsx",
        output_path=output,
        daily=daily,
        slots=slots,
        events=q2.emergency_events(slots),
    )
    print("当前 q2:", current)
    print("惩罚版 q2:", penalty)
    print("总费用变化: {:.2f} 元".format(
        penalty["total_cost_yuan"] - current["total_cost_yuan"]
    ))
    print("惩罚版结果:", output.resolve())


if __name__ == "__main__":
    main()
