from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "q3" / "figures3"
OUT.mkdir(parents=True, exist_ok=True)


def configure_style():
    """Register an available CJK font so Chinese labels render in PNG/PDF."""
    for directory in [
        Path.home() / ".local/share/fonts",
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]:
        if not directory.exists():
            continue
        for path in directory.glob("*"):
            if path.suffix.lower() in {".ttf", ".ttc", ".otf"}:
                try:
                    font_manager.fontManager.addfont(str(path))
                except (OSError, RuntimeError, TypeError):
                    pass
    names = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "PingFang SC", "Heiti SC", "Songti SC", "STSong", "Noto Sans CJK SC",
        "Noto Serif CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS",
    ]
    selected = [name for name in candidates if name in names]
    if not selected:
        raise RuntimeError("未找到可用的中文字体，无法生成中文标注图。")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": selected + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# Reuse the executed Q2 notebook model so the comparison uses its final W=30
# walk-forward specification instead of reconstructing a second implementation.
nb = json.loads((ROOT / "q2" / "02_日曲线PCA岭回归_调参与消融_W30_含概率校准.ipynb").read_text(encoding="utf-8"))
plt.show = lambda *args, **kwargs: None
namespace = {"__name__": "q2_notebook_runtime", "__file__": str(ROOT / "q2" / "02_日曲线PCA岭回归_调参与消融_W30_含概率校准.ipynb")}
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        code = "".join(cell.get("source", []))
        code = code.replace("from IPython.display import display", "display = lambda *args, **kwargs: None")
        exec(compile(code, "q2_notebook_cell", "exec"), namespace)

dates = namespace["DATES"]
eval_idx = namespace["EVAL_IDX"]
actual_pv = namespace["PV"]
q2_pred_pv = namespace["FINAL_P"]

q2_rows = []
for slot in range(144):
    errors = np.abs(q2_pred_pv[eval_idx, slot] - actual_pv[eval_idx, slot])
    q2_rows.append({
        "source": "Q2最终模型",
        "issue_hour": "次日00:00发布",
        "target_hour": slot / 6,
        "n": int(np.isfinite(errors).sum()),
        "MAE_kw": float(np.nanmean(errors)),
    })
q2 = pd.DataFrame(q2_rows)

forecast = pd.read_csv(ROOT / "Data_preprocessed" / "processed" / "pv_forecast_hourly.csv", parse_dates=["issue_time", "target_time"])
actual = pd.read_csv(ROOT / "Data_preprocessed" / "processed" / "actuals_10min.csv", parse_dates=["interval_end"], usecols=["interval_end", "pv_kw"])
q3 = forecast.merge(actual, left_on="target_time", right_on="interval_end", how="inner")
q3 = q3[q3["target_in_actual_coverage"] & q3["pv_kw"].notna()].copy()
q3["abs_error_kw"] = (q3["pv_forecast_kw"] - q3["pv_kw"]).abs()
q3["issue_hour"] = q3["issue_time"].dt.hour
q3["target_hour"] = q3["target_time"].dt.hour + q3["target_time"].dt.minute / 60
q3_summary = q3.groupby(["issue_hour", "target_hour"], as_index=False).agg(
    n=("abs_error_kw", "size"), MAE_kw=("abs_error_kw", "mean")
)
q3_summary.insert(0, "source", "Q3附件3预报")
q3_summary["issue_hour"] = q3_summary["issue_hour"].map(lambda h: f"{int(h):02d}:00发布预报")
q3_summary = q3_summary[["source", "issue_hour", "target_hour", "n", "MAE_kw"]]

comparison = pd.concat([q2, q3_summary], ignore_index=True)
comparison.to_csv(OUT / "q2_q3_pv_mae_by_target_hour.csv", index=False, encoding="utf-8-sig")

configure_style()
fig, ax = plt.subplots(figsize=(11.5, 6.2))
for hour, group in q3_summary.groupby("issue_hour", sort=False):
    ax.plot(group["target_hour"], group["MAE_kw"], lw=1.8, marker="o", ms=3.2, label=f"Q3 {hour}")
ax.plot(q2["target_hour"], q2["MAE_kw"], color="black", lw=2.6, marker="s", ms=3.2, label="Q2最终模型（次日预测）")
ax.axvspan(0, 6, color="#DCEEF9", alpha=.55)
ax.axvspan(6, 18, color="#FCF3DD", alpha=.55)
ax.axvspan(18, 24, color="#F7E5E2", alpha=.55)
ax.set(xlim=(0, 24), xticks=np.arange(0, 25, 2), xlabel="目标时刻（小时）", ylabel="光伏预测 MAE（kW）")
ax.set_title("Q2最终模型与Q3各版本光伏预测误差的目标时刻对比")
ax.grid(True, ls="--", alpha=.35)
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "q2_q3_pv_mae_by_target_hour.png", dpi=220)
fig.savefig(OUT / "q2_q3_pv_mae_by_target_hour.pdf")
plt.close(fig)

print(comparison.groupby(["source", "issue_hour"], sort=False)["MAE_kw"].mean().round(2).to_string())
print(f"saved: {OUT}")
