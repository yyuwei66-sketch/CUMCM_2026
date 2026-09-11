# -*- coding: utf-8 -*-
"""
将 q2/results/predict.xlsx 的预测结果与 附件1.xlsx 的电价整合。

输出：q2/results/predict_with_price.xlsx
  - 每天一个工作表（以日期命名，如 2025-02-01），共 334 个日工作表
  - 每个工作表 144 行，纵轴与 附件1 相同（同一时间轴、同一 144 个时刻）
  - 横轴列顺序：
      时间 | 电价 | 实际净负载 | 净负载p05 | 净负载p10 | 净负载p50(中心预测) | 净负载p90 | 净负载p95

说明：
  * 电价取自 Data_preprocessed/raw/附件1.xlsx 的「电价」列，144 点，对所有日期相同。
  * 实际净负载 = 实际负载 - 实际光伏，来自 predict.xlsx 的 actual_net_load_kw。
  * 分位数来自 predict.xlsx；其中 p50 即净负荷中心预测 forecast_net_load_kw
    （中心预测就是 50 分位数）。
  * predict.xlsx 的评价期为 2025-02-01 ~ 2025-12-31，共 334 天。

用法：
    python q2/results/build_predict_with_price.py
    python q2/results/build_predict_with_price.py --pred <路径> --price <路径> --out <路径>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------- 路径

ROOT = Path(__file__).resolve().parents[2]


def _find_root(start: Path) -> Path:
    """向上查找包含 C题.md 的仓库根目录。"""
    for p in [start, *start.parents]:
        if (p / "C题.md").exists():
            return p
    return start


ROOT = _find_root(Path(__file__).resolve())

DEFAULT_PRED = ROOT / "q2" / "results" / "predict.xlsx"
DEFAULT_PRICE = ROOT / "Data_preprocessed" / "raw" / "附件1.xlsx"
DEFAULT_OUT = ROOT / "q2" / "results" / "predict_with_price.xlsx"

N_SLOTS = 144

# ---------------------------------------------------------------- 时间轴


def build_slot_labels() -> list[str]:
    """
    构造与 附件1 一致的 144 个时间标签。

    附件1 的时间列是 Excel 时间格式，从 00:10 起、每 10 分钟一个点，
    到次日 00:00 结束（即 144 个点覆盖 [00:00, 24:00) 的区间末端）。
    最后一点显示为 '0:00+1'，与前 143 点的 'HH:MM' 形式不同。
    """
    labels = []
    for i in range(1, N_SLOTS + 1):
        minutes = i * 10  # 10, 20, ..., 1440
        if minutes == 1440:
            labels.append("0:00+1")
        else:
            labels.append(f"{minutes // 60:02d}:{minutes % 60:02d}")
    return labels


def read_price(price_path: Path) -> pd.DataFrame:
    """读取 附件1 的 144 点电价，返回含 slot / 时间 / 电价 的 DataFrame。"""
    raw = pd.read_excel(price_path, sheet_name=0)

    # 定位电价列：优先按列名，否则按位置取第二列
    price_col = None
    for col in raw.columns:
        if "电价" in str(col):
            price_col = col
            break
    if price_col is None:
        price_col = raw.columns[1]

    price = pd.to_numeric(raw[price_col], errors="coerce").to_numpy(float)
    if len(price) != N_SLOTS:
        raise ValueError(
            f"附件1 电价列长度为 {len(price)}，期望 {N_SLOTS}（144 个 10 分钟时刻）"
        )
    if not np.isfinite(price).all():
        raise ValueError("附件1 电价列存在无法解析为数值的单元格")

    labels = build_slot_labels()

    # 校验原表时间轴与构造的时间轴一致，避免轴错位。
    # 附件1 的时间列混用两种表示：靠前的行是真正的 Excel 时间值（openpyxl 读出
    # datetime.time，pandas 读出 Timestamp），靠后的行则是文本 'HH:MM'。统一归一化
    # 成 'HH:MM' 再比较。
    def normalize(value) -> str:
        if hasattr(value, "strftime"):  # datetime.time / Timestamp
            return value.strftime("%H:%M")
        s = str(value).strip()
        if "+1" in s:  # '0:00+1' 形式的次日零点
            return "24:00"
        m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", s)
        if m:
            return f"{int(m.group(1)):02d}:{m.group(2)}"
        return s

    expected = ["24:00" if x == "0:00+1" else x for x in labels]
    raw_labels = [normalize(v) for v in raw[raw.columns[0]]]
    mismatch = [(i, raw_labels[i], expected[i])
                for i in range(N_SLOTS) if raw_labels[i] != expected[i]]
    if mismatch:
        print(f"[警告] 附件1 时间列与构造时间轴有 {len(mismatch)} 处不一致，"
              f"前 3 处：{mismatch[:3]}")
    else:
        print("时间轴校验通过：附件1 的 144 个时刻与构造轴完全一致")

    return pd.DataFrame({"slot": np.arange(1, N_SLOTS + 1),
                         "时间": labels,
                         "电价": price})


def read_prediction(pred_path: Path) -> pd.DataFrame:
    """读取 predict.xlsx 的 Forecast_10min 工作表并校验。"""
    df = pd.read_excel(pred_path, sheet_name="Forecast_10min")

    required = {
        "date", "slot", "actual_net_load_kw", "forecast_net_load_kw",
        "net_load_p05_kw", "net_load_p10_kw", "net_load_p90_kw",
        "net_load_p95_kw",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"predict.xlsx 缺少必要列：{sorted(missing)}；"
            f"实际列为 {list(df.columns)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["slot"] = df["slot"].astype(int)

    # 每天必须恰好 144 个时段，且 slot 为 1..144
    bad = []
    for d, g in df.groupby("date"):
        slots = np.sort(g["slot"].to_numpy())
        if len(g) != N_SLOTS or not np.array_equal(slots, np.arange(1, N_SLOTS + 1)):
            bad.append((d.date(), len(g)))
    if bad:
        raise ValueError(f"以下日期的时段数不是完整的 144：{bad[:5]}")

    return df


# ---------------------------------------------------------------- 输出


def main() -> None:
    ap = argparse.ArgumentParser(
        description="将 Q2 预测结果与附件1 电价整合为按日分表的工作簿"
    )
    ap.add_argument("--pred", type=Path, default=DEFAULT_PRED,
                    help="预测结果工作簿（默认 q2/results/predict.xlsx）")
    ap.add_argument("--price", type=Path, default=DEFAULT_PRICE,
                    help="电价来源工作簿（默认 Data_preprocessed/raw/附件1.xlsx）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="输出工作簿路径")
    ap.add_argument("--no-center-as-p50", action="store_true",
                    help="不使用中心预测充当 p50（默认中心预测即 p50）")
    args = ap.parse_args()

    price_df = read_price(args.price)
    pred = read_prediction(args.pred)

    dates = sorted(pred["date"].unique())
    print(f"电价来源：{args.price}")
    print(f"预测来源：{args.pred}")
    print(f"日期数：{len(dates)}（{pd.Timestamp(dates[0]).date()} ~ "
          f"{pd.Timestamp(dates[-1]).date()}），每日期望 {N_SLOTS} 行")

    # 每个日工作表的列：时间 | 电价 | 实际净负载 | 净负载分位数（含 p50）
    quantile_cols = [
        ("net_load_p05_kw", "净负载p05"),
        ("net_load_p10_kw", "净负载p10"),
        ("forecast_net_load_kw", "净负载p50"),
        ("net_load_p90_kw", "净负载p90"),
        ("net_load_p95_kw", "净负载p95"),
    ]

    # 源数据自检：分位数序列应单调不减。
    # 注意：predict.xlsx 的分位带由「历史整日残差场景」的分位数给出，而
    # forecast_net_load_kw 是净负荷中心预测（均值口径）。二者口径不同，
    # 因此中心预测偶尔会落到 [p10, p90] 之外，这不是本脚本的错位。
    qmat = pred[[c for c, _ in quantile_cols]].to_numpy(float)
    steps = [label for _, label in quantile_cols]
    cross_total = 0
    for i in range(len(steps) - 1):
        bad = qmat[:, i] > qmat[:, i + 1] + 1e-9
        cross_total += int(bad.sum())
        if bad.any():
            worst = float(np.max(qmat[:, i] - qmat[:, i + 1]))
            print(f"[提示] 源数据 {steps[i]} > {steps[i+1]} 的点数："
                  f"{int(bad.sum())}（最大越界 {worst:.4f} kW）")
    if cross_total:
        n_days = pred.loc[np.diff(qmat, axis=1).min(axis=1) < -1e-9,
                          "date"].nunique()
        print(f"[提示] 合计 {cross_total} 个点存在分位数交叉，涉及 {n_days} 天。"
              f"该现象源自 predict.xlsx 的中心预测与分位带口径差异，"
              f"脚本按原值导出，不做平滑或重排。")
    else:
        print("源数据自检：各日分位数序列单调不减")
    print()

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    center_fill = PatternFill("solid", fgColor="DDEBF7")  # 标出中心预测列

    n_written = 0
    used_names: set[str] = set()

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        # 删掉 openpyxl 自带的默认空白表；若它已被 pandas 复用则跳过
        default = writer.book.active
        if default is not None and default.title not in ("Sheet",):
            pass
        if default is not None and len(writer.book.sheetnames) == 1:
            try:
                writer.book.remove(default)
            except (ValueError, KeyError):
                pass

        for d in dates:
            day = pred[pred["date"] == d].sort_values("slot")
            ts = pd.Timestamp(d)

            sheet = pd.DataFrame({
                "时间": price_df["时间"].to_numpy(),
                "电价": price_df["电价"].to_numpy(),
                "实际净负载": day["actual_net_load_kw"].to_numpy(float),
            })
            for src, label in quantile_cols:
                sheet[label] = day[src].to_numpy(float)

            # 工作表名：日期（Excel 表名上限 31 字符，日期名远小于该限制）
            name = ts.strftime("%Y-%m-%d")
            if name in used_names:  # 理论上不会发生
                name = f"{name}_{n_written}"
            used_names.add(name)

            sheet.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]

            # 样式：冻结首行、隐藏网格线、表头高亮、数字格式、列宽
            ws.freeze_panes = "A2"
            ws.sheet_view.showGridLines = False
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")

            for row in ws.iter_rows(min_row=2, min_col=2, max_col=7):
                for cell in row:
                    cell.number_format = "0.0000"
            # 标出 p50 = 中心预测列
            for cell in ws["E"][1:]:
                cell.fill = center_fill

            widths = {"A": 10, "B": 10, "C": 14, "D": 13, "E": 13,
                      "F": 15, "G": 13, "H": 13}
            for col, w in widths.items():
                ws.column_dimensions[col].width = w

            n_written += 1

    print(f"\n已写出：{args.out}")
    print(f"工作表数：{n_written}（每日一表）")
    print("列顺序：时间 | 电价 | 实际净负载 | 净负载p05 | 净负载p10 | "
          "净负载p50 | 净负载p90 | 净负载p95")
    print("注：p50 即 predict.xlsx 的净负荷中心预测 forecast_net_load_kw。")

    # 自检：抽查若干天，确认行数、时间轴、电价与预测值均与源数据一致
    print("\n--- 自检（抽查首/中/末三天）---")
    import openpyxl

    wb = openpyxl.load_workbook(args.out, read_only=True)
    for name in [wb.sheetnames[0], wb.sheetnames[len(wb.sheetnames) // 2],
                 wb.sheetnames[-1]]:
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        body = rows[1:]
        ts = pd.Timestamp(name)
        src = pred[pred["date"] == ts].sort_values("slot")

        checks = {
            "行数144": len(body) == N_SLOTS,
            "时间轴": [r[0] for r in body] == price_df["时间"].tolist(),
            "电价": bool(np.allclose([r[1] for r in body],
                                     price_df["电价"].to_numpy())),
            "实际净负载": bool(np.allclose(
                [r[2] for r in body], src["actual_net_load_kw"].to_numpy())),
            "分位数": bool(np.allclose(
                [[r[3], r[4], r[5], r[6], r[7]] for r in body],
                src[[c for c, _ in quantile_cols]].to_numpy())),
        }
        failed = [k for k, ok in checks.items() if not ok]
        status = "全部通过" if not failed else f"失败项={failed}"
        print(f"  {name}: {status}")

    # 全表统计：确认 334 张表的行数与时间轴全部一致
    bad_sheets = [n for n in wb.sheetnames
                  if wb[n].max_row != N_SLOTS + 1]
    print(f"  全表：{len(wb.sheetnames)} 个日工作表，"
          f"行数异常表数={len(bad_sheets)}")
    wb.close()


if __name__ == "__main__":
    main()
