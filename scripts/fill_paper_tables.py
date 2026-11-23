# -*- coding: utf-8 -*-
"""把结果 CSV 展开成 Markdown 表格，写入 论文/论文.md 的占位区域。

论文.md 中用如下标记圈定表格位置（脚本可重复执行，区域内容每次重建）：

    <!-- BEGIN_TABLE:表4-3_附件2预测结果.csv -->
    <!-- END_TABLE:表4-3_附件2预测结果.csv -->

用法：
    python code/scripts/fill_paper_tables.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
PAPER = WORKSPACE / "论文" / "论文.md"
TABLES = WORKSPACE / "code" / "outputs" / "tables"

COLUMN_ALIASES = {
    "排名": "排名",
    "用户编号": "用户编号",
    "预测购买概率": "预测概率",
    "潜力评分": "潜力评分",
    "潜力等级": "潜力等级",
    "主要正向驱动": "主要正向驱动",
    "建议动作": "建议动作",
    "是否在附件1目标用户中": "在附件1中",
    "附件1购买标记": "附件1已购",
    "画像标签": "画像标签",
    "附件2序号": "附件2序号",
}
ROUND_RULES = {"预测购买概率": 4, "预测概率": 4, "潜力评分": 1}
KEEP_ORDER = [
    "排名",
    "用户编号",
    "预测概率",
    "潜力评分",
    "潜力等级",
    "画像标签",
    "主要正向驱动",
    "建议动作",
    "在附件1中",
]


def format_table(csv_path: Path) -> str:
    frame = pd.read_csv(csv_path)
    frame = frame.rename(columns=COLUMN_ALIASES)
    columns = [column for column in KEEP_ORDER if column in frame.columns]
    frame = frame[columns].copy()
    for column, digits in ROUND_RULES.items():
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: f"{value:.{digits}f}" if pd.notna(value) else ""
            )
    for column in frame.columns:
        frame[column] = frame[column].fillna("").astype(str).str.replace("|", "/", regex=False)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    text = PAPER.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(<!-- BEGIN_TABLE:([^>]+?) -->)(.*?)(<!-- END_TABLE:\2 -->)", re.DOTALL
    )

    def replace(match: re.Match[str]) -> str:
        begin, name, _, end = match.groups()
        csv_path = TABLES / name.strip()
        if not csv_path.exists():
            return match.group(0)
        return f"{begin}\n{format_table(csv_path)}\n{end}"

    replaced, count = pattern.subn(replace, text)
    PAPER.write_text(replaced, encoding="utf-8")
    print(f"已展开 {count} 个表格区域：{PAPER}")


if __name__ == "__main__":
    main()
