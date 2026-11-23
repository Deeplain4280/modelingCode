# -*- coding: utf-8 -*-
"""将 论文/论文.md 转换为符合模板版式的 论文/论文.docx。

用法：
    python 代码/scripts/build_paper_docx.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

WORKSPACE = Path(__file__).resolve().parents[2]
PAPER_DIR = WORKSPACE / "论文"
SOURCE = PAPER_DIR / "论文.md"
TARGET = PAPER_DIR / "论文.docx"

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
LATIN_FONT = "Times New Roman"


def set_run_font(run, name: str, size: float | None = None, bold: bool | None = None,
                 italic: bool | None = None, latin: str = LATIN_FONT) -> None:
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_rich_text(paragraph, text: str, size: float = 12, base_bold: bool = False) -> None:
    """处理 **粗体**、`代码`、$公式$ 三类行内标记。"""
    text = text.replace("\\", "")
    pattern = re.compile(r"(\*\*.+?\*\*|`.+?`|\$.+?\$)")
    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            run = paragraph.add_run(text[position:match.start()])
            set_run_font(run, BODY_FONT, size, bold=base_bold)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, BODY_FONT, size, bold=True)
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, BODY_FONT, size - 1, bold=base_bold, latin="Consolas")
        else:
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, BODY_FONT, size, bold=base_bold, italic=True)
        position = match.end()
    if position < len(text):
        run = paragraph.add_run(text[position:])
        set_run_font(run, BODY_FONT, size, bold=base_bold)


def setup_document() -> Document:
    document = Document()
    for section in document.sections:
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.25)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
    style = document.styles["Normal"]
    style.font.name = LATIN_FONT
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    style.paragraph_format.line_spacing = 1.5
    style.paragraph_format.space_after = Pt(3)
    return document


def add_heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(10 if level <= 2 else 6)
    paragraph.paragraph_format.space_after = Pt(4)
    if level == 1:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        set_run_font(run, HEAD_FONT, 18, bold=True)
    elif level == 2:
        run = paragraph.add_run(text)
        set_run_font(run, HEAD_FONT, 14, bold=True)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, HEAD_FONT, 12, bold=True)


def add_paragraph(document: Document, text: str, first_indent: bool = True) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.5
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Pt(24)
    add_rich_text(paragraph, text)


def add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.line_spacing = 1.5
    add_rich_text(paragraph, text, size=11.5)


def add_image(document: Document, path: Path, caption: str | None = None) -> None:
    if not path.exists():
        paragraph = document.add_paragraph()
        run = paragraph.add_run(f"[缺失图片：{path.name}]")
        set_run_font(run, BODY_FONT, 10.5, italic=True)
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Cm(14.0))
    if caption:
        cap = document.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        set_run_font(run, BODY_FONT, 10.5, bold=True)


def add_table(document: Document, rows: list[list[str]]) -> None:
    header, body = rows[0], rows[1:]
    table = document.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, text in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        set_run_font(run, HEAD_FONT, 9, bold=True)
    for row_index, row in enumerate(body, start=1):
        for column_index, text in enumerate(row[: len(header)]):
            cell = table.rows[row_index].cells[column_index]
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if column_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = paragraph.add_run(text)
            set_run_font(run, BODY_FONT, 9)
    document.add_paragraph()


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            continue
        rows.append(cells)
    return rows


def convert() -> Path:
    document = setup_document()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    pending_image = None  # 独立的图片说明行
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            add_heading(document, line[2:].strip(), 1)
        elif line.startswith("## "):
            add_heading(document, line[3:].strip(), 2)
        elif line.startswith("### "):
            add_heading(document, line[4:].strip(), 3)
        elif line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            add_table(document, parse_table(block))
            continue
        elif line.startswith("!["):
            match = re.search(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                add_image(document, (PAPER_DIR / match.group(2)).resolve(), match.group(1) or None)
        elif line.startswith("> "):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Pt(18)
            add_rich_text(paragraph, line[2:], size=10.5)
            for run in paragraph.runs:
                run.italic = True
        elif line.startswith("- ") or line.startswith("* "):
            add_bullet(document, line[2:])
        elif re.match(r"^\d+[.、]\s", line):
            add_paragraph(document, line, first_indent=True)
        elif line == "---":
            pass
        else:
            add_paragraph(document, line)
        index += 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(TARGET))
    print(f"已生成：{TARGET}")
    return TARGET


if __name__ == "__main__":
    convert()
