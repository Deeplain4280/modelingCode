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
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

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
    # 使用 Word 内置 Heading 1/2/3 样式，保留大纲级别以便自动生成目录
    heading_sizes = {"Heading 1": 18, "Heading 2": 14, "Heading 3": 12}
    for name, size in heading_sizes.items():
        heading = document.styles[name]
        heading.font.name = LATIN_FONT
        heading.font.size = Pt(size)
        heading.font.bold = True
        heading.font.color.rgb = RGBColor(0, 0, 0)
        heading.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
        heading.paragraph_format.space_before = Pt(10 if size >= 14 else 6)
        heading.paragraph_format.space_after = Pt(4)
        heading.paragraph_format.line_spacing = 1.5
        heading.paragraph_format.keep_with_next = True
    add_page_number_footer(document)
    return document


def add_page_number_footer(document: Document) -> None:
    """页脚居中页码（PAGE 域）。"""
    for section in document.sections:
        footer_paragraph = section.footer.paragraphs[0]
        footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer_paragraph.add_run()
        set_run_font(run, BODY_FONT, 9)
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = " PAGE "
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.append(begin)
        run._r.append(instr)
        run._r.append(end)


def add_heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
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
    inline = paragraph.add_run().add_picture(str(path), width=Cm(14.0))
    # 图片替代文本，便于无障碍阅读与 Word“可选文字”检查
    descr = caption or path.stem.replace("_", " ")
    inline._inline.docPr.set("descr", descr)
    inline._inline.docPr.set("title", descr)
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
    # 长表跨页时重复表头
    header_row = table.rows[0]
    tr_pr = header_row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
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


def convert(source: Path = SOURCE, target: Path = TARGET,
            stop_before: str | None = None, start_at: str | None = None) -> Path:
    document = setup_document()
    lines = source.read_text(encoding="utf-8").splitlines()
    if start_at is not None:
        for position, candidate in enumerate(lines):
            if candidate.strip() == start_at:
                lines = lines[position:]
                break
    if stop_before is not None:
        for position, candidate in enumerate(lines):
            if candidate.strip() == stop_before:
                lines = lines[:position]
                break
    index = 0
    pending_image = None  # 独立的图片说明行
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue
        if line.startswith("<!--") and line.endswith("-->"):
            # Markdown 注释：如表格展开标记，仅用于脚本定位，不写入 Word
            index += 1
            continue
        if line.startswith("```"):
            # 代码块：等宽字体、不首行缩进，逐行保留
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            for code_line in code_lines:
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.paragraph_format.left_indent = Pt(12)
                run = paragraph.add_run(code_line if code_line else " ")
                run.font.name = "Consolas"
                run.font.size = Pt(9)
            index += 1
            continue
        if line == "$$":
            # 独立公式块：居中、斜体，公式内容按原文保留
            index += 1
            formula_lines = []
            while index < len(lines) and lines[index].strip() != "$$":
                formula_lines.append(lines[index].strip())
                index += 1
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1.5
            run = paragraph.add_run(" ".join(formula_lines))
            run.font.name = "Times New Roman"
            run.font.size = Pt(11)
            run.italic = True
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
    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(target))
    print(f"已生成：{target}")
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Markdown 论文 → Word")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--stop-before", default=None, help="遇到该行（strip 后完全相等）即停止")
    parser.add_argument("--start-at", default=None, help="从该行（strip 后完全相等）开始")
    args = parser.parse_args()
    convert(
        source=args.source,
        target=args.target,
        stop_before=args.stop_before,
        start_at=args.start_at,
    )
