"""Convert docs/report.md to docs/report.pdf using fpdf2 with markdown support."""

import pathlib
import re

from fpdf import FPDF
from fpdf.enums import XPos, YPos

DOCS_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs"
MD_PATH = DOCS_DIR / "report.md"
PDF_PATH = DOCS_DIR / "report.pdf"


def _sanitize(text: str) -> str:
    """Replace unicode characters that Helvetica cannot encode."""
    replacements = {
        "\u2014": "--",   # em-dash
        "\u2013": "-",    # en-dash
        "\u2022": "-",    # bullet
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u2192": "->",   # right arrow
        "\u2264": "<=",   # less than or equal
        "\u2265": ">=",   # greater than or equal
        "\u00d7": "x",    # multiplication sign
        "\u251c": "|",    # ├
        "\u2500": "-",    # ─
        "\u2502": "|",    # │
        "\u2514": "`",    # └
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Fallback: replace any remaining non-latin1 chars
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text

class ReportPDF(FPDF):
    """Custom PDF with header/footer and markdown rendering."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Heart Disease Prediction MLOps Report", align="C")
        self.ln(8)
        self.set_draw_color(42, 94, 168)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def parse_table(lines):
    """Parse markdown table lines into header and rows."""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    # rows[0] = header, rows[1] = separator, rows[2:] = data
    if len(rows) >= 2:
        header = rows[0]
        data = [r for i, r in enumerate(rows) if i >= 2]
        return header, data
    return rows[0] if rows else [], []


def render_table(pdf, header, data):
    """Render a table with styled header and alternating row colors."""
    pdf.ln(4)
    num_cols = len(header)
    avail_width = pdf.w - 20
    col_width = avail_width / num_cols

    # Header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(42, 94, 168)
    pdf.set_text_color(255, 255, 255)
    for cell in header:
        pdf.cell(col_width, 7, _sanitize(cell[:30]), border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 8)
    for i, row in enumerate(data):
        if i % 2 == 0:
            pdf.set_fill_color(244, 247, 251)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        for j, cell in enumerate(row):
            pdf.cell(col_width, 6, _sanitize(cell[:30]), border=1, fill=True, align="C" if j > 0 else "L")
        pdf.ln()
    pdf.ln(4)


def render_code_block(pdf, code_lines):
    """Render a fenced code block."""
    pdf.ln(2)
    pdf.set_font("Courier", "", 8)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_text_color(40, 40, 40)
    for line in code_lines:
        # Check if we need a page break
        if pdf.get_y() > pdf.h - 25:
            pdf.add_page()
        pdf.cell(0, 5, "  " + _sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    lines = md_text.split("\n")

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Skip mermaid blocks (can't render in PDF) ---
        if line.strip().startswith("```mermaid"):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, "[Architecture diagram - see report.md for Mermaid source]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            # Skip until closing ```
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1  # skip closing ```
            continue

        # --- Fenced code blocks ---
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            render_code_block(pdf, code_lines)
            i += 1  # skip closing ```
            continue

        # --- Tables ---
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            header, data = parse_table(table_lines)
            render_table(pdf, header, data)
            continue

        # --- Headings ---
        if line.startswith("# "):
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(26, 60, 110)
            pdf.cell(0, 12, _sanitize(line[2:].strip()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_draw_color(26, 60, 110)
            pdf.set_line_width(0.8)
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(4)
            i += 1
            continue

        if line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(42, 94, 168)
            pdf.cell(0, 10, _sanitize(line[3:].strip()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            i += 1
            continue

        if line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(58, 58, 58)
            pdf.cell(0, 8, _sanitize(line[4:].strip()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            i += 1
            continue

        # --- Bullet points ---
        bullet_match = re.match(r"^(\s*)[*-]\s+(.*)", line)
        if bullet_match:
            indent = min(len(bullet_match.group(1)), 8)  # cap indent
            text = bullet_match.group(2).strip()
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            x_offset = 14 + indent * 3
            pdf.set_x(x_offset)
            # Clean markdown formatting
            text = re.sub(r"`(.+?)`", r"\1", text)  # inline code
            text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # links
            text = _sanitize(text)
            cell_width = max(pdf.w - x_offset - 10, 40)
            pdf.multi_cell(cell_width, 6, "-  " + text, align="L", markdown=True)
            i += 1
            continue

        # --- Horizontal rule ---
        if line.strip() == "---":
            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.3)
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(4)
            i += 1
            continue

        # --- Empty lines ---
        if line.strip() == "":
            pdf.ln(3)
            i += 1
            continue

        # --- Normal paragraph text ---
        pdf.set_x(10)  # reset to left margin
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        text = line.strip()
        # Clean markdown formatting
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        text = _sanitize(text)
        pdf.multi_cell(0, 6, text, align="L", markdown=True)
        i += 1

    pdf.output(str(PDF_PATH))
    print(f"PDF saved to: {PDF_PATH}")


if __name__ == "__main__":
    main()
