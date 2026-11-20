"""
FactIndic - Word version of the paper (paper/FactIndic_IEEE.docx)
=================================================================
The LaTeX source (main.tex + sections/*.tex) is the master copy. This script
converts it to Word and then applies the IEEE conference template layout:

  1. flatten main.tex (inline the sections, simplify the few XeTeX-only
     macros) and convert it with pandoc (+ citeproc, IEEE reference style);
  2. post-process with python-docx:
       - US Letter, IEEE margins (top 0.75", bottom 1", left/right 0.625");
       - title and authors in one column, the body in two columns
         (3.5" each, 0.25" gap), the two wide tables in one-column blocks;
       - Times New Roman 10 pt body, justified, first-line indent;
       - IEEE headings: "I. INTRODUCTION" (centred small caps),
         "A. Subsection" (italic), run-in "a) Paragraph:" (italic);
       - abstract and index terms in bold 9 pt, captions 8 pt,
         "TABLE I" captions above tables, "Fig. 1." captions below figures,
         references 8 pt.

Needs pandoc (brew install pandoc) and python-docx (pip install python-docx).
Run from the repository root:
    python paper/make_docx.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "FactIndic_IEEE.docx"
CSL = PAPER / "ieee.csl"  # IEEE style from github.com/citation-style-language/styles
FONT = "Times New Roman"

TITLE = "FactIndic: Factual Consistency of LLM Summaries for Hindi, Romanized Hindi and Hinglish"
AUTHORS = "Arman Sahni, Ananya Mishra, Gauri Mahajan, Krati Khanna, Arushi Chakraborty"
AFFILIATION = "Department of Computer Science and Engineering, Bennett University, Greater Noida, India"
EMAILS = "{s24cseu1430, s24cseu1431, s24cseu1435, s24cseu0024, s24cseu1269}@bennett.edu.in"


# ---------------------------------------------------------------- 1. LaTeX -> docx
def flatten() -> str:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    tex = re.sub(r"\\input\{(sections/[^}]+)\}",
                 lambda m: (PAPER / f"{m.group(1)}.tex").read_text(encoding="utf-8"), tex)
    # XeTeX-only Devanagari font: pandoc keeps the Unicode text as it is
    tex = re.sub(r"\\font\\devfont=.*\n", "", tex)
    tex = tex.replace(r"\newcommand{\dev}[1]{{\devfont #1}}", r"\newcommand{\dev}[1]{#1}")
    # title block and keywords are written by the post-processing step
    tex = re.sub(r"\\title\{.*?\}\n\n\\author\{.*?\n\}\n", "", tex, flags=re.S)
    tex = tex.replace(r"\maketitle", "")
    tex = re.sub(r"\\begin\{abstract\}\s*", r"ABSTRACTSTART ", tex)
    tex = tex.replace(r"\end{abstract}", "")
    tex = re.sub(r"\\begin\{IEEEkeywords\}\s*(.*?)\s*\\end\{IEEEkeywords\}", r"KEYWORDSSTART \1", tex, flags=re.S)
    # figures: use the PNG versions
    tex = re.sub(r"\\includegraphics\[[^]]*\]\{(fig_[a-z_]+)\}", r"\\includegraphics{figures/\1.png}", tex)
    tex = re.sub(r"\\bibliographystyle\{.*?\}\s*\\bibliography\{.*?\}", "", tex)
    # wide tables: pandoc loses table* captions and \multirow cells
    tex = tex.replace(r"\begin{table*}", r"\begin{table}").replace(r"\end{table*}", r"\end{table}")
    tex = re.sub(r"\\multirow\{\d+\}\{\*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", tex)
    tex = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", tex)
    tex = tex.replace("$n$", r"\emph{n}")  # pandoc drops lone math in captions
    return resolve_section_refs(tex)


def resolve_section_refs(tex: str) -> str:
    """\\ref to a section -> IEEE numbering ("VI", "VI-E"), as in the PDF;
    pandoc would print "6.5". Figure/table references are left to pandoc."""
    names, sec, sub = {}, 0, 0
    for m in re.finditer(r"\\(section|subsection)\*?\{[^}]*\}\s*(?:\\label\{([^}]+)\})?", tex):
        starred = m.group(0).split("{")[0].endswith("*")
        if m.group(1) == "section" and not starred:
            sec, sub = sec + 1, 0
            label = roman(sec)
        elif m.group(1) == "subsection":
            sub += 1
            label = f"{roman(sec)}-{chr(64 + sub)}"
        else:
            continue
        if m.group(2):
            names[m.group(2)] = label
    return re.sub(r"\\ref\{(sec:[^}]+)\}", lambda m: names.get(m.group(1), "?"), tex)


def run_pandoc(tex: str, out: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".tex", dir=PAPER, delete=False, encoding="utf-8") as f:
        f.write(tex)
        src = Path(f.name)
    try:
        subprocess.run(["pandoc", src.name, "-f", "latex", "-t", "docx", "-o", str(out),
                        "--citeproc", "--bibliography", "references.bib", "--csl", str(CSL),
                        "-M", "reference-section-title=References", "-M", "link-citations=false",
                        "--resource-path", "."], cwd=PAPER, check=True)
    finally:
        src.unlink()


# ---------------------------------------------------------------- 2. layout helpers
def set_font(run, size=None, bold=None, italic=None, small_caps=None):
    run.font.name = FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if small_caps is not None:
        run.font.small_caps = small_caps


def fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0.0, before=0, after=0, **run_kw):
    p.alignment = align
    pf = p.paragraph_format
    pf.first_line_indent = Inches(indent) if indent else None
    pf.left_indent = None
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.0
    for r in p.runs:
        set_font(r, size, **run_kw)


def set_columns(section, num: int):
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(0.25 * 1440)))


def section_break_after(paragraph, columns: int):
    """End a section at this paragraph (continuous break); the section that
    ENDS here gets `columns` columns."""
    pPr = paragraph._p.get_or_add_pPr()
    sectPr = OxmlElement("w:sectPr")
    t = OxmlElement("w:type")
    t.set(qn("w:val"), "continuous")
    sectPr.append(t)
    size = OxmlElement("w:pgSz")
    size.set(qn("w:w"), "12240")
    size.set(qn("w:h"), "15840")
    sectPr.append(size)
    mar = OxmlElement("w:pgMar")
    for k, v in {"top": 1080, "bottom": 1440, "left": 900, "right": 900, "header": 720, "footer": 720,
                 "gutter": 0}.items():
        mar.set(qn(f"w:{k}"), str(v))
    sectPr.append(mar)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(columns))
    cols.set(qn("w:space"), "360")
    sectPr.append(cols)
    pPr.append(sectPr)


def insert_paragraph_before(par, text=""):
    new = par.insert_paragraph_before(text)
    return new


def has_drawing(element) -> bool:
    return next(element.iter(qn("w:drawing")), None) is not None


def roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out, n = out + s, n - v
    return out


# ---------------------------------------------------------------- 3. IEEE layout
def ieee_layout(path: Path):
    doc = Document(path)
    body = doc.paragraphs

    # page setup for the final section (the body)
    sec = doc.sections[-1]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
    sec.left_margin = sec.right_margin = Inches(0.625)
    set_columns(sec, 2)

    # default style
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = FONT, Pt(10)

    sec_no, sub_no, par_no = 0, 0, 0
    fig_no, tab_no = 0, 0
    in_refs = False
    for p in list(doc.paragraphs):
        style = p.style.name
        text = p.text.strip()
        if text.startswith("ABSTRACTSTART"):
            for r in p.runs:
                r.text = r.text.replace("ABSTRACTSTART ", "").replace("ABSTRACTSTART", "")
            lead = p.runs[0]._r
            r = p.add_run("Abstract\u2014")
            lead.addprevious(r._r)
            fmt(p, size=9, indent=0.2, after=6, bold=True)
            r.italic = True
            first_body = p
            continue
        if text.startswith("KEYWORDSSTART"):
            for r in p.runs:
                r.text = r.text.replace("KEYWORDSSTART ", "").replace("KEYWORDSSTART", "")
            lead = p.runs[0]._r
            r = p.add_run("Index Terms\u2014")
            lead.addprevious(r._r)
            fmt(p, size=9, indent=0.2, after=6, bold=True)
            r.italic = True
            continue
        if style.startswith("Heading 1") or (style == "Heading 1"):
            if text == "References":
                in_refs = True
                p.text = "References"
            elif text.startswith("Acknowledg"):
                p.text = "Acknowledgment"
            else:
                sec_no, sub_no, par_no = sec_no + 1, 0, 0
                p.text = f"{roman(sec_no)}. {text}"
            fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, small_caps=True,
                bold=False, italic=False)
            p.paragraph_format.keep_with_next = True
            continue
        if style.startswith("Heading 2"):
            sub_no, par_no = sub_no + 1, 0
            p.text = f"{chr(64 + sub_no)}. {text}"
            fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=3, italic=True, bold=False)
            p.paragraph_format.keep_with_next = True
            continue
        if style.startswith("Heading"):  # \paragraph -> run-in "a) Title:"
            par_no += 1
            p.text = f"{chr(96 + par_no)}) {text}:"
            fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, indent=0.2, before=2, italic=True, bold=False)
            p.paragraph_format.keep_with_next = True
            continue
        if style == "Image Caption":
            fig_no += 1
            r = p.add_run(f"Fig. {fig_no}. ")
            p.runs[0]._r.addprevious(r._r)
            fmt(p, size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6)
            continue
        if style == "Table Caption":
            tab_no += 1
            caption_text = p.text
            p.text = f"TABLE {roman(tab_no)}"
            fmt(p, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, small_caps=True)
            p.paragraph_format.keep_with_next = True
            sub = p.insert_paragraph_before("")
            p._p.addnext(sub._p)  # move below "TABLE I"
            sub.add_run(caption_text)
            fmt(sub, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=3, small_caps=True)
            sub.paragraph_format.keep_with_next = True
            continue
        if has_drawing(p._p):  # a figure
            for blip in p._p.iter(qn("wp:extent")):
                cx, cy = int(blip.get("cx")), int(blip.get("cy"))
                w = Inches(3.45)
                blip.set("cx", str(int(w))), blip.set("cy", str(int(cy * w / cx)))
            for ext in p._p.iter(qn("a:ext")):
                ext.set("cx", str(int(Inches(3.45)))) if ext.get("cx") else None
            fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=2)
            p.paragraph_format.keep_with_next = True
            continue
        if in_refs or style == "Bibliography":
            fmt(p, size=8, align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
            continue
        if style in ("Compact",) or p._p.getparent().tag == qn("w:tc"):
            continue
        fmt(p, size=10, indent=0.2 if style in ("Body Text", "First Paragraph", "Normal") else 0)

    # images keep the aspect ratio: fix the picture's own extent too
    for pic in doc.inline_shapes:
        if pic.width > Inches(3.45):
            ratio = Inches(3.45) / pic.width
            pic.width, pic.height = Inches(3.45), int(pic.height * ratio)

    # tables: drop empty trailing columns (from the LaTeX column spec)
    for t in doc.tables:
        while all(not row.cells[-1].text.strip() for row in t.rows) and len(t.columns) > 1:
            for row in t.rows:
                tc = row.cells[-1]._tc
                tc.getparent().remove(tc)
            grid = t._tbl.tblGrid
            grid.remove(grid.findall(qn("w:gridCol"))[-1])

    # tables: 7 pt, centred, each in a one-column block
    for t in doc.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for row in t.rows:
            for cell in row.cells:
                for cp in cell.paragraphs:
                    fmt(cp, size=7, align=WD_ALIGN_PARAGRAPH.CENTER)
        # column widths in proportion to the longest text in each column
        total = Inches(7.25 if len(t.columns) > 4 else 3.4)
        longest = [max(5, max(len(row.cells[j].text) for row in t.rows) + 2) for j in range(len(t.columns))]
        t.autofit = False
        grid = t._tbl.tblGrid.findall(qn("w:gridCol"))
        for j, n in enumerate(longest):
            w = int(total * n / sum(longest))
            grid[j].set(qn("w:w"), str(int(w / 635)))  # EMU -> twips
            for row in t.rows:
                row.cells[j].width = w
        if len(t.columns) <= 4:
            continue  # a narrow table stays in its column
        # find the caption paragraphs right above the table and wrap
        # [caption, table] in a one-column section
        before = t._tbl.getprevious()
        while before is not None and before.tag == qn("w:p") and "TABLE" not in "".join(before.itertext()):
            before = before.getprevious()
        start = before.getprevious() if before is not None else None  # last body paragraph before
        while start is not None and start.tag != qn("w:p"):  # skip bookmarks
            start = start.getprevious()
        if start is not None and start.tag == qn("w:p"):
            from docx.text.paragraph import Paragraph
            section_break_after(Paragraph(start, None), columns=2)
        end = OxmlElement("w:p")
        t._tbl.addnext(end)
        from docx.text.paragraph import Paragraph
        section_break_after(Paragraph(end, None), columns=1)

    # title block (one column) at the very top
    first = doc.paragraphs[0]
    title = first.insert_paragraph_before(TITLE)
    fmt(title, size=24, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    for text, size, italic in [(AUTHORS, 11, False), (AFFILIATION, 10, True), (EMAILS, 10, False)]:
        q = first.insert_paragraph_before(text)
        fmt(q, size=size, align=WD_ALIGN_PARAGRAPH.CENTER, italic=italic)
    spacer = first.insert_paragraph_before("")
    fmt(spacer, after=6)
    section_break_after(spacer, columns=1)

    # every page-margin element needs header/footer/gutter (OOXML schema)
    for mar in doc.element.body.iter(qn("w:pgMar")):
        for k, v in {"header": "720", "footer": "720", "gutter": "0"}.items():
            if mar.get(qn(f"w:{k}")) is None:
                mar.set(qn(f"w:{k}"), v)

    doc.core_properties.title = TITLE
    doc.core_properties.author = AUTHORS
    doc.save(path)


def main():
    if not CSL.exists():
        sys.exit(f"Missing {CSL}: download ieee.csl from github.com/citation-style-language/styles")
    run_pandoc(flatten(), OUT)
    ieee_layout(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
