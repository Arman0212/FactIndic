"""
FactIndic - 8-slide presentation (slides/FactIndic_presentation.pptx)
=====================================================================
16:9, clean academic style: dark title and conclusion slides, white content
slides, large figures from paper/figures (re-rendered at 400 dpi from the
vector PDFs for sharpness), little text, speaker notes on every slide.
Colours follow the paper's figures (Okabe-Ito blue and orange).

Needs python-pptx (pip install python-pptx) and pdftoppm (poppler).
Run from the repository root:
    python slides/make_slides.py
"""

import subprocess
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG_PDF = ROOT / "paper" / "figures"
FIG_PNG = ROOT / "slides" / "figures"
OUT = ROOT / "slides" / "FactIndic_presentation.pptx"

NAVY = RGBColor(0x1F, 0x2A, 0x44)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x5B, 0x64, 0x72)
BLUE = RGBColor(0x00, 0x72, 0xB2)
ORANGE = RGBColor(0xE6, 0x9F, 0x00)
VERMILION = RGBColor(0xB8, 0x5C, 0x00)
GREEN = RGBColor(0x00, 0x7A, 0x5E)
TINT = RGBColor(0xF1, 0xF4, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED_LIGHT = RGBColor(0xB8, 0xC2, 0xD3)
HEAD, BODY = "Cambria", "Calibri"
DEVA = "Nirmala UI"  # Windows; PowerPoint on macOS falls back to Kohinoor Devanagari

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
GITHUB = "github.com/Arman0212/FactIndic"


# ---------------------------------------------------------------- helpers
def figure(name: str) -> Path:
    """High-resolution PNG of a paper figure (made from its PDF)."""
    FIG_PNG.mkdir(parents=True, exist_ok=True)
    png = FIG_PNG / f"{name}.png"
    if not png.exists() or png.stat().st_mtime < (FIG_PDF / f"{name}.pdf").stat().st_mtime:
        subprocess.run(["pdftoppm", "-png", "-r", "400", "-singlefile", str(FIG_PDF / f"{name}.pdf"),
                        str(png.with_suffix(""))], check=True)
    return png


def text(slide, x, y, w, h, runs, size=18, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, spacing=6):
    """A text box. runs: a string, or a list of paragraphs; a paragraph is a
    string or a list of (text, {options}) runs."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paragraphs = [runs] if isinstance(runs, str) else runs
    for k, par in enumerate(paragraphs):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        for t, opt in ([(par, {})] if isinstance(par, str) else par):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", italic)
            f.color.rgb = opt.get("color", color)
            if "url" in opt:
                r.hyperlink.address = opt["url"]
            if "underline" in opt:
                f.underline = opt["underline"]
    return box


def box(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    s.shadow.inherit = False
    return s


def picture(slide, name, x, y, w=None, h=None):
    """Insert a figure; give w or h (aspect ratio kept), or both as a bounding box."""
    path = figure(name)
    iw, ih = Image.open(path).size
    if w and h:  # fit into the box, centred
        scale = min(w / iw, h / ih)
        pw, ph = int(iw * scale), int(ih * scale)
        return slide.shapes.add_picture(str(path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)
    return slide.shapes.add_picture(str(path), x, y, w, h)


def title(slide, t, sub=None):
    text(slide, MARGIN, Inches(0.45), W - 2 * MARGIN, Inches(0.8), t, size=34, font=HEAD, bold=True,
         color=NAVY)
    if sub:
        text(slide, MARGIN, Inches(1.2), W - 2 * MARGIN, Inches(0.5), sub, size=18, color=MUTED)


def footer(slide, n, dark=False):
    c = RGBColor(0xB8, 0xC2, 0xD3) if dark else MUTED
    text(slide, MARGIN, H - Inches(0.45), Inches(8), Inches(0.3), "FactIndic  ·  CSET 346, Bennett University",
         size=10, color=c)
    text(slide, W - MARGIN - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(n), size=10, color=c,
         align=PP_ALIGN.RIGHT)


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color


def stat(slide, x, y, w, big, label, color=NAVY, size=54):
    text(slide, x, y, w, Inches(0.95), big, size=size, font=HEAD, bold=True, color=color)
    text(slide, x, y + Inches(0.95), w, Inches(0.8), label, size=15, color=INK)


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = MUTED
    c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
                          {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


# ---------------------------------------------------------------- slides
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, NAVY)

    # 1. Project Title
    text(s, MARGIN, Inches(1.10), W - 2 * MARGIN, Inches(1.8),
         [[("FactIndic", {"color": ORANGE, "size": 42})],
          [("Factuality-Aware Summarization for Hindi and Code-Mixed Hindi-English",
            {"color": WHITE, "size": 26, "font": HEAD, "bold": True})]],
         size=26, font=HEAD, bold=True, color=WHITE, spacing=8)

    # 2. Team members with enrolment numbers (clean 2-column list)
    col_w = Inches(5.8)
    col1 = [
        [("Arman Sahni", {"bold": True, "color": WHITE}),
         (" - S24CSEU1430", {"color": MUTED_LIGHT, "bold": False})],
        [("Ananya Mishra", {"bold": True, "color": WHITE}),
         (" - S24CSEU1431", {"color": MUTED_LIGHT, "bold": False})],
        [("Gauri Mahajan", {"bold": True, "color": WHITE}),
         (" - S24CSEU1435", {"color": MUTED_LIGHT, "bold": False})],
    ]
    col2 = [
        [("Krati Khanna", {"bold": True, "color": WHITE}),
         (" - S24CSEU0024", {"color": MUTED_LIGHT, "bold": False})],
        [("Arushi Chakraborty", {"bold": True, "color": WHITE}),
         (" - S24CSEU1269", {"color": MUTED_LIGHT, "bold": False})],
    ]
    text(s, MARGIN, Inches(3.25), col_w, Inches(1.3), col1, size=17, spacing=8)
    text(s, MARGIN + col_w + Inches(0.4), Inches(3.25), col_w, Inches(1.3), col2, size=17, spacing=8)

    # 3. Public repository link (clickable)
    repo_url = f"https://{GITHUB}"
    text(s, MARGIN, Inches(4.75), W - 2 * MARGIN, Inches(0.5),
         [[("Code, paper & slides:  ", {"color": MUTED_LIGHT, "size": 17}),
           (repo_url, {"color": ORANGE, "bold": True, "size": 17, "url": repo_url, "underline": True})]],
         size=17)

    # 4. Course line
    text(s, MARGIN, Inches(5.75), W - 2 * MARGIN, Inches(0.9),
         ["CSET 346 Natural Language Processing  ·  Bennett University",
          "Instructor: Dr. Geetanjali Singh"],
         size=14, color=MUTED_LIGHT, spacing=4)

    # Speaker notes
    s.notes_slide.notes_text_frame.text = (
        "Good morning. We are presenting FactIndic, a study of whether LLM summaries stay faithful to "
        "the article when the input is Hindi, Romanized Hindi or Hinglish. "
        "Our team members are Arman Sahni, Ananya Mishra, Gauri Mahajan, Krati Khanna, and Arushi Chakraborty. "
        f"All project resources—including our code, paper, and presentation deck—are available in one place on GitHub at {repo_url}. "
        "Everything is training-free: we only prompt existing models and evaluate their outputs.")


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Same facts, three forms of Hindi",
          "Does the script or code-mixing change how faithful LLM summaries are?")
    rows = [
        ("hi", "Devanagari Hindi", "हालांकि बिहार सरकार की ओर से 7,200 करोड़ के नुक़सान का ज्ञापन दिया गया था", DEVA, BLUE),
        ("hi_roman", "Romanized Hindi: same words, Roman script", "halanki bihar sarakar ki or se 7,200 karod ke nuqasan ka jnapan diya gaya tha",
         BODY, ORANGE),
        ("hinglish", "Hinglish: code-mixed rewrite (mean CMI 17.6)",
         "Halanki Bihar government ki or se 7,200 crore ke nuksan ka gyaapan diya gaya tha", BODY, GREEN),
    ]
    y = Inches(1.95)
    for tag, desc, example, font, color in rows:
        box(s, MARGIN, y, W - 2 * MARGIN, Inches(1.3))
        tagbox = box(s, MARGIN + Inches(0.3), y + Inches(0.35), Inches(1.6), Inches(0.6), fill=color)
        tf = tagbox.text_frame
        tf.text = tag
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(18), True, BODY, WHITE
        text(s, MARGIN + Inches(2.2), y + Inches(0.18), Inches(9.6), Inches(0.4), desc, size=15, color=MUTED)
        text(s, MARGIN + Inches(2.2), y + Inches(0.6), Inches(9.7), Inches(0.6), example, size=19,
             font=font, color=INK, italic=(font == BODY))
        y += Inches(1.5)
    text(s, MARGIN, Inches(6.4), W - 2 * MARGIN, Inches(0.4),
         [[("Questions:  ", {"bold": True, "color": NAVY}),
           ("does the form matter?  ·  does a factual prompt help?  ·  which metrics can we trust?", {})]],
         size=16, color=INK)
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        "Much of India writes Hindi online in Roman script or mixes it with English, but factuality "
        "research is almost all English. We keep the facts fixed and change only the form: the original "
        "Devanagari, a rule-based transliteration that changes only the script, and an LLM rewrite into "
        "Hinglish that keeps every name, number and date. All three rows are the same sentence from a "
        "Bihar flood article: 'Bihar's government had submitted a memorandum of 7,200 crore in losses'. "
        "Our questions: does the form change faithfulness, does a factual prompt help, and which metrics work.")


def slide_method(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Method: one pipeline, no training")
    steps = [
        ("XL-Sum Hindi", "186 BBC news articles\n(random sample)"),
        ("3 parallel versions", "hi · hi_roman · hinglish\nfacts checked automatically"),
        ("Models, prompts", "gemma3:4b (local, 1,116)\ngpt-oss-120b (50 articles, 200)\nplain vs factual"),
        ("Evaluation", "number & entity P/R, NLI,\nROUGE, BERTScore\n+ 60 human-labelled summaries"),
    ]
    bw, gap = Inches(2.75), Inches(0.33)
    x, y, bh = MARGIN, Inches(1.85), Inches(2.4)
    for k, (head, body) in enumerate(steps):
        fill = NAVY if k in (0, 3) else TINT
        b = box(s, x, y, bw, bh, fill=fill)
        fg = WHITE if k in (0, 3) else INK
        text(s, x + Inches(0.2), y + Inches(0.25), bw - Inches(0.4), Inches(0.5), head, size=19, bold=True,
             color=ORANGE if k in (0, 3) else NAVY, font=HEAD)
        text(s, x + Inches(0.2), y + Inches(0.85), bw - Inches(0.4), bh - Inches(1.0), body.split("\n"),
             size=14, color=fg, spacing=3)
        if k < len(steps) - 1:
            arrow(s, x + bw + Inches(0.04), y + bh // 2, x + bw + gap - Inches(0.04), y + bh // 2)
        x += bw + gap
    stats = [("186", "articles × 3 conditions"), ("1,316", "summaries in total"),
             ("17.6", "mean Code-Mixing Index\nof the Hinglish rewrites"), ("5", "annotators, 60 summaries,\nblind to LLM suggestions")]
    x = MARGIN
    for big, label in stats:
        stat(s, x, Inches(4.75), Inches(2.9), big, label.split("\n"))
        x += Inches(3.05)
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        "We sampled 186 Hindi BBC articles from XL-Sum. Each was transliterated to Roman script by rules "
        "and rewritten into Hinglish by gpt-oss-120b; every pair is checked so that numbers and names "
        "are preserved. gemma3:4b summarizes everything on a laptop with a plain and a factual prompt, "
        "and gpt-oss-120b is a large-model reference on 50 articles. Metrics work across scripts: we parse "
        "numbers including Hindi number words, and match names through a phonetic skeleton. "
        "The Code-Mixing Index of 17.6 means real mixing, not occasional English words.")


def slide_result1(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Factual prompt raises recall; precision ≈ 1")
    picture(s, "fig_recall_by_condition", MARGIN, Inches(1.45), w=Inches(8.0), h=Inches(5.2))
    x = Inches(9.0)
    stat(s, x, Inches(1.7), Inches(3.8), "0.23 → 0.56", ["number recall, Hindi", "plain → factual prompt"],
         color=BLUE, size=40)
    stat(s, x, Inches(3.75), Inches(3.8), "≥ 0.97", ["number and entity precision", "in every condition"],
         color=NAVY)
    text(s, x, Inches(5.65), Inches(3.8), Inches(0.9),
         "Invented names and numbers are rare; the prompt mainly makes summaries keep more facts.",
         size=14, color=MUTED)
    footer(s, 4)
    s.notes_slide.notes_text_frame.text = (
        "Each pair of bars is one condition, plain in blue, factual in orange. The factual prompt raises "
        "number and entity recall in every condition, most for Devanagari Hindi, where number recall more "
        "than doubles. Precision, how many of the summary's numbers and names are in the article, is 0.97 "
        "or higher everywhere, so the model rarely invents names or numbers. Differences between the three "
        "conditions are small compared with the prompt effect.")


def slide_result2(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Small models first fail at language control")
    half = (W - 2 * MARGIN - Inches(0.4)) // 2
    picture(s, "fig_wrong_language", MARGIN, Inches(1.5), w=half, h=Inches(3.9))
    picture(s, "fig_model_recall", MARGIN + half + Inches(0.4), Inches(1.5), w=half, h=Inches(3.9))
    text(s, MARGIN, Inches(5.55), half, Inches(1.1),
         [[("9.1%", {"bold": True, "color": BLUE, "size": 22}),
           ("  of gemma3:4b's factual-prompt Hinglish summaries are in the wrong language, mostly English; "
            "gpt-oss-120b: 0 of 200", {})]], size=15)
    text(s, MARGIN + half + Inches(0.4), Inches(5.55), half, Inches(1.1),
         [[("Higher recall", {"bold": True, "color": BLUE, "size": 22}),
           ("  for the 120B model in all four cells, on the same 50 articles", {})]], size=15)
    footer(s, 5)
    s.notes_slide.notes_text_frame.text = (
        "Left: the share of summaries in the wrong language or script. The small model never fails on "
        "Devanagari, but on Hinglish it answers in English in up to 9 percent of cases, and the factual "
        "prompt, whose rules are in English, makes it worse. The large model never fails. Right: on the "
        "same 50 articles the large model keeps more numbers and names. So for small local models the "
        "first thing to measure is the wrong-language rate.")


def slide_result3(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Humans find errors; no metric beats chance")
    stat(s, MARGIN, Inches(1.65), Inches(4.2), "57%", ["of 60 labelled summaries contain an error",
                                                       "(20% a major one)"], color=VERMILION)
    stat(s, MARGIN, Inches(3.55), Inches(4.2), "κ = 0.38", ["Fleiss' kappa, faithful vs error, 5 annotators",
                                                            "(0.20 for 3 labels): fair agreement"], color=NAVY)
    text(s, MARGIN, Inches(5.45), Inches(4.2), Inches(1.2),
         "Most errors are wrong relations between correct names and numbers, which every metric misses.",
         size=14, color=MUTED)
    picture(s, "fig_agreement", Inches(5.2), Inches(1.6), w=Inches(7.55), h=Inches(4.6))
    text(s, Inches(5.2), Inches(6.25), Inches(7.55), Inches(0.5),
         "AUC for faithful vs error, 95% bootstrap CI; every interval includes 0.5 (chance)",
         size=13, color=MUTED, align=PP_ALIGN.CENTER)
    footer(s, 6)
    s.notes_slide.notes_text_frame.text = (
        "Five annotators labelled 60 gemma summaries as faithful, minor error or major error. 57 percent "
        "have at least one error. Agreement is fair, kappa 0.38 for faithful versus error, at the lower end "
        "of what FRANK and SummEval report; most disagreement is about severity. The right-hand plot asks "
        "whether any automatic metric separates faithful from wrong summaries: AUC 0.5 is chance. Both NLI "
        "variants are at or below chance, and no interval excludes 0.5, so we do not pick a primary NLI "
        "metric. Leaving out annotator A1, who agreed least, gives the same conclusion.")


def slide_examples(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Error examples: what the metrics catch and miss")
    rows = [
        ("Example", "Condition", "What went wrong", "Caught?"),
        ("Dreamliner 1995", "hi, plain", "Dates the Dreamliner to 1995; 1995 was the Boeing 777 launch",
         "Missed: number P 1.0, NLI 0.76 / 0.82"),
        ("4 lakh", "hi_roman, plain", "\"char lakh\" (4 lakh) children; source says 4 lakh 40 thousand",
         "Caught: number P 0.86"),
        ("James Anderson", "hi, plain", "Adds the first name \"James\"; source has only \"Anderson\"",
         "Caught: entity P 0.83"),
        ("barh → बारह", "hi_roman, factual", "\"barh\" (flood) back-transliterated as बारह (twelve): "
         "\"twelve victims\"", "Wrong script: excluded"),
        ("Fernandes", "hi_roman, factual", "Says he supported the police brutality he condemned",
         "Missed: NLI-window 0.66"),
        ("Abandoned match", "hinglish, plain", "Says Pakistan won a match that was abandoned for rain",
         "Missed: NLI-window 0.80"),
    ]
    widths = [Inches(2.2), Inches(2.0), Inches(4.9), Inches(3.033)]
    shape = s.shapes.add_table(len(rows), 4, MARGIN, Inches(1.45), sum(widths, Inches(0)), Inches(5.2))
    table = shape.table
    for j, wdt in enumerate(widths):
        table.columns[j].width = wdt
    for i, row in enumerate(rows):
        table.rows[i].height = Inches(0.62 if i == 0 else 0.75)
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (TINT if i % 2 else WHITE)
            p = cell.text_frame.paragraphs[0]
            parts = value.split("बारह")
            for k, part in enumerate(parts):  # Devanagari in its own font
                if k:
                    r = p.add_run()
                    r.text = "बारह"
                    r.font.name, r.font.size = DEVA, Pt(15)
                    r.font.color.rgb = INK
                if part:
                    r = p.add_run()
                    r.text = part
                    r.font.name, r.font.size = BODY, Pt(15 if i else 16)
                    r.font.bold = i == 0 or j == 0
                    color = WHITE if i == 0 else INK
                    if i and j == 3:
                        color = GREEN if value.startswith("Caught") else (VERMILION if value.startswith("Missed")
                                                                          else MUTED)
                    r.font.color.rgb = color
    footer(s, 7)
    s.notes_slide.notes_text_frame.text = (
        "Six cases. The script-independent checks catch the rare invented details: a rounded number and "
        "an added first name. But the most serious errors are relation errors: the Dreamliner gets the "
        "Boeing 777's launch year, Fernandes is said to support police brutality he condemned, and an "
        "abandoned match becomes a Pakistani win. Every name and number is copied correctly, so precision "
        "is 1 and NLI passes them. The flood case shows a script problem: Roman 'barh' was turned back "
        "into Devanagari as the word for twelve.")


def slide_conclusion(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, NAVY)
    text(s, MARGIN, Inches(0.45), W - 2 * MARGIN, Inches(0.8), "Conclusion", size=34, font=HEAD, bold=True,
         color=WHITE)
    col = (W - 2 * MARGIN - Inches(0.5)) // 2
    light = RGBColor(0xDC, 0xE3, 0xEE)
    text(s, MARGIN, Inches(1.5), col, Inches(0.5), "Takeaways", size=22, bold=True, color=ORANGE, font=HEAD)
    text(s, MARGIN, Inches(2.15), col, Inches(3.8), [
        [("1  ", {"bold": True, "color": ORANGE}), ("Measure the wrong-language rate first: a 4B model "
                                                    "answers Hinglish in English up to 9.1% of the time", {})],
        [("2  ", {"bold": True, "color": ORANGE}), ("A factual prompt raises recall (0.23 → 0.56); "
                                                    "invented names and numbers are rare", {})],
        [("3  ", {"bold": True, "color": ORANGE}), ("57% of summaries still have errors, mostly wrong "
                                                    "relations; no metric detects them", {})],
    ], size=17, color=WHITE, spacing=14)
    x2 = MARGIN + col + Inches(0.5)
    text(s, x2, Inches(1.5), col, Inches(0.5), "Limitations", size=22, bold=True, color=ORANGE, font=HEAD)
    text(s, x2, Inches(2.15), col, Inches(3.8), [
        [("1  ", {"bold": True, "color": ORANGE}), ("Hinglish articles are LLM rewrites, not natural "
                                                    "Hinglish news", {})],
        [("2  ", {"bold": True, "color": ORANGE}), ("60 human labels, small-model summaries only; "
                                                    "agreement is fair (κ = 0.38)", {})],
        [("3  ", {"bold": True, "color": ORANGE}), ("One small and one large model; "
                                                    "the large one on 50 articles", {})],
    ], size=17, color=light, spacing=14)
    text(s, MARGIN, Inches(6.05), W - 2 * MARGIN, Inches(0.5),
         [[("Code, data, summaries and labels:  ", {"color": light}), (GITHUB, {"bold": True, "color": WHITE})]],
         size=18)
    footer(s, 8, dark=True)
    s.notes_slide.notes_text_frame.text = (
        "Three takeaways: check the output language first, a factual prompt helps coverage, and the "
        "remaining errors are relation errors that need human evaluation. Limitations: synthetic Hinglish, "
        "a small human study, and two models. Everything is on GitHub. Thank you; we are happy to take "
        "questions.")


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # Update theme hyperlink color so links default to vibrant orange (E69F00) instead of dark blue
    for part in prs.part.package.iter_parts():
        if "theme" in str(part.partname):
            xml = part.blob.decode("utf-8")
            xml = xml.replace('<a:hlink><a:srgbClr val="0000FF"/></a:hlink>',
                              '<a:hlink><a:srgbClr val="E69F00"/></a:hlink>')
            part._blob = xml.encode("utf-8")

    for make in (slide_title, slide_problem, slide_method, slide_result1, slide_result2, slide_result3,
                 slide_examples, slide_conclusion):
        make(prs)
    prs.core_properties.title = "FactIndic"
    prs.core_properties.author = "Arman Sahni, Ananya Mishra, Gauri Mahajan, Krati Khanna, Arushi Chakraborty"
    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
