"""
FactIndic - split the manual annotation across 5 annotators (A1-A5)
===================================================================
Creates results/annotation_packs/:
  assignment.csv       which annotator(s) label which of the 60 rows
  labels_A1.csv        A1's own labels (written by src/eval/label_server.py,
                       blind-first protocol)
  A2.html .. A5.html   one self-contained offline labelling page per teammate

The split (60 rows = 6 cells "condition x prompt", 10 rows per cell):
  - 10 SHARED rows, labelled by all five -> inter-annotator agreement (kappa);
  - the other 50 rows, 10 per annotator  -> more labelled data.
  Stratified: inside each cell the rows are shuffled (seed 42), then all 60 are
  dealt in turn to [shared, A1, A2, A3, A4, A5]. Ten consecutive rows of a cell
  cover all six buckets, so every bucket gets 1-2 rows from EVERY cell.
  Rows A1 had already labelled are swapped into A1's own bucket, so those
  labels were kept at the time of the split. (Those early labels, made with
  the draft visible, were later moved to labels_A1_assisted.csv; A1 relabels
  all 20 rows blind.)

Each annotator sees 20 rows (shared + own), in a per-annotator shuffled order,
and does not know which rows are the shared ones.

The teammate pages contain ONLY id, condition, prompt, source, summary and the
highlights: no draft labels (the script checks that no draft text ends up in a
file). Labels autosave in the browser (localStorage) and are exported with the
"Download my labels" button as labels_A2.csv etc.

assignment.csv is created once and then re-used, so rebuilding the pages
(e.g. after a template fix) never changes who labels what.

Run:  python -m src.eval.make_annotation_packs
"""

import html
import json
import random
import re
from pathlib import Path

import pandas as pd

from src.eval.highlight import highlight, load_gold
from src.eval.make_annotation_set import OUT_PATH as ANNOTATION_PATH

PACK_DIR = Path("results/annotation_packs")
ASSIGNMENT_PATH = PACK_DIR / "assignment.csv"
GUIDELINES_PATH = PACK_DIR / "GUIDELINES.md"
TEMPLATE_PATH = Path(__file__).with_name("pack_template.html")
ANNOTATORS = ["A1", "A2", "A3", "A4", "A5"]
OWNER = "A1"  # labels locally with label_server.py (with drafts, blind mode)
LABEL_COLUMNS = ["id", "condition", "prompt_type", "annotator", "label", "error_type", "notes"]
KEY = ["id", "condition", "prompt_type"]  # identifies a row (an article can appear twice)
SEED = 42


# ---------------------------------------------------------------- files
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def write_csv(df: pd.DataFrame, path: Path):
    """Write to a temporary file, then replace: the CSV is never half-written."""
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def labels_path(annotator: str) -> Path:
    return PACK_DIR / f"labels_{annotator}.csv"


# ---------------------------------------------------------------- assignment
def make_assignment(ann: pd.DataFrame) -> pd.DataFrame:
    buckets = ["shared"] + ANNOTATORS
    rng = random.Random(SEED)
    order = []
    for _, cell in ann.groupby(["condition", "prompt_type"], sort=True):
        idx = list(cell.index)
        rng.shuffle(idx)
        order += idx
    bucket = {row: buckets[k % len(buckets)] for k, row in enumerate(order)}

    # keep A1's existing labels: move labelled rows into A1's bucket
    # (swap with an unlabelled A1 row, from the same cell if possible)
    labelled = [i for i in ann.index if ann.at[i, "label"]]
    for i in labelled:
        if bucket[i] in ("shared", OWNER):
            continue
        cell = lambda j: (ann.at[j, "condition"], ann.at[j, "prompt_type"])
        swaps = [j for j in ann.index if bucket[j] == OWNER and j not in labelled]
        swaps.sort(key=lambda j: cell(j) != cell(i))  # same cell first
        j = swaps[0]
        bucket[i], bucket[j] = bucket[j], bucket[i]

    out = ann[KEY].copy()
    out.insert(0, "row", ann.index + 1)  # 1-based row number in manual_annotation.csv
    out["set"] = ["shared" if bucket[i] == "shared" else "single" for i in ann.index]
    out["annotators"] = [";".join(ANNOTATORS) if bucket[i] == "shared" else bucket[i] for i in ann.index]
    return out


def load_assignment(ann: pd.DataFrame) -> pd.DataFrame:
    if ASSIGNMENT_PATH.exists():
        return read_csv(ASSIGNMENT_PATH)
    assign = make_assignment(ann)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(assign, ASSIGNMENT_PATH)
    print(f"Wrote {ASSIGNMENT_PATH}")
    return read_csv(ASSIGNMENT_PATH)


def rows_for(annotator: str, ann: pd.DataFrame, assign: pd.DataFrame) -> pd.DataFrame:
    """The annotator's 20 rows of manual_annotation.csv (shared + own), in
    their own shuffled order."""
    mine = assign[assign["annotators"].str.split(";").apply(lambda a: annotator in a)]
    rows = ann.merge(mine[KEY + ["set"]], on=KEY, how="inner")
    order = list(rows.index)
    random.Random(f"{SEED}-{annotator}").shuffle(order)
    return rows.loc[order].reset_index(drop=True)


# A1's label file has extra columns for the blind-first protocol (see label_server.py)
OWNER_COLUMNS = LABEL_COLUMNS + ["label_source", "locked_at",
                                 "revised_label", "revised_error_type", "revised_notes"]


def init_owner_labels(ann: pd.DataFrame, assign: pd.DataFrame):
    """Create an EMPTY labels_A1.csv (A1's 20 rows, nothing labelled) if it
    does not exist. Labels are only ever written by label_server.py, when A1
    locks a blind decision."""
    path = labels_path(OWNER)
    if path.exists():
        return
    rows = rows_for(OWNER, ann, assign)[KEY].copy()
    rows["annotator"] = OWNER
    for col in OWNER_COLUMNS:
        if col not in rows.columns:
            rows[col] = ""
    write_csv(rows[OWNER_COLUMNS], path)
    print(f"Wrote {path} (empty, {len(rows)} rows)")


# ---------------------------------------------------------------- pages
def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def markdown_to_html(md: str) -> str:
    """Tiny converter for GUIDELINES.md: headings, '- ' bullets, '> ' quotes,
    paragraphs, **bold** and `code`. Enough to embed the guidelines."""
    out, in_list = [], False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline_md(s[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if s.startswith("#"):
            level = min(len(s) - len(s.lstrip("#")) + 2, 6)  # '#' -> h3
            out.append(f"<h{level}>{inline_md(s.lstrip('#').strip())}</h{level}>")
        elif s.startswith(">"):
            out.append(f"<blockquote>{inline_md(s.lstrip('>').strip())}</blockquote>")
        elif s:
            out.append(f"<p>{inline_md(s)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def build_page(annotator: str, ann: pd.DataFrame, assign: pd.DataFrame, gold: dict) -> str:
    rows = []
    for _, r in rows_for(annotator, ann, assign).iterrows():
        src_seg, sum_seg = highlight(r["source"], r["summary"], gold.get(r["id"], []))
        # ONLY these fields go into the page - no drafts, no "shared" flag
        rows.append({"id": r["id"], "condition": r["condition"], "prompt_type": r["prompt_type"],
                     "source_seg": src_seg, "summary_seg": sum_seg})
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")  # safe inside <script>
    guidelines = markdown_to_html(GUIDELINES_PATH.read_text(encoding="utf-8"))
    page = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (page.replace("__GUIDELINES_HTML__", guidelines)
                .replace("__ANNOTATOR__", annotator)
                .replace("__ROWS_JSON__", data))


def check_no_drafts(page: str, ann: pd.DataFrame):
    """Refuse to write a page that contains any draft text."""
    assert "draft" not in page.lower(), "the word 'draft' appears in a teammate page"
    for note in ann["draft_note"]:
        assert not note or note not in page, "a draft note leaked into a teammate page"


def main():
    ann = read_csv(ANNOTATION_PATH)
    assign = load_assignment(ann)
    init_owner_labels(ann, assign)

    print(f"\n{'annotator':<10} {'rows':>4}  cells (condition/prompt: n)")
    for a in ["shared"] + ANNOTATORS:
        sel = assign[assign["annotators"] == a] if a != "shared" else assign[assign["set"] == "shared"]
        cells = sel.groupby(["condition", "prompt_type"]).size()
        print(f"{a:<10} {len(sel):>4}  " + ", ".join(f"{c}/{p}: {n}" for (c, p), n in cells.items()))

    gold = load_gold()
    for a in ANNOTATORS:
        if a == OWNER:
            continue
        page = build_page(a, ann, assign, gold)
        check_no_drafts(page, ann)
        (PACK_DIR / f"{a}.html").write_text(page, encoding="utf-8")
        print(f"Wrote {PACK_DIR / f'{a}.html'} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
