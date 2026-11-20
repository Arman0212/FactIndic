"""
FactIndic - A1's local labelling tool (blind-first protocol)
============================================================
A tiny local web app (Python standard library only) for A1's 20 rows of the
manual annotation: A1's own 10 rows + the 10 rows shared by all five
annotators (see src/eval/make_annotation_packs.py). Source on the left,
summary on the right, numbers and names highlighted. Saves to
results/annotation_packs/labels_A1.csv.

Blind-first protocol (enforced by the SERVER, not just hidden in the page):
  1. A1 chooses a label, an error type (required for errors) and a note
     (required for errors), then clicks "Lock & reveal draft".
  2. The server stores this decision in label / error_type / notes with a
     timestamp (locked_at). A locked decision can never be changed: a second
     lock request is refused.
  3. Only then does the server send the LLM draft for that row to the page.
     Before locking, the draft is not in anything the browser receives.
  4. After seeing the draft, A1 may record a REVISION in revised_label /
     revised_error_type / revised_notes. Revisions are kept for the paper
     (how often the draft changed A1's mind) but the merge uses the blind
     locked label, so A1's labels are as independent as the teammates'.

label_source (set at lock time): same_as_draft_blind / changed_from_draft_blind,
comparing the locked blind label (and error type) with the draft.

Teammates (A2-A5) use offline HTML pages that never contain drafts.
src/eval/merge_annotations.py combines the five label files;
src/eval/draft_agreement.py compares A1's blind labels with the drafts.

Run:   python -m src.eval.label_server
Open:  http://127.0.0.1:8765   (opens automatically; Ctrl+C to stop)
The server listens on 127.0.0.1 only, so nothing is exposed to the network.
"""

import json
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.eval.highlight import highlight, load_gold
from src.eval.make_annotation_packs import (
    ANNOTATION_PATH, KEY, OWNER, OWNER_COLUMNS, init_owner_labels, labels_path, load_assignment,
    read_csv, rows_for, write_csv,
)

HOST, PORT = "127.0.0.1", 8765
PAGE = Path(__file__).with_name("label_page.html")
LABELS_PATH = labels_path(OWNER)
LABELS = {"faithful", "minor_error", "major_error"}
ERROR_TYPES = {"entity", "number", "added_info", "meaning", "omission", "other"}
DRAFT_FIELDS = ["draft_label", "draft_error_type", "draft_note"]
LOCK = threading.Lock()


# ---------------------------------------------------------------- rules
def check_decision(label: str, error_type: str, notes: str) -> str:
    """Return an error message, or "" if the decision is complete.
    faithful: no error type. Errors: an error type AND a note are required."""
    if label not in LABELS:
        return "Choose a label (1, 2 or 3)."
    if label == "faithful":
        return "" if not error_type else "A faithful summary has no error type."
    if error_type not in ERROR_TYPES:
        return "Choose an error type for an error label."
    if not notes.strip():
        return "Write a note: quote the wrong claim and what the article says."
    return ""


def compare_with_draft(label: str, error_type: str, draft_label: str, draft_error_type: str) -> str:
    same = label == draft_label and error_type == draft_error_type
    return ("same_as_draft" if same else "changed_from_draft") + "_blind"


# ---------------------------------------------------------------- data
def build_rows():
    """A1's 20 rows and A1's label file, in the same order. Drafts are kept
    on the server (Handler.drafts) and only attached to locked rows."""
    ann = read_csv(ANNOTATION_PATH)
    assign = load_assignment(ann)
    init_owner_labels(ann, assign)
    mine = rows_for(OWNER, ann, assign)
    labels = read_csv(LABELS_PATH)
    for col in OWNER_COLUMNS:  # older files: add missing columns
        if col not in labels.columns:
            labels[col] = ""
    # put the label file in the page order, so row i of the page = row i of the file
    labels = mine[KEY].merge(labels[OWNER_COLUMNS], on=KEY, how="left").fillna("")
    labels["annotator"] = OWNER
    write_csv(labels[OWNER_COLUMNS], LABELS_PATH)

    gold = load_gold()
    rows, drafts = [], []
    for i, r in mine.iterrows():
        src_seg, sum_seg = highlight(r["source"], r["summary"], gold.get(r["id"], []))
        rows.append({"i": int(i), "id": r["id"], "condition": r["condition"],
                     "prompt_type": r["prompt_type"], "source_seg": src_seg, "summary_seg": sum_seg,
                     **{k: labels.at[i, k] for k in OWNER_COLUMNS[4:]}})
        drafts.append({k: r[k] for k in DRAFT_FIELDS})
    return rows, drafts


def public_row(i: int) -> dict:
    """What the browser may see: the draft only once the row is locked."""
    row = dict(Handler.rows[i])
    if row["locked_at"]:
        row.update(Handler.drafts[i])
    return row


# ---------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    rows, drafts = [], []

    def send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/rows":
            self.send_json({"rows": [public_row(i) for i in range(len(Handler.rows))]})
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/lock", "/api/revise"):
            return self.send(404, b"not found", "text/plain")
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        i = int(data["i"])
        label = str(data.get("label", "")).strip()
        etype = str(data.get("error_type", "")).strip()
        notes = str(data.get("notes", "")).strip()
        problem = check_decision(label, etype, notes)
        if problem:
            return self.send_json({"ok": False, "error": problem}, 400)

        with LOCK:
            df = read_csv(LABELS_PATH)
            row = Handler.rows[i]
            assert df.at[i, "id"] == row["id"], "label file and page are out of sync - restart the server"
            if self.path == "/api/lock":
                if df.at[i, "locked_at"]:
                    return self.send_json({"ok": False, "error": "Already locked - the blind label is permanent."}, 409)
                draft = Handler.drafts[i]
                updates = {"label": label, "error_type": etype, "notes": notes,
                           "locked_at": datetime.now().isoformat(timespec="seconds"),
                           "label_source": compare_with_draft(label, etype, draft["draft_label"],
                                                              draft["draft_error_type"])}
            else:  # /api/revise - only after locking; never touches the blind label
                if not df.at[i, "locked_at"]:
                    return self.send_json({"ok": False, "error": "Lock your blind label first."}, 409)
                same = label == df.at[i, "label"] and etype == df.at[i, "error_type"]
                updates = {"revised_label": "" if same else label,
                           "revised_error_type": "" if same else etype,
                           "revised_notes": "" if same else notes}
            for k, v in updates.items():
                df.at[i, k] = v
                row[k] = v
            write_csv(df, LABELS_PATH)
            n_locked = int((df["locked_at"] != "").sum())
        self.send_json({"ok": True, "row": public_row(i), "n_locked": n_locked})

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main():
    print("Preparing highlights (first run loads the matcher, a few seconds)...")
    Handler.rows, Handler.drafts = build_rows()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    n_locked = sum(1 for r in Handler.rows if r["locked_at"])
    print(f"Labelling {len(Handler.rows)} rows ({n_locked} locked) at {url} - saving to {LABELS_PATH}. Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
