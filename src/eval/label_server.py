"""
FactIndic - A1's local labelling tool
=====================================
A tiny local web app (Python standard library only) for A1's 20 rows of the
manual annotation: A1's own 10 rows + the 10 rows shared by all five
annotators (see src/eval/make_annotation_packs.py). One row at a time: source
on the left, summary on the right, numbers and names highlighted, the draft
label shown as a suggestion, every change saved to
results/annotation_packs/labels_A1.csv.

Teammates (A2-A5) use offline HTML pages instead, which never contain drafts.
src/eval/merge_annotations.py later combines all five label files into
results/manual_annotation.csv.

Run:   python -m src.eval.label_server
Open:  http://127.0.0.1:8765   (opens automatically; Ctrl+C to stop)

Blind mode is ON by default: the draft stays hidden until you have chosen
your own label. The column label_source records how each label relates to
the draft:
  same_as_draft / changed_from_draft            - draft was visible
  same_as_draft_blind / changed_from_draft_blind - draft was hidden (blind mode)
Report these counts in the paper: labels copied from an LLM draft are not
independent human judgments.

The server listens on 127.0.0.1 only, so nothing is exposed to the network.
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.eval.highlight import highlight, load_gold
from src.eval.make_annotation_packs import (
    ANNOTATION_PATH, KEY, OWNER, labels_path, load_assignment, read_csv, rows_for, write_csv,
)

HOST, PORT = "127.0.0.1", 8765
PAGE = Path(__file__).with_name("label_page.html")
LABELS_PATH = labels_path(OWNER)
EDITABLE = ["label", "error_type", "notes"]
LOCK = threading.Lock()


# ---------------------------------------------------------------- data
def build_rows():
    """A1's 20 rows (with drafts) and A1's label file, in the same order."""
    ann = read_csv(ANNOTATION_PATH)
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} not found - run: python -m src.eval.make_annotation_packs")
    mine = rows_for(OWNER, ann, load_assignment(ann))
    labels = read_csv(LABELS_PATH)
    # put the label file in the page order, so row i of the page = row i of the file
    labels = mine[KEY].merge(labels, on=KEY, how="left").fillna("")
    write_csv(labels, LABELS_PATH)

    gold = load_gold()
    rows = []
    for i, r in mine.iterrows():
        src_seg, sum_seg = highlight(r["source"], r["summary"], gold.get(r["id"], []))
        rows.append({
            "i": int(i), "id": r["id"], "condition": r["condition"], "prompt_type": r["prompt_type"],
            "source_seg": src_seg, "summary_seg": sum_seg,
            "draft_label": r["draft_label"], "draft_error_type": r["draft_error_type"],
            "draft_note": r["draft_note"],
            **{k: labels.at[i, k] for k in EDITABLE + ["label_source"]},
        })
    return rows


# ---------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    rows = []

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
            self.send_json({"rows": Handler.rows})
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            return self.send(404, b"not found", "text/plain")
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        i = int(data["i"])
        with LOCK:
            df = read_csv(LABELS_PATH)
            row = Handler.rows[i]
            assert df.at[i, "id"] == row["id"], "label file and page are out of sync - restart the server"
            for k in EDITABLE:
                if k in data:
                    df.at[i, k] = str(data[k])
            # label_source changes only when the label or error type changes
            if "label" in data or "error_type" in data:
                if not df.at[i, "label"]:
                    df.at[i, "label_source"] = ""
                else:
                    same = (df.at[i, "label"] == row["draft_label"]
                            and df.at[i, "error_type"] == row["draft_error_type"])
                    src = "same_as_draft" if same else "changed_from_draft"
                    df.at[i, "label_source"] = src if data.get("seen_draft", True) else src + "_blind"
            write_csv(df, LABELS_PATH)
            for k in EDITABLE + ["label_source"]:
                row[k] = df.at[i, k]
            n_done = int((df["label"] != "").sum())
        self.send_json({"ok": True, "label_source": row["label_source"], "n_done": n_done})

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main():
    print("Preparing highlights (first run loads the matcher, a few seconds)...")
    Handler.rows = build_rows()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Labelling {len(Handler.rows)} rows at {url}  - saving to {LABELS_PATH}. Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
