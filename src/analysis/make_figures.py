"""
FactIndic - Phase 5: figures for the paper
==========================================
Saves every figure to paper/figures/ as PDF (vector, for LaTeX) and PNG
(300 dpi, for Word / slides). Sized for ONE column of a 2-column IEEE paper
(3.5 in wide, 8 pt text), colours from the Okabe-Ito colour-blind-safe
palette, plus hatching so the bars also work in black and white.

  fig_recall_by_condition   (a) number and entity recall, condition x prompt,
                                gemma3:4b, all articles, 95% bootstrap CI
  fig_wrong_language        (b) wrong-language rate, condition x prompt,
                                gemma3:4b (186 articles) vs gpt-oss-120b (50),
                                95% Wilson CI
  fig_model_recall          (c) gemma3:4b vs gpt-oss-120b recall on the same
                                50-article subset (rows where both models
                                passed the language check), 95% bootstrap CI
  fig_cmi_recall            (d) per-article recall vs CMI, hinglish condition,
                                scatter + least-squares line + Spearman rho
  fig_agreement             placeholder: metric vs human labels (needs the
                                merged annotation; TODO)

Also writes paper/figures/cmi_correlations.csv: Spearman rho for (d), plus the
same correlation for the hi condition as a CONTROL. CMI describes the
Hinglish rewrite, so if recall on the Devanagari version of the same article
correlates with it just as much, the effect is about the article, not about
code-mixing.

Run:  python -m src.analysis.make_figures
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.fix_flagged import FINAL_PATH
from src.data.make_hinglish import read_jsonl
from src.evaluate import SEED, bootstrap_ci, scores_path, table_path
from src.summarize import output_path

OUT_DIR = Path("paper/figures")
GEMMA, GPT = "gemma3:4b", "openai/gpt-oss-120b"
MODEL_NAMES = {GEMMA: "gemma3:4b", GPT: "gpt-oss-120b"}
CONDITIONS = ["hi", "hi_roman", "hinglish"]
COND_NAMES = {"hi": "Hindi", "hi_roman": "Romanized\nHindi", "hinglish": "Hinglish"}
PROMPTS = ["plain", "factual"]
KEY = ["id", "condition", "prompt_type"]

# Okabe-Ito palette (colour-blind safe)
COLORS = {"plain": "#0072B2", "factual": "#E69F00", GEMMA: "#009E73", GPT: "#CC79A7"}
HATCH = {"plain": "", "factual": "///", GEMMA: "", GPT: "..."}

COL_WIDTH = 3.5  # inches, one IEEE column
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "hatch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in the PDF
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {OUT_DIR / name}.pdf/.png")


def bars(ax, groups: list, series: list, values: dict, colors: dict, hatches: dict, labels: dict):
    """Grouped bars with error bars. values[(group, series)] = (mean, lo, hi);
    a missing key leaves a gap marked 'n/a'."""
    width = 0.8 / len(series)
    for k, s in enumerate(series):
        x = np.arange(len(groups)) + (k - (len(series) - 1) / 2) * width
        for xi, g in zip(x, groups):
            if (g, s) not in values:
                ax.text(xi, 0.01, "n/a", ha="center", va="bottom", fontsize=6, color="0.4", rotation=90)
                continue
            m, lo, hi = values[(g, s)]
            if m == 0:  # a zero bar is invisible: mark it with a short line
                ax.plot([xi - width * 0.4, xi + width * 0.4], [0, 0], color=colors[s], lw=2.5,
                        solid_capstyle="butt", zorder=3)
            ax.bar(xi, m, width * 0.92, color=colors[s], hatch=hatches[s], edgecolor="black",
                   linewidth=0.4, label=labels[s] if g == groups[0] or (groups[0], s) not in values else None)
            ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], color="black", lw=0.6, capsize=1.5)
    ax.set_xticks(np.arange(len(groups)))


def dedupe_legend(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), frameon=False, **kw)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Rate and 95% Wilson score interval for k successes out of n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- (a)
def fig_recall_by_condition():
    table = pd.read_csv(table_path(GEMMA))
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH, 1.9), sharey=True)
    for ax, metric, title in zip(axes, ["num_r", "ent_r"], ["Number recall", "Entity recall"]):
        vals = {(r.condition, r.prompt_type): (r[metric], r[f"{metric}_lo"], r[f"{metric}_hi"])
                for _, r in table.iterrows()}
        bars(ax, CONDITIONS, PROMPTS, vals, COLORS, HATCH, {p: p for p in PROMPTS})
        ax.set_xticklabels([COND_NAMES[c] for c in CONDITIONS])
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Recall (mean, 95% CI)")
    dedupe_legend(axes[0], loc="upper left", title="Prompt", title_fontsize=7)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_recall_by_condition")


# ---------------------------------------------------------------- (b)
def fig_wrong_language():
    vals, counts = {}, {}
    for model in (GEMMA, GPT):
        s = pd.DataFrame(read_jsonl(output_path(model)))
        for (cond, ptype), g in s.groupby(["condition", "prompt_type"]):
            k, n = int((~g["script_ok"]).sum()), len(g)
            p, lo, hi = wilson(k, n)
            vals[(f"{cond}|{ptype}", model)] = (100 * p, 100 * lo, 100 * hi)
            counts[(f"{cond}|{ptype}", model)] = (k, n)
    groups = [f"{c}|{p}" for c in CONDITIONS for p in PROMPTS]
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.9))
    bars(ax, groups, [GEMMA, GPT], vals, COLORS, HATCH,
         {GEMMA: "gemma3:4b (186 articles)", GPT: "gpt-oss-120b (50 articles)"})
    for k, m in enumerate([GEMMA, GPT]):  # raw counts above the bars
        for i, g in enumerate(groups):
            if (g, m) in counts:
                x = i + (k - 0.5) * 0.4
                ax.text(x, vals[(g, m)][2] + 0.6, f"{counts[(g, m)][0]}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticklabels([g.split("|")[1] for g in groups], fontsize=6.5)
    for i, cond in enumerate(CONDITIONS):  # condition name under each pair of groups
        ax.text(2 * i + 0.5, -0.17, COND_NAMES[cond].replace("\n", " "), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7)
    ax.set_ylabel("Wrong language/script (%)")
    ax.set_ylim(0, 20)
    dedupe_legend(ax, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_wrong_language")


# ---------------------------------------------------------------- (c)
def fig_model_recall():
    gem, gpt = pd.read_csv(scores_path(GEMMA)), pd.read_csv(scores_path(GPT))
    # score files contain only summaries that passed the script check, so the
    # inner join = rows where BOTH models answered in the right language
    common = gpt[KEY].merge(gem[KEY], on=KEY)
    rng = np.random.default_rng(SEED)
    groups = [f"{c}|{p}" for c in ["hi", "hinglish"] for p in PROMPTS]
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH, 1.9), sharey=True)
    for ax, metric, title in zip(axes, ["num_r", "ent_r"], ["Number recall", "Entity recall"]):
        vals = {}
        for model, sc in ((GEMMA, gem), (GPT, gpt)):
            sub = sc.merge(common, on=KEY)
            for g in groups:
                cond, ptype = g.split("|")
                v = sub.loc[(sub["condition"] == cond) & (sub["prompt_type"] == ptype), metric].to_numpy(float)
                vals[(g, model)] = bootstrap_ci(v, rng)
        bars(ax, groups, [GEMMA, GPT], vals, COLORS, HATCH, MODEL_NAMES)
        ax.set_xticklabels([f"{'Hindi' if g.startswith('hi|') else 'Hinglish'}\n{g.split('|')[1]}" for g in groups],
                           fontsize=6)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Recall (mean, 95% CI)")
    dedupe_legend(axes[0], loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_model_recall")
    print(f"    (c) uses {len(common)} rows where both models passed the language check")


# ---------------------------------------------------------------- (d)
def fig_cmi_recall():
    cmi = pd.DataFrame(read_jsonl(FINAL_PATH))[["id", "cmi"]].astype({"cmi": float})
    sc = pd.read_csv(scores_path(GEMMA)).merge(cmi, on="id")
    stats = []
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH, 2.1), sharey=True)
    for ax, metric, title in zip(axes, ["num_r", "ent_r"], ["Number recall", "Entity recall"]):
        for ptype, marker in zip(PROMPTS, ["o", "^"]):
            d = sc[(sc["condition"] == "hinglish") & (sc["prompt_type"] == ptype)][["cmi", metric]].dropna()
            rho, p = spearmanr(d["cmi"], d[metric])
            ax.scatter(d["cmi"], d[metric], s=6, marker=marker, color=COLORS[ptype], alpha=0.55,
                       linewidths=0, label=f"{ptype}: $\\rho$={rho:.2f}, p={p:.2f}")
            slope, icpt = np.polyfit(d["cmi"], d[metric], 1)
            xs = np.linspace(d["cmi"].min(), d["cmi"].max(), 50)
            ax.plot(xs, slope * xs + icpt, color=COLORS[ptype], lw=1.0, ls="-" if ptype == "plain" else "--")
            stats.append({"condition": "hinglish", "prompt_type": ptype, "metric": metric,
                          "n": len(d), "spearman_rho": rho, "p": p})
            # control: recall on the Devanagari version of the same articles
            c = sc[(sc["condition"] == "hi") & (sc["prompt_type"] == ptype)][["cmi", metric]].dropna()
            rho_c, p_c = spearmanr(c["cmi"], c[metric])
            stats.append({"condition": "hi (control)", "prompt_type": ptype, "metric": metric,
                          "n": len(c), "spearman_rho": rho_c, "p": p_c})
        ax.set_title(title)
        ax.set_xlabel("Code-Mixing Index (CMI)")
        ax.set_ylim(-0.03, 1.38)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(frameon=False, loc="upper left", fontsize=5.5, handletextpad=0.2, borderaxespad=0.1)
    axes[0].set_ylabel("Recall (per summary)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_cmi_recall")
    stats = pd.DataFrame(stats).round(4)
    stats.to_csv(OUT_DIR / "cmi_correlations.csv", index=False)
    print(stats.to_string(index=False))


# ---------------------------------------------------------------- placeholder
def fig_agreement():
    """TODO (after src.eval.merge_annotations): one bar per metric with the AUC
    (faithful vs error) and its bootstrap CI, from src.eval.agreement. Not drawn
    yet: it needs the human labels. The paper shows a framed placeholder."""
    print("  fig_agreement: skipped (needs merged human labels - TODO)")


def main():
    print("Writing figures:")
    fig_recall_by_condition()
    fig_wrong_language()
    fig_model_recall()
    fig_cmi_recall()
    fig_agreement()


if __name__ == "__main__":
    main()
