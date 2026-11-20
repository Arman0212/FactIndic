# FactIndic paper (IEEE conference format)

- `main.tex` - IEEEtran 2-column; sections are in `sections/*.tex`.
- `references.bib` - every entry with a DOI was checked against Crossref; the two
  without a DOI (Das & Gambäck 2014, Gambäck & Das 2016) against the ACL Anthology,
  and the Krippendorff (2011) report against its University of Pennsylvania page.
- `figures/` - made by `python -m src.analysis.make_figures` (PDF for LaTeX,
  300-dpi PNG for Word/slides); `cmi_correlations.csv` holds the Spearman numbers.
- `fonts/NotoSerifDevanagari-Regular.ttf` - Devanagari in the text (`\dev{...}`),
  SIL Open Font License (`fonts/OFL.txt`).
- `FactIndic_IEEE.pdf` - the final PDF (a copy of `main.pdf`).
- `FactIndic_IEEE.docx` - Word version made by `make_docx.py`.

Build the PDF (needs XeTeX for the Devanagari font; tectonic uses it):

```bash
brew install tectonic                     # once
cd paper && tectonic main.tex && cp main.pdf FactIndic_IEEE.pdf
```

Build the Word version (IEEE template layout: Letter, two 3.5" columns,
Times New Roman 10 pt, IEEE headings, captions and reference style):

```bash
brew install pandoc && pip install python-docx   # once
python paper/make_docx.py                        # from the repository root
```

Fix by hand in Word: run-in paragraph headings ("a) Labels:") are on their own
line; check figure and caption positions and page breaks; inline math uses
Cambria Math. To submit on the official IEEE template, paste the text into it
with "Keep Text Only" and re-apply the template's styles.
