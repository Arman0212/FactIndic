# FactIndic paper (IEEE conference format)

- `main.tex` - IEEEtran 2-column; sections are in `sections/*.tex`.
- `references.bib` - every entry with a DOI was checked against Crossref; the two
  without a DOI (Das & Gambäck 2014, Gambäck & Das 2016) against the ACL Anthology.
- `figures/` - made by `python -m src.analysis.make_figures` (PDF for LaTeX,
  300-dpi PNG for Word/slides); `cmi_correlations.csv` holds the Spearman numbers.
- Unfinished parts are marked in red with `\todo{...}`: search for "TODO".

Build the PDF:

```bash
brew install tectonic          # once; or use Overleaf / latexmk -pdf
cd paper && tectonic main.tex  # -> main.pdf
```

Convert to Word (later):

```bash
brew install pandoc
cd paper && pandoc main.tex --citeproc --bibliography=references.bib -o FactIndic.docx
```
Pandoc does not keep the IEEE two-column layout; use the IEEE Word template and paste in.
