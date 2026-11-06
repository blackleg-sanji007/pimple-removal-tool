# Project write-up

This folder contains the completed report based on the supplied CVPR 2026
`main.tex` template. The official 2026 `cvpr.sty`, `preamble.tex`, and
`ieeenat_fullname.bst` files are included, along with the report's figures,
bibliography, full application and training source code, and a compiled PDF.

To rebuild with Tectonic, run `tectonic main.tex` from this folder. With a
standard TeX installation, run `pdflatex main.tex`, `bibtex main`, and
`pdflatex main.tex` twice. The report uses the CVPR layout as a project
write-up; it is not a CVPR submission.

The training curve comes from `evidence/training_results.csv`. The held-out test
metrics and the exact evaluation settings are in `evidence/test_metrics.json`.
The `code/` copies are frozen versions of the two Python files and
`requirements.txt` used by this project. The trained binary checkpoint
remains in the main project's `models/` folder.
