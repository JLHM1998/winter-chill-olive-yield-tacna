# latexmkrc — auto-build for MDPI manuscript + supplementary
# Usage:  latexmk            (builds both)
#         latexmk -pvc       (watch mode: recompiles on every save)

$pdf_mode = 1;            # use pdflatex
$bibtex_use = 2;          # run bibtex when needed, delete .bbl on clean
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

@default_files = ('manuscript.tex', 'supplementary.tex');
