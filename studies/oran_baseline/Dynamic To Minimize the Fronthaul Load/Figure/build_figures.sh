#!/usr/bin/env bash
# Render Figure/fig_*.pdf from the pgfplots sources Figure/fig_*.tex and Figure/data/*.dat.
#
# main.tex only includes the PDFs, so it compiles without pgfplots (e.g. on the local Overleaf).
# Run this script after changing a figure source or its data; it needs TeX Live with pgfplots
# and biber. The figures are exported from a full build of the paper (TikZ external library),
# so their size, fonts, \ref and \cite numbers match the paper.
#
# Usage: bash Figure/build_figures.sh
set -euo pipefail

cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -r main.tex abbr.tex ./*.bib IEEEtran.cls Figure "$tmp"/
cd "$tmp"
rm -f Figure/fig_*.pdf

# Stage 1: draw the figures from their sources and resolve references and citations.
python3 - <<'EOF'
import re
s = open('main.tex').read()
pre = (r'\usepackage{pgfplots}\pgfplotsset{compat=1.18}\usepgfplotslibrary{groupplots}'
       r'\usetikzlibrary{external}' '\n')
s = s.replace(r'\begin{document}', pre + r'\begin{document}', 1)
s, n = re.subn(r'\\includegraphics\{Figure/(fig_\w+)\.pdf\}',
               r'\\tikzsetnextfilename{\1}\\input{Figure/\1.tex}', s)
assert n > 0, 'no \\includegraphics{Figure/fig_*.pdf} found in main.tex'
open('main.tex', 'w').write(s)
EOF
latex_run() { pdflatex "$@" -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { tail -40 main.log; exit 1; }; }
latex_run
biber main > /dev/null
latex_run
latex_run

# Stage 2: export every figure to Figure/<name>.pdf.
sed -i 's|\\usetikzlibrary{external}|\\usetikzlibrary{external}\\tikzexternalize[prefix=Figure/]|' main.tex
latex_run -shell-escape

cd - > /dev/null
cp "$tmp"/Figure/fig_*.pdf Figure/
ls Figure/fig_*.pdf
