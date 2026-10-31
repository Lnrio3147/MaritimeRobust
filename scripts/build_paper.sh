#!/usr/bin/env bash
# Rebuild the paper: Markdown -> LaTeX body -> PDF.
#
# The draft in paper/DRAFT.md is the single source; body.tex is generated and
# should not be edited. Tectonic caches its bundle under ~/.cache/Tectonic, so
# the first build is slow and later ones are ~20 s.
set -eu
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/md_to_latex.py
cd paper/latex
tectonic -X compile main.tex 2>&1 | grep -vE "^note: downloading" | tail -12
echo "wrote paper/latex/main.pdf"
