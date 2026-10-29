#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# WSL: prefer the Windows TeX Live binary so Windows fonts (SimSun/SimHei)
# are available to XeLaTeX. Fall back to xelatex.exe on PATH.
if [ -n "${XELATEX:-}" ]; then
  :
elif [ -x /mnt/c/Users/Banbo/AppData/Roaming/TinyTeX/bin/windows/xelatex.exe ]; then
  XELATEX=/mnt/c/Users/Banbo/AppData/Roaming/TinyTeX/bin/windows/xelatex.exe
elif command -v xelatex.exe >/dev/null 2>&1; then
  XELATEX=$(command -v xelatex.exe)
else
  echo "xelatex.exe not found; install TeX Live or set XELATEX." >&2
  exit 1
fi

"$XELATEX" -interaction=nonstopmode -halt-on-error manuscript_iop_zh.tex
"$XELATEX" -interaction=nonstopmode -halt-on-error manuscript_iop_zh.tex
