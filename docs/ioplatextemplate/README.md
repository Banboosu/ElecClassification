# IOP LaTeX draft

Build from WSL with the Windows XeLaTeX binary:

```bash
./build.sh
```

The script uses `xelatex.exe` because the Chinese fonts are Windows fonts
(`SimSun`, `SimHei`, `Microsoft YaHei`). It runs two passes so references settle.

`fig2_overall_framework.pdf` is generated from the project root by:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from scripts.generate_journal_figures import plot_fig1_overall_framework
plot_fig1_overall_framework()
PY
```

The figure keeps only flow labels; details are in the caption below it.

Important: IOP journals require the body text and references to be in English,
with CJK characters allowed only in the author list. This file is the Chinese
review draft using the IOP layout, not yet an IOP-ready submission.
