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
with CJK characters allowed only in the author list. `manuscript_iop_zh.tex` is
the Chinese review draft using the IOP layout, not an IOP-ready submission.

## English submission version

`manuscript_iop_en.tex` is the English manuscript for submission. It follows
the supervisor's English draft (`../导师英文稿.pdf`) and adds the todo1102
results (sealed hold-out, leave-cluster-out audit, noise-augmented training,
random-kernel noise tests, 36-D feature screen, ONNX/int8 export; see
`../../todo1102-report.md`). It needs the TeX packages `multirow` and `cite`
(`tlmgr install multirow cite` on TinyTeX).

Its figures are vector PDFs in `figures_en/` (with 300-dpi PNG previews; copies
in `../figures/en/`). They follow the wording and layout of the supervisor's
draft and are generated from the audited result files by:

```powershell
$env:PYTHONPATH = "src"; python scripts/generate_journal_figures_en.py
```

(run from the project root; needs `matplotlib` and `numpy`, and the Arial and
Times New Roman fonts). The manuscript references them by explicit path,
because LaTeX would otherwise pick up the Chinese figures with the same names
in this folder.

On Windows, build directly from this folder:

```powershell
xelatex -interaction=nonstopmode -halt-on-error manuscript_iop_en.tex
xelatex -interaction=nonstopmode -halt-on-error manuscript_iop_en.tex
```
