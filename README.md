# Cover-evo

LLM-guided heuristic evolution for the minimum weighted dominating set problem.

[Paper](report/report.pdf) | [LaTeX source](report/report.tex) | [Author](authors.md)

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Usage

```bash
# Evaluate frozen heuristics on one test instance.
python scripts/evaluate.py

# Verify saved solutions and benchmark inputs.
python scripts/evaluate.py --verify-saved

# Rebuild summary tables and the report figure from saved observations.
python scripts/reproduce_results.py
```

## Structure

- `src/coverevo/`, `scripts/`: algorithms, evaluation and analysis.
- `configs/`, `prompts/`: experiment settings and LLM prompts.
- `data/`, `results/`: benchmark instances, frozen programs and observations.
- `report/`: paper, bibliography and figure.

## Build the paper

Requires XeLaTeX and BibTeX.

```bash
cd report
xelatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```
