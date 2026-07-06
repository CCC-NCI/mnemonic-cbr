"""Export aggregated tables to CSV / Markdown / LaTeX.

Outputs are designed to be dropped into the IJAIED manuscript with
minimal hand editing. CSV is the source of truth; Markdown is for
quick inspection; LaTeX is for direct insertion into the .tex file.

Each export function takes a pandas DataFrame and writes one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _fmt_float(value, ndigits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{value:.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)


def export_csv(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_markdown(df: pd.DataFrame, path, ndigits: int = 3) -> Path:
    """Markdown table — pipe-separated, GitHub-compatible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = df.copy()
    for col in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[col]):
            fmt[col] = fmt[col].apply(lambda v: _fmt_float(v, ndigits))
    lines = ["| " + " | ".join(str(c) for c in fmt.columns) + " |"]
    lines.append("|" + "|".join("---" for _ in fmt.columns) + "|")
    for _, row in fmt.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_latex(
    df: pd.DataFrame,
    path,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    ndigits: int = 3,
    resize_to_textwidth: bool = False,
) -> Path:
    """LaTeX longtable-style output ready for direct insertion into the
    manuscript. Uses booktabs (toprule/midrule/bottomrule).

    Captions are LaTeX-escaped so that underscores in free-text strings
    (e.g. 'leg_a', 'pure_cbr_llm') do not trigger math-mode errors.

    If resize_to_textwidth=True, the tabular is wrapped in a resizebox
    to the page text-width — useful for wide tables that would otherwise
    overflow.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = df.copy()
    for col in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[col]):
            fmt[col] = fmt[col].apply(lambda v: _fmt_float(v, ndigits))
        else:
            fmt[col] = fmt[col].astype(str)

    n_cols = len(fmt.columns)
    col_spec = "l" + "r" * (n_cols - 1) if n_cols >= 2 else "l"
    parts = []
    parts.append("\\begin{table}[htbp]")
    parts.append("\\centering")
    if caption:
        parts.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        parts.append(f"\\label{{{label}}}")
    if resize_to_textwidth:
        parts.append("\\resizebox{\\textwidth}{!}{%")
    parts.append(f"\\begin{{tabular}}{{{col_spec}}}")
    parts.append("\\toprule")
    parts.append(" & ".join(_latex_escape(c) for c in fmt.columns) + " \\\\")
    parts.append("\\midrule")
    for _, row in fmt.iterrows():
        parts.append(" & ".join(_latex_escape(v) for v in row) + " \\\\")
    parts.append("\\bottomrule")
    parts.append("\\end{tabular}")
    if resize_to_textwidth:
        parts.append("}")
    parts.append("\\end{table}")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def _latex_escape(value) -> str:
    s = str(value)
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("$", "\\$")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
         .replace("^", "\\textasciicircum{}")
         .replace("~", "\\textasciitilde{}")
         .replace("×", "$\\times$")
         .replace("²", "$^2$")
         .replace("η", "$\\eta$")
         .replace("—", "---")
    )


# ---------------------------------------------------------------------
# Convenience: write all three formats to one base path
# ---------------------------------------------------------------------

def export_all(
    df: pd.DataFrame,
    out_dir,
    name: str,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    ndigits: int = 3,
) -> dict:
    """Write CSV + Markdown + LaTeX to {out_dir}/{name}.{csv,md,tex}."""
    out_dir = Path(out_dir)
    paths = {
        "csv":  export_csv(df, out_dir / f"{name}.csv"),
        "md":   export_markdown(df, out_dir / f"{name}.md", ndigits),
        "tex":  export_latex(df, out_dir / f"{name}.tex",
                             caption=caption, label=label, ndigits=ndigits),
    }
    return paths
