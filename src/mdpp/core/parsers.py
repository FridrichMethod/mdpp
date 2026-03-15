"""Thin wrappers around external parsers for MD engine output files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mdpp._types import StrPath

_LEGEND_PATTERN = re.compile(r's(\d+)\s+legend\s+"(.+)"', re.IGNORECASE)


@dataclass
class _XVGMetadata:
    """Parsed metadata from an XVG file header."""

    legends: dict[int, str] = field(default_factory=dict)
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""


def _extract_quoted(line: str) -> str:
    """Extract the first double-quoted substring from a line."""
    parts = line.split('"')
    return parts[1] if len(parts) >= 2 else ""


def _parse_xvg_lines(path: StrPath) -> tuple[_XVGMetadata, list[str]]:
    """Separate an XVG file into metadata and data lines."""
    meta = _XVGMetadata()
    data_lines: list[str] = []

    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("@"):
                _parse_xvg_directive(stripped, meta)
                continue
            if stripped:
                data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data lines found in {path}.")
    return meta, data_lines


def _parse_xvg_directive(line: str, meta: _XVGMetadata) -> None:
    """Parse a single ``@`` directive and update metadata in place."""
    legend_match = _LEGEND_PATTERN.search(line)
    if legend_match:
        meta.legends[int(legend_match.group(1))] = legend_match.group(2)
        return

    lower = line.lower()
    if "title" in lower:
        meta.title = _extract_quoted(line)
    elif "xaxis" in lower and "label" in lower:
        meta.xlabel = _extract_quoted(line)
    elif "yaxis" in lower and "label" in lower:
        meta.ylabel = _extract_quoted(line)


def _build_column_names(meta: _XVGMetadata, n_cols: int) -> list[str]:
    """Build DataFrame column names from parsed metadata."""
    columns: list[str] = [meta.xlabel or "Time"]
    for col_index in range(1, n_cols):
        legend_index = col_index - 1
        if legend_index in meta.legends:
            columns.append(meta.legends[legend_index])
        elif n_cols == 2 and meta.ylabel:
            columns.append(meta.ylabel)
        else:
            columns.append(f"col_{col_index}")
    return columns


def read_xvg(path: StrPath) -> pd.DataFrame:
    """Read a GROMACS XVG file into a DataFrame.

    Parses metadata lines (lines starting with ``@``) to extract column labels
    from legend entries. Data lines are read with NumPy for performance.

    Args:
        path: Path to a ``.xvg`` file.

    Returns:
        DataFrame whose first column is typically time and remaining columns
        are labeled from the XVG legend entries (or ``"col_0"``, ``"col_1"``,
        etc. when legends are absent).
    """
    meta, data_lines = _parse_xvg_lines(path)

    data = np.loadtxt(data_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    columns = _build_column_names(meta, data.shape[1])
    df = pd.DataFrame(data, columns=columns[: data.shape[1]])
    df.attrs["title"] = meta.title
    df.attrs["xlabel"] = meta.xlabel
    df.attrs["ylabel"] = meta.ylabel
    return df


def read_edr(path: StrPath) -> pd.DataFrame:
    """Read a GROMACS EDR energy file into a DataFrame.

    Uses ``panedr`` internally. Install it with ``pip install panedr``.

    Args:
        path: Path to a ``.edr`` file.

    Returns:
        DataFrame with a ``Time`` column and one column per energy term.

    Raises:
        ImportError: If ``panedr`` is not installed.
    """
    import panedr

    return panedr.edr_to_df(str(path))
