"""APBS Poisson-Boltzmann input generation and log parsing.

Helpers for driving APBS (Adaptive Poisson-Boltzmann Solver) from Python.

Two pure functions are exposed:

- :func:`write_apbs_input` generates a multigrid APBS ``.in`` file from an
  existing ``.pqr`` by deriving the grid bounding box from the radius-inflated
  atom coordinates and rounding ``dime`` up to the nearest ``c * 2**n + 1``
  value required by APBS multigrid.
- :func:`infer_debye_length` parses a Debye length (in Angstrom) out of an
  APBS log; used to bootstrap downstream BrownDye input from the same APBS run.

Both functions are pure Python with no subprocess calls; they only read/write
text files. Run APBS itself by calling the ``apbs`` CLI separately.
"""

from __future__ import annotations

import re
from math import ceil
from pathlib import Path

from mdpp._types import StrPath

DEFAULT_IONIC_STRENGTH_M: float = 0.150
DEFAULT_SOLUTE_DIELECTRIC: float = 2.0
DEFAULT_SOLVENT_DIELECTRIC: float = 78.54
DEFAULT_SOLVENT_RADIUS_A: float = 1.4
DEFAULT_TEMPERATURE_K: float = 298.15
DEFAULT_FINE_SPACING_A: float = 0.75
DEFAULT_FINE_PADDING_A: float = 60.0
DEFAULT_COARSE_PADDING_A: float = 100.0

_NA_PAULING_RADIUS_A: float = 1.8750
_CL_PAULING_RADIUS_A: float = 1.8150

_DEBYE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[Dd]ebye[- ]length[:\s]+([0-9.]+)"),
    re.compile(r"[Gg]ot debye length\s+([0-9.]+)"),
)


def _read_pqr_atoms(pqr_path: Path) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Parse atom coordinates and radii from a PQR file.

    Args:
        pqr_path: Path to a PQR file with whitespace-separated columns.

    Returns:
        ``(coords, radii)`` with one entry per ``ATOM``/``HETATM`` row.
        Coordinates are read from the 5th-to-last through 3rd-to-last
        whitespace-separated fields; the radius is the last field.

    Raises:
        ValueError: if no ATOM/HETATM rows are present.
    """
    coords: list[tuple[float, float, float]] = []
    radii: list[float] = []
    with pqr_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            x, y, z = (float(value) for value in fields[-5:-2])
            coords.append((x, y, z))
            radii.append(float(fields[-1]))
    if not coords:
        raise ValueError(f"No ATOM/HETATM records found in {pqr_path}.")
    return coords, radii


def _apbs_friendly_dime(length: float, spacing: float) -> int:
    """Round ``length`` up to the nearest APBS multigrid grid count.

    APBS multigrid requires ``dime = c * 2**n + 1`` for small integers ``c``
    and levels ``n``. We pick the smallest such number that covers
    ``ceil(length / spacing) + 1``.

    Args:
        length: edge length of the fine grid in Angstrom.
        spacing: target fine grid spacing in Angstrom.

    Returns:
        APBS-compatible integer grid count.
    """
    target = ceil(length / spacing) + 1
    candidates = sorted({c * 2**n + 1 for c in range(1, 7) for n in range(1, 12)})
    for candidate in candidates:
        if candidate >= target:
            return candidate
    return candidates[-1]


def write_apbs_input(
    stem: str,
    work_dir: StrPath,
    *,
    ionic_strength_m: float = DEFAULT_IONIC_STRENGTH_M,
    solute_dielectric: float = DEFAULT_SOLUTE_DIELECTRIC,
    solvent_dielectric: float = DEFAULT_SOLVENT_DIELECTRIC,
    solvent_radius_a: float = DEFAULT_SOLVENT_RADIUS_A,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    fine_spacing_a: float = DEFAULT_FINE_SPACING_A,
    fine_padding_a: float = DEFAULT_FINE_PADDING_A,
    coarse_padding_a: float = DEFAULT_COARSE_PADDING_A,
) -> Path:
    """Write an APBS multigrid input file ``{stem}.in`` for ``{stem}.pqr``.

    Physics defaults mirror ``pdb2pqr --apbs-input`` canonical defaults
    (``lpbe`` / ``bcfl sdh`` / ``srfm smol`` / ``chgm spl2`` / ``pdie 2.0`` /
    ``sdie 78.54`` / ``srad 1.4`` / ``sdens 10`` / ``swin 0.30`` /
    ``temp 298.15``) with two intentional overrides:

    - Explicit Na+/Cl- ion lines at ``ionic_strength_m`` with Pauling radii
      (1.875 / 1.815 A). The pdb2pqr canonical input omits ions, which sets
      the Debye length to infinity. Explicit ions are required when the
      resulting ``.dx`` feeds a BrownDye2 simulation that needs a finite
      Debye length for far-field electrostatics.
    - Larger grid padding (``fine_padding_a`` / ``coarse_padding_a``) than
      pdb2pqr's ``fadd=20`` / ``cfac=1.7`` defaults so the outer grid
      comfortably exceeds the BrownDye b-radius. ``dime`` is rounded up to
      the nearest ``c * 2**n + 1`` value required by APBS multigrid.

    Args:
        stem: PQR file stem (without extension). ``work_dir / "{stem}.pqr"``
            must already exist; the input file is written next to it at
            ``work_dir / "{stem}.in"``.
        work_dir: directory containing ``{stem}.pqr``.
        ionic_strength_m: ion concentration in mol/L (default 0.150 M).
        solute_dielectric: interior (solute) dielectric constant.
        solvent_dielectric: bulk solvent dielectric constant.
        solvent_radius_a: probe solvent radius in Angstrom.
        temperature_k: simulation temperature in Kelvin.
        fine_spacing_a: target fine grid spacing in Angstrom.
        fine_padding_a: fine grid padding added to the radius-inflated
            atom bounding box (per axis).
        coarse_padding_a: coarse grid padding added to the radius-inflated
            atom bounding box (per axis); the coarse grid never shrinks
            below the fine grid.

    Returns:
        Path to the written ``{stem}.in`` file.

    Raises:
        ValueError: if ``{stem}.pqr`` has no ATOM/HETATM records.
    """
    work = Path(work_dir)
    pqr_path = work / f"{stem}.pqr"
    apbs_input = work / f"{stem}.in"

    coords, radii = _read_pqr_atoms(pqr_path)

    lower = [min(c[i] - radii[idx] for idx, c in enumerate(coords)) for i in range(3)]
    upper = [max(c[i] + radii[idx] for idx, c in enumerate(coords)) for i in range(3)]
    center = [(lo + hi) / 2.0 for lo, hi in zip(lower, upper, strict=True)]
    span = [hi - lo for lo, hi in zip(lower, upper, strict=True)]
    fglen = [value + fine_padding_a for value in span]
    cglen = [max(value + coarse_padding_a, fine) for value, fine in zip(span, fglen, strict=True)]
    dime = [_apbs_friendly_dime(length, fine_spacing_a) for length in fglen]

    apbs_input.write_text(
        f"""read
    mol pqr {stem}.pqr
end
elec
    mg-auto
    dime {dime[0]} {dime[1]} {dime[2]}
    cglen {cglen[0]:.4f} {cglen[1]:.4f} {cglen[2]:.4f}
    fglen {fglen[0]:.4f} {fglen[1]:.4f} {fglen[2]:.4f}
    cgcent {center[0]:.4f} {center[1]:.4f} {center[2]:.4f}
    fgcent {center[0]:.4f} {center[1]:.4f} {center[2]:.4f}
    mol 1
    lpbe
    bcfl sdh
    ion charge -1.00 conc {ionic_strength_m:.4f} radius {_CL_PAULING_RADIUS_A:.4f}
    ion charge 1.00 conc {ionic_strength_m:.4f} radius {_NA_PAULING_RADIUS_A:.4f}
    pdie {solute_dielectric:.4f}
    sdie {solvent_dielectric:.4f}
    srfm smol
    chgm spl2
    sdens 10.00
    srad {solvent_radius_a:.4f}
    swin 0.30
    temp {temperature_k:.2f}
    calcenergy total
    calcforce no
    write pot dx {stem}
end
print elecEnergy 1 end
quit
""",
        encoding="utf-8",
    )
    return apbs_input


def infer_debye_length(*apbs_logs: StrPath) -> float:
    """Return the first Debye length (Angstrom) parsed from any APBS log.

    Scans logs in argument order; returns as soon as a Debye length is
    found in any one of them. Missing log files are skipped silently.

    Args:
        *apbs_logs: paths to one or more APBS log files.

    Returns:
        Debye length in Angstrom.

    Raises:
        RuntimeError: if no log contains a recognisable Debye length entry.
    """
    for raw_path in apbs_logs:
        path = Path(raw_path)
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        for pattern in _DEBYE_PATTERNS:
            match = pattern.search(text)
            if match:
                return float(match.group(1))
    paths = [str(Path(p)) for p in apbs_logs]
    raise RuntimeError(f"Could not infer Debye length from any of: {paths}")
