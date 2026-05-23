"""BrownDye2 input.xml and contact_types.xml generation.

Helpers for building the XML inputs consumed by BrownDye2's ``bd_top``:

- :func:`write_contact_types` writes ``contact_types.xml`` with one entry
  per unique ``(atom_name, residue_name)`` heavy-atom pair per body.
- :func:`build_input_xml` and :func:`write_input_xml` produce the top-level
  ``input.xml`` from two :class:`BrownDyeBody` descriptors and a shared
  :class:`BrownDyeSolvent` configuration.

All helpers are pure Python: no subprocess calls, no XML schema validation.
Run BrownDye's own tools (``pqr2xml``, ``make_rxn_pairs``, ``make_rxn_file``,
``bd_top``, ``nam_simulation``) separately.

The Debye length feeding :class:`BrownDyeSolvent` is typically obtained from
an APBS run via :func:`mdpp.prep.apbs.infer_debye_length`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mdpp._types import StrPath

DEFAULT_N_THREADS: int = 1
DEFAULT_SEED: int = 11111111
DEFAULT_N_TRAJECTORIES: int = 100_000
DEFAULT_N_TRAJECTORIES_PER_OUTPUT: int = 10
DEFAULT_MAX_N_STEPS: int = 1_000_000
DEFAULT_N_STEPS_PER_OUTPUT: int = 10
DEFAULT_RESULTS_FILE: str = "results.xml"
DEFAULT_TRAJECTORY_FILE: str = "trajectory"
DEFAULT_REACTION_FILE: str = "reactions.xml"

DEFAULT_BD_SOLVENT_DIELECTRIC: float = 78.0
DEFAULT_RELATIVE_VISCOSITY: float = 1.0
DEFAULT_KT: float = 1.0
DEFAULT_DESOLVATION_PARAMETER: float = 1.0
DEFAULT_SOLVENT_RADIUS_A: float = 1.4

DEFAULT_BODY_DIELECTRIC: float = 4.0


@dataclass(frozen=True, slots=True)
class BrownDyeBody:
    """Configuration for one BrownDye core/body block in ``input.xml``.

    Attributes:
        name: BrownDye body name. Also used as the ``<core><name>`` tag.
        atoms_xml: Relative path to the ``atoms.xml`` produced by
            ``pqr2xml``, as it should appear inside ``input.xml``
            (typically just ``"{name}_atoms.xml"`` when running
            BrownDye from the same directory).
        grid_dx: Relative path to the APBS ``.dx`` grid for this body.
        is_protein: Maps to the ``<is_protein>`` tag (lowercase
            ``true``/``false`` in the serialised XML).
        dielectric: Interior dielectric for this body.
        all_in_surface: Maps to the ``<all_in_surface>`` tag.
    """

    name: str
    atoms_xml: str
    grid_dx: str
    is_protein: bool = True
    dielectric: float = DEFAULT_BODY_DIELECTRIC
    all_in_surface: bool = False


@dataclass(frozen=True, slots=True)
class BrownDyeSolvent:
    """Solvent block parameters shared by all bodies in a BrownDye system.

    BrownDye uses kT-units internally, so :attr:`dielectric` is the BrownDye
    solvent dielectric (typically ``78.0``) and may differ from the APBS
    ``sdie`` value used to compute the electrostatic grid.

    Attributes:
        debye_length_a: Debye length in Angstrom (usually obtained from the
            APBS log via :func:`mdpp.prep.apbs.infer_debye_length`).
        dielectric: BrownDye solvent dielectric (kT-units).
        relative_viscosity: Relative solvent viscosity.
        kT: Thermal energy unit (BrownDye uses ``kT = 1``).
        desolvation_parameter: BrownDye desolvation scale factor.
        solvent_radius_a: Probe solvent radius in Angstrom.
    """

    debye_length_a: float
    dielectric: float = DEFAULT_BD_SOLVENT_DIELECTRIC
    relative_viscosity: float = DEFAULT_RELATIVE_VISCOSITY
    kT: float = DEFAULT_KT
    desolvation_parameter: float = DEFAULT_DESOLVATION_PARAMETER
    solvent_radius_a: float = DEFAULT_SOLVENT_RADIUS_A


def _heavy_atom_keys(pqr_path: Path) -> list[tuple[str, str]]:
    """Return ordered unique ``(atom_name, residue_name)`` heavy-atom keys.

    Hydrogens are filtered out by atom-name prefix (case-insensitive ``H``).
    Order matches first appearance in the PQR file.

    Args:
        pqr_path: PQR file to scan.

    Returns:
        Insertion-ordered list of unique heavy-atom keys.
    """
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for line in pqr_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        atom = parts[2]
        residue = parts[3]
        key = (atom, residue)
        if atom and not atom.upper().startswith("H") and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def write_contact_types(
    mol0_pqr: StrPath,
    mol1_pqr: StrPath,
    out_path: StrPath,
) -> Path:
    """Write a BrownDye ``contact_types.xml`` from two PQR files.

    Lists every unique heavy-atom ``(atom_name, residue_name)`` per body.
    The output is consumed by ``make_rxn_pairs`` to enumerate candidate
    contact pairs between the two bodies.

    Args:
        mol0_pqr: PQR file for the first body (writes ``<molecule0>`` block).
        mol1_pqr: PQR file for the second body (writes ``<molecule1>`` block).
        out_path: Destination ``contact_types.xml`` path.

    Returns:
        ``out_path`` as a :class:`Path`, for chaining.
    """
    out = Path(out_path)
    lines: list[str] = ["<contacts>", "  <combinations>"]
    for label, entries in (
        ("molecule0", _heavy_atom_keys(Path(mol0_pqr))),
        ("molecule1", _heavy_atom_keys(Path(mol1_pqr))),
    ):
        lines.append(f"    <{label}>")
        for atom, residue in entries:
            lines.append(
                f"      <contact> <atom> {atom} </atom> <residue> {residue} </residue> </contact>"
            )
        lines.append(f"    </{label}>")
    lines += ["  </combinations>", "</contacts>", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _bool_xml(value: bool) -> str:
    """Render a Python bool as a lowercase XML ``true``/``false`` token."""
    return "true" if value else "false"


def _body_block(body: BrownDyeBody) -> str:
    """Format a single ``<group><core>...</core></group>`` block."""
    return f"""    <group>
      <name> {body.name} </name>
      <core>
        <name> {body.name} </name>
        <all_in_surface> {_bool_xml(body.all_in_surface)} </all_in_surface>
        <is_protein> {_bool_xml(body.is_protein)} </is_protein>
        <atoms> {body.atoms_xml} </atoms>
        <electric_field>
          <grid> {body.grid_dx} </grid>
        </electric_field>
        <dielectric> {body.dielectric} </dielectric>
      </core>
    </group>"""


def build_input_xml(
    body0: BrownDyeBody,
    body1: BrownDyeBody,
    *,
    solvent: BrownDyeSolvent,
    reaction_file: str = DEFAULT_REACTION_FILE,
    n_threads: int = DEFAULT_N_THREADS,
    seed: int = DEFAULT_SEED,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    n_trajectories_per_output: int = DEFAULT_N_TRAJECTORIES_PER_OUTPUT,
    max_n_steps: int = DEFAULT_MAX_N_STEPS,
    n_steps_per_output: int = DEFAULT_N_STEPS_PER_OUTPUT,
    results_file: str = DEFAULT_RESULTS_FILE,
    trajectory_file: str = DEFAULT_TRAJECTORY_FILE,
) -> str:
    """Build the BrownDye top-level ``input.xml`` as a string.

    The minimum core ``dt`` tolerances are hardcoded to ``0.0`` (BrownDye's
    own defaults); the time step is determined dynamically. Override after
    the fact if you need non-default tolerances.

    Args:
        body0: First body descriptor.
        body1: Second body descriptor.
        solvent: Shared solvent parameters (including Debye length).
        reaction_file: Filename of the BrownDye reaction definition XML.
        n_threads: Number of BrownDye worker threads.
        seed: Random seed for trajectory propagation.
        n_trajectories: Total number of trajectories to launch.
        n_trajectories_per_output: Trajectories per ``results.xml`` flush.
        max_n_steps: Maximum BrownDye steps per trajectory.
        n_steps_per_output: Stride between trajectory frames written to
            ``trajectory{N}.xml``. Set to ``1`` to record every step.
        results_file: Filename for cumulative results.
        trajectory_file: Base name for per-thread trajectory XML dumps
            (BrownDye writes ``{trajectory_file}{thread}.xml`` plus a
            matching ``.index.xml``).

    Returns:
        The full ``input.xml`` content as a UTF-8 string.
    """
    return f"""<top>
  <n_threads> {n_threads} </n_threads>
  <seed> {seed} </seed>
  <output> {results_file} </output>

  <n_trajectories> {n_trajectories} </n_trajectories>
  <n_trajectories_per_output> {n_trajectories_per_output} </n_trajectories_per_output>
  <max_n_steps> {max_n_steps} </max_n_steps>
  <trajectory_file> {trajectory_file} </trajectory_file>
  <n_steps_per_output> {n_steps_per_output} </n_steps_per_output>

  <system>
    <reaction_file> {reaction_file} </reaction_file>

    <solvent>
      <debye_length> {solvent.debye_length_a} </debye_length>
      <dielectric> {solvent.dielectric} </dielectric>
      <relative_viscosity> {solvent.relative_viscosity} </relative_viscosity>
      <kT> {solvent.kT} </kT>
      <desolvation_parameter> {solvent.desolvation_parameter} </desolvation_parameter>
      <solvent_radius> {solvent.solvent_radius_a} </solvent_radius>
    </solvent>

    <time_step_tolerances>
      <minimum_core_dt> 0.0 </minimum_core_dt>
      <minimum_core_reaction_dt> 0.0 </minimum_core_reaction_dt>
    </time_step_tolerances>

{_body_block(body0)}

{_body_block(body1)}
  </system>
</top>
"""


def write_input_xml(
    out_path: StrPath,
    body0: BrownDyeBody,
    body1: BrownDyeBody,
    *,
    solvent: BrownDyeSolvent,
    reaction_file: str = DEFAULT_REACTION_FILE,
    n_threads: int = DEFAULT_N_THREADS,
    seed: int = DEFAULT_SEED,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    n_trajectories_per_output: int = DEFAULT_N_TRAJECTORIES_PER_OUTPUT,
    max_n_steps: int = DEFAULT_MAX_N_STEPS,
    n_steps_per_output: int = DEFAULT_N_STEPS_PER_OUTPUT,
    results_file: str = DEFAULT_RESULTS_FILE,
    trajectory_file: str = DEFAULT_TRAJECTORY_FILE,
) -> Path:
    """Write the BrownDye top-level ``input.xml`` to ``out_path``.

    Thin filesystem wrapper around :func:`build_input_xml`; see that
    function for parameter semantics.

    Returns:
        ``out_path`` as a :class:`Path`, for chaining.
    """
    text = build_input_xml(
        body0,
        body1,
        solvent=solvent,
        reaction_file=reaction_file,
        n_threads=n_threads,
        seed=seed,
        n_trajectories=n_trajectories,
        n_trajectories_per_output=n_trajectories_per_output,
        max_n_steps=max_n_steps,
        n_steps_per_output=n_steps_per_output,
        results_file=results_file,
        trajectory_file=trajectory_file,
    )
    out = Path(out_path)
    out.write_text(text, encoding="utf-8")
    return out
