"""Bundled data resources for MD simulations.

Provides access to MDP (Molecular Dynamics Parameter) template files
distributed with the ``mdpp`` package.  Templates are organized by
force field: ``charmm`` (CHARMM36) and ``amber`` (AMBER ff14SB/ff19SB).

Example::

    from mdpp.data import list_mdp_templates, get_mdp_template, copy_mdp_files

    # List available MDP templates for a force field
    list_mdp_templates("charmm")
    # ['mdps/charmm/step4.0_minimization.mdp', ...]

    # Read MDP content as a string
    content = get_mdp_template("step5_production", ff="amber")

    # Copy all MDP files for a force field to a working directory
    copy_mdp_files("./my_simulation/", ff="charmm")
"""

from pathlib import Path

from mdpp._types import StrPath
from mdpp.data._resources import (
    copy_tree as _copy_tree,
)
from mdpp.data._resources import (
    list_files as _list_files,
)
from mdpp.data._resources import (
    read_text as _read_text,
)

__all__ = [
    "FORCE_FIELDS",
    "copy_mdp_files",
    "get_mdp_template",
    "list_mdp_templates",
]

_PACKAGE = "mdpp.data"

FORCE_FIELDS: frozenset[str] = frozenset({"charmm", "amber"})


def _validate_ff(ff: str) -> str:
    """Validate and return the force field name."""
    ff = ff.lower()
    if ff not in FORCE_FIELDS:
        msg = f"Unknown force field {ff!r}; choose from {sorted(FORCE_FIELDS)}"
        raise ValueError(msg)
    return ff


def list_mdp_templates(ff: str = "charmm") -> list[str]:
    """List available MDP template files for a force field.

    Args:
        ff: Force field name (``"charmm"`` or ``"amber"``).

    Returns:
        Sorted list of MDP filenames (e.g.
        ``["mdps/charmm/step4.0_minimization.mdp", ...]``).
    """
    ff = _validate_ff(ff)
    return _list_files(_PACKAGE, prefix=f"mdps/{ff}")


def get_mdp_template(name: str, *, ff: str = "charmm") -> str:
    """Return the text content of an MDP template.

    Args:
        name: MDP name with or without the ``.mdp`` extension.
            For example: ``"step5_production"`` or ``"step5_production.mdp"``.
        ff: Force field name (``"charmm"`` or ``"amber"``).

    Returns:
        MDP file content as a string.

    Raises:
        FileNotFoundError: If no matching MDP template is found.
        ValueError: If *ff* is not a recognized force field.
    """
    ff = _validate_ff(ff)
    if not name.endswith(".mdp"):
        name = f"{name}.mdp"
    # Strip any leading prefix so callers can pass bare names or full paths.
    bare = name.split("/")[-1]
    return _read_text(_PACKAGE, f"mdps/{ff}/{bare}")


def copy_mdp_files(
    dest: StrPath,
    *,
    ff: str = "charmm",
    overwrite: bool = False,
) -> list[Path]:
    """Copy all MDP template files for a force field to *dest*.

    Args:
        dest: Destination directory.
        ff: Force field name (``"charmm"`` or ``"amber"``).
        overwrite: Allow overwriting existing files.

    Returns:
        List of paths to the written files.
    """
    ff = _validate_ff(ff)
    return _copy_tree(_PACKAGE, f"mdps/{ff}", dest, overwrite=overwrite)
