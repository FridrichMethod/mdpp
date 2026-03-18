"""Bundled data resources for MD simulations.

Provides access to MDP (Molecular Dynamics Parameter) template files
distributed with the ``mdpp`` package.

Example::

    from mdpp.data import list_mdp_templates, get_mdp_template, copy_mdp_files

    # List available MDP templates
    list_mdp_templates()
    # ['mdps/step4.0_minimization.mdp', ..., 'mdps/step5_production.mdp']

    # Read MDP content as a string
    content = get_mdp_template("step5_production")

    # Copy all MDP files to a working directory
    copy_mdp_files("./my_simulation/")
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
    "copy_mdp_files",
    "get_mdp_template",
    "list_mdp_templates",
]

_PACKAGE = "mdpp.data"


def list_mdp_templates() -> list[str]:
    """List available MDP template files.

    Returns:
        Sorted list of MDP filenames (e.g.
        ``["mdps/step4.0_minimization.mdp", ...]``).
    """
    return _list_files(_PACKAGE, prefix="mdps")


def get_mdp_template(name: str) -> str:
    """Return the text content of an MDP template.

    Args:
        name: MDP name with or without the ``.mdp`` extension, and with or
            without the ``mdps/`` prefix.  For example, all of these work:
            ``"step5_production"``, ``"step5_production.mdp"``,
            ``"mdps/step5_production.mdp"``.

    Returns:
        MDP file content as a string.

    Raises:
        FileNotFoundError: If no matching MDP template is found.
    """
    if not name.endswith(".mdp"):
        name = f"{name}.mdp"
    if not name.startswith("mdps/"):
        name = f"mdps/{name}"
    return _read_text(_PACKAGE, name)


def copy_mdp_files(
    dest: StrPath,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Copy all MDP template files to *dest*.

    Args:
        dest: Destination directory.
        overwrite: Allow overwriting existing files.

    Returns:
        List of paths to the written files.
    """
    return _copy_tree(_PACKAGE, "mdps", dest, overwrite=overwrite)
