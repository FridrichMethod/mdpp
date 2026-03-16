"""Bundled shell scripts and utilities for MD engines.

Provides discovery, inspection, and copying of utility scripts that are
distributed with the ``mdpp`` package.  These include GROMACS analysis
wrappers, compilation helpers, runtime monitoring tools, and more.

Example::

    from mdpp.scripts import list_scripts, copy_scripts, get_script_path

    # List everything available
    list_scripts()  # all scripts
    list_scripts("gromacs/analysis")  # just analysis wrappers

    # Get the filesystem path to a single script
    path = get_script_path("gromacs/runtime/restart.sh")

    # Copy an entire category to a working directory
    copy_scripts("gromacs/analysis", dest="./my_simulation/")
"""

from pathlib import Path

from mdpp._types import StrPath
from mdpp.scripts._resources import (
    copy_file as _copy_file,
)
from mdpp.scripts._resources import (
    copy_tree as _copy_tree,
)
from mdpp.scripts._resources import (
    get_resource_path as _get_resource_path,
)
from mdpp.scripts._resources import (
    list_files as _list_files,
)
from mdpp.scripts._resources import (
    read_text as _read_text,
)

__all__ = [
    "copy_scripts",
    "get_script_path",
    "list_scripts",
    "read_script",
]

_PACKAGE = "mdpp.scripts"


def list_scripts(prefix: str = "") -> list[str]:
    """List available utility scripts, optionally filtered by *prefix*.

    Args:
        prefix: Slash-separated prefix to filter by
            (e.g. ``"gromacs/analysis"``).

    Returns:
        Sorted list of relative paths such as
        ``"gromacs/analysis/gmx_rmsd.sh"``.
    """
    return _list_files(_PACKAGE, prefix=prefix)


def get_script_path(relative_path: str) -> Path:
    """Return the filesystem path to a bundled script.

    Args:
        relative_path: Slash-separated path relative to
            ``mdpp/scripts/`` (e.g. ``"gromacs/runtime/restart.sh"``).

    Returns:
        Absolute ``Path`` to the file.

    Raises:
        FileNotFoundError: If the script does not exist.
    """
    return _get_resource_path(_PACKAGE, relative_path)


def read_script(relative_path: str) -> str:
    """Return the text content of a bundled script.

    Args:
        relative_path: Slash-separated path relative to ``mdpp/scripts/``.

    Returns:
        File content as a string.
    """
    return _read_text(_PACKAGE, relative_path)


def copy_scripts(
    category: str,
    dest: StrPath,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Copy all scripts in *category* to *dest*.

    Args:
        category: Slash-separated category path
            (e.g. ``"gromacs/analysis"``, ``"gromacs/runtime"``).
        dest: Destination directory.
        overwrite: Allow overwriting existing files.

    Returns:
        List of paths to the written files.
    """
    return _copy_tree(_PACKAGE, category, dest, overwrite=overwrite)
