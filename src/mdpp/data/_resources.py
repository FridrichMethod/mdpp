"""Low-level helpers for locating and copying bundled package data."""

from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

from mdpp._types import StrPath

# File extensions for bundled data files.
_DATA_EXTENSIONS = frozenset({".mdp"})


def _package_files(package: str) -> Traversable:
    """Return the ``importlib.resources`` Traversable root for *package*."""
    return files(package)


def _iter_files(root: Traversable, *, prefix: str = "") -> list[str]:
    """Recursively list all data files under *root*.

    Returns relative POSIX paths (``"mdps/step5_production.mdp"``).
    Skips ``__init__.py``, ``__pycache__``, and ``.pyc`` files.
    """
    result: list[str] = []
    for item in root.iterdir():
        rel = f"{prefix}{item.name}" if prefix == "" else f"{prefix}/{item.name}"
        if item.is_dir():
            if item.name in {"__pycache__"}:
                continue
            result.extend(_iter_files(item, prefix=rel))
        elif item.is_file():
            if item.name == "__init__.py" or item.name.endswith(".pyc"):
                continue
            # Only include known data extensions.
            suffix = Path(item.name).suffix
            if suffix in _DATA_EXTENSIONS:
                result.append(rel)
    return result


def list_files(package: str, *, prefix: str = "") -> list[str]:
    """List all data files under *package*, optionally filtered by *prefix*.

    Args:
        package: Dotted package name (e.g. ``"mdpp.data"``).
        prefix: Optional slash-separated prefix to filter by
            (e.g. ``"gromacs/analysis"``).

    Returns:
        Sorted list of relative POSIX paths.
    """
    root = _package_files(package)
    all_files = _iter_files(root)
    if prefix:
        normalized = prefix.rstrip("/")
        all_files = [f for f in all_files if f.startswith(normalized)]
    return sorted(all_files)


def get_resource_path(package: str, relative_path: str) -> Path:
    """Return an absolute filesystem ``Path`` to a bundled file.

    Args:
        package: Dotted package name (e.g. ``"mdpp.data"``).
        relative_path: Slash-separated path relative to the package root
            (e.g. ``"mdps/step5_production.mdp"``).

    Returns:
        Absolute path. For editable installs this points directly into the
        source tree; for wheel installs it may be a temporary extraction.

    Raises:
        FileNotFoundError: If the file does not exist in the package.
    """
    parts = relative_path.split("/")
    node: Traversable = _package_files(package)
    for part in parts:
        node = node.joinpath(part)

    # Materialise to a real filesystem path.
    ctx = as_file(node)
    path = ctx.__enter__()
    if not path.exists():
        ctx.__exit__(None, None, None)
        msg = f"{relative_path!r} not found in {package}"
        raise FileNotFoundError(msg)
    return path


def read_text(package: str, relative_path: str) -> str:
    """Read and return the text content of a bundled file.

    Args:
        package: Dotted package name.
        relative_path: Slash-separated path relative to the package root.

    Returns:
        File content as a string.
    """
    parts = relative_path.split("/")
    node: Traversable = _package_files(package)
    for part in parts:
        node = node.joinpath(part)
    return node.read_text(encoding="utf-8")


def copy_file(
    package: str,
    relative_path: str,
    dest: StrPath,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a single bundled file to *dest*.

    If *dest* is a directory the file is placed inside it keeping its
    original name.  Returns the path of the written file.

    Args:
        package: Dotted package name.
        relative_path: Slash-separated path relative to the package root.
        dest: Destination file or directory.
        overwrite: Allow overwriting an existing file.

    Returns:
        Path to the written file.

    Raises:
        FileExistsError: If target exists and *overwrite* is ``False``.
    """
    src = get_resource_path(package, relative_path)
    dest_path = Path(dest)
    if dest_path.is_dir():
        dest_path = dest_path / src.name
    if dest_path.exists() and not overwrite:
        msg = f"{dest_path} already exists"
        raise FileExistsError(msg)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    return dest_path


def copy_tree(
    package: str,
    relative_dir: str,
    dest: StrPath,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Copy every data file under *relative_dir* into *dest*.

    Preserves subdirectory structure.  Returns the list of written files.

    Args:
        package: Dotted package name.
        relative_dir: Slash-separated directory path relative to the package.
        dest: Destination directory.
        overwrite: Allow overwriting existing files.

    Returns:
        List of paths to the written files.
    """
    matching = list_files(package, prefix=relative_dir)
    if not matching:
        msg = f"No files found under {relative_dir!r} in {package}"
        raise FileNotFoundError(msg)

    dest_root = Path(dest)
    written: list[Path] = []
    normalized = relative_dir.rstrip("/")
    for rel in matching:
        # Strip the prefix so files are placed directly in dest.
        suffix = rel[len(normalized) :].lstrip("/")
        target = dest_root / suffix
        if target.exists() and not overwrite:
            msg = f"{target} already exists"
            raise FileExistsError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)
        src = get_resource_path(package, rel)
        shutil.copy2(src, target)
        written.append(target)
    return written
