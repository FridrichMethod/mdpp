"""Command-line interface for mdpp script and data utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_list(args: argparse.Namespace) -> None:
    """Handle the ``list`` subcommand."""
    from mdpp.scripts import list_scripts

    for script in list_scripts(args.prefix):
        print(script)


def _cmd_show(args: argparse.Namespace) -> None:
    """Handle the ``show`` subcommand."""
    from mdpp.scripts import read_script

    try:
        print(read_script(args.path))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def _cmd_copy(args: argparse.Namespace) -> None:
    """Handle the ``copy`` subcommand."""
    from mdpp.scripts import copy_scripts

    try:
        written = copy_scripts(args.category, args.dest, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    for path in written:
        print(path)


def _cmd_mdps(args: argparse.Namespace) -> None:
    """Handle the ``mdps`` subcommand."""
    from mdpp.data import copy_mdp_files

    try:
        written = copy_mdp_files(args.dest, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    for path in written:
        print(path)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mdpp",
        description="MD simulation pre- and post-processing utilities.",
    )
    sub = parser.add_subparsers(dest="command")

    # mdpp list [prefix]
    p_list = sub.add_parser("list", help="List available utility scripts.")
    p_list.add_argument(
        "prefix",
        nargs="?",
        default="",
        help="Filter by prefix (e.g. 'gromacs/analysis').",
    )

    # mdpp show <path>
    p_show = sub.add_parser("show", help="Print a script's content to stdout.")
    p_show.add_argument("path", help="Relative path (e.g. 'gromacs/runtime/restart.sh').")

    # mdpp copy <category> <dest>
    p_copy = sub.add_parser("copy", help="Copy scripts in a category to a directory.")
    p_copy.add_argument("category", help="Category path (e.g. 'gromacs/analysis').")
    p_copy.add_argument("dest", type=Path, help="Destination directory.")
    p_copy.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    # mdpp mdps <dest>
    p_mdps = sub.add_parser("mdps", help="Copy MDP template files to a directory.")
    p_mdps.add_argument("dest", type=Path, help="Destination directory.")
    p_mdps.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``mdpp`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "list": _cmd_list,
        "show": _cmd_show,
        "copy": _cmd_copy,
        "mdps": _cmd_mdps,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(0)

    handler(args)


if __name__ == "__main__":
    main()
