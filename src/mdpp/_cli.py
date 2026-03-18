"""Command-line interface for mdpp data utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
        "mdps": _cmd_mdps,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(0)

    handler(args)


if __name__ == "__main__":
    main()
