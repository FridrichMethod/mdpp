"""Tests for scripts/gromacs/postprocessing/run_postprocessing.sh."""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

RUN_PP_ORIG = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "gromacs"
    / "postprocessing"
    / "run_postprocessing.sh"
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


PRODUCTION = "step5_production"


def _setup_workspace(
    tmp_path: Path,
    subdirs: list[str],
    *,
    with_tpr: bool = True,
    failing: set[str] | None = None,
) -> tuple[Path, Path]:
    """Build a fake postprocessing workspace.

    Copies run_postprocessing.sh into a temp script dir alongside a stub
    gmx_postprocessing_fast.sh so SCRIPT_DIR resolution picks up the stub.

    ``subdirs`` may contain nested paths (e.g. ``"a/b/c"``); intermediate
    directories are created automatically via ``mkdir(parents=True)``.

    Returns (target_dir, script_copy_path).
    """
    # Copy the real script into a temp "scripts" dir with a stub companion.
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script_copy = script_dir / "run_postprocessing.sh"
    shutil.copy2(RUN_PP_ORIG, script_copy)

    fail_names = " ".join(failing or [])
    stub = textwrap.dedent(
        f"""\
        #!/bin/bash
        dir_name="$(basename "$PWD")"
        for f in {fail_names}; do
            [[ "$dir_name" == "$f" ]] && exit 1
        done
        exit 0
        """
    )
    stub_path = script_dir / "gmx_postprocessing_fast.sh"
    stub_path.write_text(stub)
    _make_executable(stub_path)

    # Build target directory with subdirs (supports nested paths).
    target = tmp_path / "target"
    target.mkdir()
    for name in subdirs:
        d = target / name
        d.mkdir(parents=True, exist_ok=True)
        if with_tpr:
            (d / f"{PRODUCTION}.tpr").write_text("")

    return target, script_copy


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        check=False,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSuccessfulRun:
    """All subdirectories complete successfully."""

    def test_all_succeed(self, tmp_path: Path) -> None:
        target, script = _setup_workspace(tmp_path, ["rep1", "rep2", "rep3"])
        result = _run(script, str(target), cwd=tmp_path)
        assert result.returncode == 0
        assert "All 3 jobs completed successfully" in _strip_ansi(result.stdout)


class TestFailingSubdir:
    """One subdirectory fails -- script reports failure and exits 1."""

    def test_one_failure(self, tmp_path: Path) -> None:
        target, script = _setup_workspace(tmp_path, ["good", "bad"], failing={"bad"})
        result = _run(script, str(target), cwd=tmp_path)
        assert result.returncode == 1
        out = _strip_ansi(result.stdout)
        assert "Failed jobs:" in out
        assert "bad" in out


class TestNoSubdirs:
    """Empty target directory -- nothing to process, exits 0."""

    def test_empty_dir(self, tmp_path: Path) -> None:
        target, script = _setup_workspace(tmp_path, [])
        result = _run(script, str(target), cwd=tmp_path)
        assert result.returncode == 0
        assert "All 0 jobs completed" in _strip_ansi(result.stdout)


class TestSubdirsWithoutTpr:
    """Subdirectories without step5_production.tpr are skipped."""

    def test_no_tpr_skipped(self, tmp_path: Path) -> None:
        target, script = _setup_workspace(tmp_path, ["has_tpr", "no_tpr"], with_tpr=False)
        (target / "has_tpr" / f"{PRODUCTION}.tpr").write_text("")
        result = _run(script, str(target), cwd=tmp_path)
        assert result.returncode == 0
        assert "All 1 jobs completed" in _strip_ansi(result.stdout)


class TestNotADirectory:
    """Non-existent target directory -- exits with error."""

    def test_missing_dir(self, tmp_path: Path) -> None:
        _, script = _setup_workspace(tmp_path, [])
        result = _run(script, str(tmp_path / "nonexistent"), cwd=tmp_path)
        assert result.returncode != 0
        assert "is not a directory" in _strip_ansi(result.stderr)


class TestNoArgument:
    """No directory argument -- prints usage."""

    def test_no_args(self, tmp_path: Path) -> None:
        _, script = _setup_workspace(tmp_path, [])
        result = _run(script, cwd=tmp_path)
        assert result.returncode == 1
        assert "Usage:" in _strip_ansi(result.stderr)


class TestMaxJobsFlag:
    """The -j flag limits parallel jobs (verify no crash)."""

    def test_j_flag(self, tmp_path: Path) -> None:
        target, script = _setup_workspace(tmp_path, ["a", "b", "c", "d"])
        result = _run(script, "-j", "2", str(target), cwd=tmp_path)
        assert result.returncode == 0
        assert "All 4 jobs completed" in _strip_ansi(result.stdout)


class TestRecursiveDiscovery:
    """Simulation directories are found recursively via step5_production.tpr."""

    def test_nested_dirs_discovered(self, tmp_path: Path) -> None:
        """Directories with .tpr files at any depth are processed."""
        target, script = _setup_workspace(
            tmp_path,
            ["level1/level2/sim1", "level1/level2/sim2", "top_sim"],
        )
        result = _run(script, str(target), cwd=tmp_path)
        assert result.returncode == 0
        assert "All 3 jobs completed" in _strip_ansi(result.stdout)

    def test_intermediate_dirs_without_tpr_skipped(self, tmp_path: Path) -> None:
        """Intermediate directories that have no .tpr are not processed."""
        target, script = _setup_workspace(
            tmp_path,
            ["parent/child"],
            with_tpr=False,
        )
        # Only the leaf has the TPR
        (target / "parent" / "child" / f"{PRODUCTION}.tpr").write_text("")
        result = _run(script, str(target), cwd=tmp_path)
        assert result.returncode == 0
        assert "All 1 jobs completed" in _strip_ansi(result.stdout)

    def test_labels_show_relative_path(self, tmp_path: Path) -> None:
        """Output labels show path relative to target, not just basename."""
        target, script = _setup_workspace(tmp_path, ["a/b/c"])
        result = _run(script, str(target), cwd=tmp_path)
        out = _strip_ansi(result.stdout)
        assert "a/b/c" in out
