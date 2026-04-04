"""Tests for scripts/gromacs/runtime/restart.sh."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

RESTART_SH = Path(__file__).resolve().parents[4] / "scripts" / "gromacs" / "runtime" / "restart.sh"

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _env_with_shims(slurm_env: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{slurm_env['_bin_dir']}:{env['PATH']}"
    return env


def _run(
    slurm_env: dict[str, Path],
    *args: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RESTART_SH), *args],
        capture_output=True,
        text=True,
        env=_env_with_shims(slurm_env),
        cwd=str(cwd),
        check=False,
    )


def _create_job_files(
    base: Path,
    subdir: str,
    job_id: str,
    *,
    out: bool = True,
    err: bool = True,
    sbatch: bool = True,
) -> Path:
    """Create mdrun job files in base/subdir and return the subdir path."""
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    if out:
        (d / f"mdrun_{job_id}.out").write_text("")
    if err:
        (d / f"mdrun_{job_id}.err").write_text("")
    if sbatch:
        (d / "mdrun.sbatch").write_text("#!/bin/bash\necho hello\n")
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSuccessfulRestart:
    """Single job ID with all required files triggers sbatch."""

    def test_sbatch_called_with_correct_args(
        self, slurm_env: dict[str, Path], tmp_path: Path
    ) -> None:
        _create_job_files(tmp_path, "subdir", "12345")

        result = _run(slurm_env, "12345", cwd=tmp_path)

        assert result.returncode == 0

        plain = _strip_ansi(result.stdout)
        assert "12345" in plain
        assert "subdir" in plain

        sbatch_log = slurm_env["sbatch"].read_text().strip()
        assert sbatch_log, "sbatch should have been called"
        # The script calls: sbatch --chdir <TARGET_DIR> <SBATCH_FILE>
        assert "--chdir" in sbatch_log
        assert "subdir" in sbatch_log
        assert "mdrun.sbatch" in sbatch_log


class TestMultipleJobIds:
    """Multiple job IDs each get resubmitted."""

    def test_both_jobs_resubmitted(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        _create_job_files(tmp_path, "sim1", "12345")
        _create_job_files(tmp_path, "sim2", "67890")

        result = _run(slurm_env, "12345", "67890", cwd=tmp_path)

        assert result.returncode == 0

        sbatch_log = slurm_env["sbatch"].read_text().strip()
        lines = sbatch_log.splitlines()
        assert len(lines) == 2, f"Expected 2 sbatch calls, got {len(lines)}: {lines}"

        # Each call should reference its respective directory
        all_log = sbatch_log
        assert "sim1" in all_log
        assert "sim2" in all_log


class TestNoArgs:
    """Running with no arguments prints usage and exits 1."""

    def test_usage_on_no_args(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        result = _run(slurm_env, cwd=tmp_path)

        assert result.returncode == 1
        assert "Usage" in result.stderr


class TestNonIntegerJobId:
    """Non-integer job ID produces an error."""

    def test_rejects_non_integer(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        result = _run(slurm_env, "abc", cwd=tmp_path)

        assert result.returncode == 1
        plain = _strip_ansi(result.stderr)
        assert "must be an integer" in plain


class TestNoMatchingFiles:
    """Job ID with no matching files produces an error."""

    def test_no_files_found(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        result = _run(slurm_env, "99999", cwd=tmp_path)

        assert result.returncode == 1
        plain = _strip_ansi(result.stderr)
        assert "no files matching" in plain


class TestMissingErrOrOut:
    """Missing .err file (only .out present) produces an error."""

    def test_missing_err_file(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        _create_job_files(tmp_path, "subdir", "12345", err=False)

        result = _run(slurm_env, "12345", cwd=tmp_path)

        assert result.returncode == 1
        plain = _strip_ansi(result.stderr)
        assert "expected both" in plain

    def test_missing_out_file(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        _create_job_files(tmp_path, "subdir", "12345", out=False)

        result = _run(slurm_env, "12345", cwd=tmp_path)

        assert result.returncode == 1
        plain = _strip_ansi(result.stderr)
        assert "expected both" in plain


class TestMissingSbatch:
    """Present .out and .err but missing mdrun.sbatch produces an error."""

    def test_missing_sbatch_file(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        _create_job_files(tmp_path, "subdir", "12345", sbatch=False)

        result = _run(slurm_env, "12345", cwd=tmp_path)

        assert result.returncode == 1
        plain = _strip_ansi(result.stderr)
        assert "not found" in plain


class TestFilesInMultipleDirs:
    """Job files split across directories produces an error."""

    def test_multiple_directories_error(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "mdrun_12345.out").write_text("")
        (dir2 / "mdrun_12345.err").write_text("")

        result = _run(slurm_env, "12345", cwd=tmp_path)

        assert result.returncode == 1
        plain = _strip_ansi(result.stderr)
        assert "multiple directories" in plain
