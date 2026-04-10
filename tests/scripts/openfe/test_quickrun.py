"""Tests for scripts/openfe/quickrun.sh."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

QUICKRUN_SH = (
    Path(__file__).resolve().parents[3] / "scripts" / "openfe" / "quickrun" / "quickrun.sh"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        ["bash", str(QUICKRUN_SH), *args],
        capture_output=True,
        text=True,
        env=_env_with_shims(slurm_env),
        cwd=str(cwd),
        check=False,
    )


def _sbatch_lines(slurm_env: dict[str, Path]) -> list[str]:
    """Return non-empty lines from the sbatch log file."""
    return [line for line in slurm_env["sbatch"].read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleTransformation:
    """Single .json file in transformations/, default repeats=1."""

    def test_sbatch_called_once(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        result = _run(slurm_env, cwd=tmp_path)
        assert result.returncode == 0

        lines = _sbatch_lines(slurm_env)
        assert len(lines) == 1

    def test_sbatch_args_contain_array_flag(
        self, slurm_env: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        _run(slurm_env, cwd=tmp_path)

        line = _sbatch_lines(slurm_env)[0]
        assert "--array=0-0" in line

    def test_sbatch_args_contain_json_path(
        self, slurm_env: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        _run(slurm_env, cwd=tmp_path)

        line = _sbatch_lines(slurm_env)[0]
        assert "rbfe_A_B.json" in line

    def test_sbatch_args_contain_output_flag(
        self, slurm_env: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        _run(slurm_env, cwd=tmp_path)

        line = _sbatch_lines(slurm_env)[0]
        assert "-o" in line
        assert "results" in line

    def test_results_dir_created(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        _run(slurm_env, cwd=tmp_path)

        assert (tmp_path / "results").is_dir()

    def test_logs_dir_created(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        _run(slurm_env, cwd=tmp_path)

        assert (tmp_path / "logs").is_dir()


class TestMultipleTransformations:
    """Multiple .json files trigger one sbatch call each."""

    def test_three_json_files_trigger_three_calls(
        self, slurm_env: dict[str, Path], tmp_path: Path
    ) -> None:
        (tmp_path / "transformations").mkdir()
        for name in ("rbfe_A_B.json", "rbfe_C_D.json", "rbfe_E_F.json"):
            (tmp_path / "transformations" / name).write_text("{}")

        result = _run(slurm_env, cwd=tmp_path)
        assert result.returncode == 0

        lines = _sbatch_lines(slurm_env)
        assert len(lines) == 3


class TestRepeatsFlag:
    """Short -r flag sets the SLURM array range."""

    def test_repeats_short_flag(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        result = _run(slurm_env, "-r", "3", cwd=tmp_path)
        assert result.returncode == 0

        line = _sbatch_lines(slurm_env)[0]
        assert "--array=0-2" in line


class TestRepeatsLongFlag:
    """Long --repeats flag sets the SLURM array range."""

    def test_repeats_long_flag(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "rbfe_A_B.json").write_text("{}")

        result = _run(slurm_env, "--repeats", "5", cwd=tmp_path)
        assert result.returncode == 0

        line = _sbatch_lines(slurm_env)[0]
        assert "--array=0-4" in line


class TestMissingTransformationsDir:
    """No transformations/ directory -> exit 1 with error message."""

    def test_exit_code(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        result = _run(slurm_env, cwd=tmp_path)
        assert result.returncode == 1

    def test_error_message(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        result = _run(slurm_env, cwd=tmp_path)
        assert "Missing transformations dir" in result.stderr


class TestNoJsonFiles:
    """transformations/ exists but contains no .json files -> exit 1."""

    def test_exit_code(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()

        result = _run(slurm_env, cwd=tmp_path)
        assert result.returncode == 1

    def test_error_message(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()

        result = _run(slurm_env, cwd=tmp_path)
        assert "No .json files found" in result.stderr


class TestUnknownOption:
    """Unknown option -> exit 1 with error message."""

    def test_exit_code(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()

        result = _run(slurm_env, "--foo", cwd=tmp_path)
        assert result.returncode == 1

    def test_error_message(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()

        result = _run(slurm_env, "--foo", cwd=tmp_path)
        assert "unknown option" in result.stderr


class TestNonJsonFilesIgnored:
    """Non-.json files in transformations/ are skipped by the glob."""

    def test_only_json_triggers_sbatch(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        (tmp_path / "transformations").mkdir()
        (tmp_path / "transformations" / "readme.txt").write_text("not a transformation")
        (tmp_path / "transformations" / "data.json").write_text("{}")

        result = _run(slurm_env, cwd=tmp_path)
        assert result.returncode == 0

        lines = _sbatch_lines(slurm_env)
        assert len(lines) == 1
        assert "data.json" in lines[0]
