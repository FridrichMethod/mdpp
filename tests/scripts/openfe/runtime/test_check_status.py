"""Tests for scripts/openfe/runtime/check_status.sh."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

CHECK_STATUS = (
    Path(__file__).resolve().parents[4] / "scripts" / "openfe" / "runtime" / "check_status.sh"
)

pytestmark = pytest.mark.skipif(
    not shutil.which("parallel") or not shutil.which("jq"),
    reason="GNU parallel and/or jq not found",
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")

_VALID_RESULT = {
    "estimate": {
        "magnitude": 1.435,
        "unit": "kilocalorie_per_mole",
        ":is_custom:": True,
        "pint_unit_registry": "openff_units",
    },
    "uncertainty": {
        "magnitude": 0.12,
        "unit": "kilocalorie_per_mole",
        ":is_custom:": True,
        "pint_unit_registry": "openff_units",
    },
}

_NULL_RESULT: dict[str, object] = {
    "estimate": None,
    "uncertainty": None,
    "protocol_result": {},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _env_with_shims(slurm_env: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{slurm_env['_bin_dir']}:{env['PATH']}"
    return env


def _place_result(
    results_dir: Path,
    tname: str,
    replica_id: int,
    *,
    valid: bool = True,
) -> Path:
    replica_dir = results_dir / tname / f"replica_{replica_id}"
    replica_dir.mkdir(parents=True, exist_ok=True)
    result_json = replica_dir / f"{tname}.json"
    result_json.write_text(json.dumps(_VALID_RESULT if valid else _NULL_RESULT))
    return result_json


def _run(
    slurm_env: dict[str, Path],
    *extra_args: str,
    root: Path,
    replicas: int | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = ["bash", str(CHECK_STATUS), "-r", str(root)]
    if replicas is not None:
        cmd += ["-n", str(replicas)]
    cmd += list(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_env_with_shims(slurm_env),
        check=False,
    )


def _parse_rows(stdout: str) -> list[dict[str, str]]:
    """Parse TSV output into list of dicts (skip header)."""
    lines = _strip_ansi(stdout).strip().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cols = line.split("\t")
        rows.append(dict(zip(header, cols)))
    return rows


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestCompletedReplica:
    """Replica with a valid result JSON shows completed + ddG."""

    def test_valid_result_shows_ddg(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        _place_result(openfe_workspace["results_dir"], "rbfe_A_complex_B_complex", 0, valid=True)

        result = _run(slurm_env, root=openfe_workspace["root"], replicas=1)
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        assert "1.435" in rows[0]["info"]
        assert "0.12" in rows[0]["info"]
        assert "kcal/mol" in rows[0]["info"]


class TestNullEstimate:
    """Replica with null estimate/uncertainty shows failed."""

    def test_null_estimate_is_failed(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        _place_result(openfe_workspace["results_dir"], "rbfe_A_complex_B_complex", 0, valid=False)

        result = _run(slurm_env, root=openfe_workspace["root"], replicas=1)
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert "null estimate/uncertainty" in rows[0]["info"]


class TestNoResultNoJob:
    """Replica with no result JSON and no active job shows failed."""

    def test_missing_result_is_failed(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        result = _run(slurm_env, root=openfe_workspace["root"], replicas=1)
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert "incomplete" in rows[0]["info"]


class TestActiveJob:
    """Replica with an active Slurm job shows active with job info."""

    def test_active_job_in_squeue(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        root_abs = openfe_workspace["root"].resolve()

        slurm_env["squeue"].write_text(f"100|0|100_0|RUNNING|{root_abs}\n")
        slurm_env["scontrol"].write_text(
            "   SubmitLine=sbatch --array=0 quickrun.sbatch "
            f"{root_abs}/transformations/rbfe_A_complex_B_complex.json "
            f"-o {root_abs}/results\n"
        )

        result = _run(slurm_env, root=openfe_workspace["root"], replicas=1)
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "active"
        assert "100_0" in rows[0]["info"]
        assert "RUNNING" in rows[0]["info"]


class TestMultipleJobs:
    """Multiple matching Slurm jobs for one replica shows error."""

    def test_error_on_duplicate_jobs(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        root_abs = openfe_workspace["root"].resolve()

        slurm_env["squeue"].write_text(
            f"100|0|100_0|RUNNING|{root_abs}\n200|0|200_0|PENDING|{root_abs}\n"
        )
        slurm_env["scontrol"].write_text(
            "   SubmitLine=sbatch --array=0 quickrun.sbatch "
            f"{root_abs}/transformations/rbfe_A_complex_B_complex.json "
            f"-o {root_abs}/results\n"
        )

        result = _run(slurm_env, root=openfe_workspace["root"], replicas=1)
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert "multiple matching jobs" in rows[0]["info"]


class TestMixedReplicas:
    """Multiple replicas with different statuses."""

    def test_mixed_statuses(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        tname = "rbfe_A_complex_B_complex"
        _place_result(openfe_workspace["results_dir"], tname, 0, valid=True)
        _place_result(openfe_workspace["results_dir"], tname, 1, valid=False)
        # replica 2: no result, no job

        result = _run(slurm_env, root=openfe_workspace["root"], replicas=3)
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 3

        statuses = {r["replica"]: r["status"] for r in rows}
        assert statuses["replica_0"] == "completed"
        assert statuses["replica_1"] == "failed"
        assert statuses["replica_2"] == "failed"


class TestAutoDetectReplicas:
    """Replica count is auto-detected from existing result directories."""

    def test_auto_detect_from_results_dir(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        tname = "rbfe_A_complex_B_complex"
        _place_result(openfe_workspace["results_dir"], tname, 0, valid=True)
        _place_result(openfe_workspace["results_dir"], tname, 1, valid=True)

        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 2


class TestRestartFlag:
    """The -R flag resubmits failed replicas via sbatch."""

    def test_restart_submits_failed(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        tname = "rbfe_A_complex_B_complex"
        _place_result(openfe_workspace["results_dir"], tname, 0, valid=True)
        _place_result(openfe_workspace["results_dir"], tname, 1, valid=False)
        # replica 2: missing

        result = _run(slurm_env, "-R", root=openfe_workspace["root"], replicas=3)
        assert result.returncode == 0

        sbatch_log = slurm_env["sbatch"].read_text().strip()
        assert sbatch_log, "sbatch should have been called for failed replicas"
        assert tname in sbatch_log

        # Replica 0 completed -- must not appear in the --array value.
        array_val = sbatch_log.split("--array=")[1].split()[0]
        assert "0" not in array_val.split(",")
        assert "1" in array_val.split(",")
        assert "2" in array_val.split(",")

    def test_no_restart_without_flag(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        result = _run(slurm_env, root=openfe_workspace["root"], replicas=1)
        assert result.returncode == 0

        sbatch_log = slurm_env["sbatch"].read_text().strip()
        assert sbatch_log == "", "sbatch should not be called without -R"


class TestNoTransformationsDir:
    """Missing transformations directory exits with error."""

    def test_missing_transforms_dir(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        result = _run(slurm_env, root=empty, replicas=1)
        assert result.returncode != 0
        assert "transformations directory not found" in result.stderr


class TestNoReplicas:
    """No replicas detected exits cleanly."""

    def test_no_replicas_found(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0
        assert "No replicas found" in result.stderr


class TestUsageHelp:
    """The -h flag prints usage and exits 0."""

    def test_help_flag(self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]) -> None:
        result = _run(slurm_env, "-h", root=openfe_workspace["root"])
        assert result.returncode == 0
        assert "Usage:" in result.stdout


class TestInvalidOption:
    """Invalid option exits with error."""

    def test_invalid_option(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        result = _run(slurm_env, "-Z", root=openfe_workspace["root"])
        assert result.returncode == 2
        assert "invalid option" in result.stderr
