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


def _ensure_replica_dir(results_dir: Path, tname: str, replica_id: int) -> Path:
    """Create an empty replica directory (no result JSON) for auto-detect."""
    replica_dir = results_dir / tname / f"replica_{replica_id}"
    replica_dir.mkdir(parents=True, exist_ok=True)
    return replica_dir


def _run(
    slurm_env: dict[str, Path],
    *extra_args: str,
    root: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = ["bash", str(CHECK_STATUS), "-r", str(root)]
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

        result = _run(slurm_env, root=openfe_workspace["root"])
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

        result = _run(slurm_env, root=openfe_workspace["root"])
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
        _ensure_replica_dir(openfe_workspace["results_dir"], "rbfe_A_complex_B_complex", 0)
        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert "incomplete" in rows[0]["info"]


class TestActiveJob:
    """Replica with an active Slurm job shows active with job info and progress."""

    def _setup_active(self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]) -> Path:
        root_abs = openfe_workspace["root"].resolve()
        slurm_env["squeue"].write_text(f"100|0|100_0|RUNNING|{root_abs}\n")
        slurm_env["scontrol"].write_text(
            "   SubmitLine=sbatch --array=0 quickrun.sbatch "
            f"{root_abs}/transformations/rbfe_A_complex_B_complex.json "
            f"-o {root_abs}/results\n"
        )
        return root_abs

    def test_active_job_in_squeue(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        self._setup_active(slurm_env, openfe_workspace)
        _ensure_replica_dir(openfe_workspace["results_dir"], "rbfe_A_complex_B_complex", 0)
        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "active"
        assert "100_0" in rows[0]["info"]
        assert "RUNNING" in rows[0]["info"]

    def test_no_yaml_shows_zero_percent(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        self._setup_active(slurm_env, openfe_workspace)
        _ensure_replica_dir(openfe_workspace["results_dir"], "rbfe_A_complex_B_complex", 0)
        result = _run(slurm_env, root=openfe_workspace["root"])

        rows = _parse_rows(result.stdout)
        assert "0%" in rows[0]["info"]

    def test_yaml_progress(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        root_abs = self._setup_active(slurm_env, openfe_workspace)

        # Create a simulation_real_time_analysis.yaml in the replica dir.
        tname = "rbfe_A_complex_B_complex"
        shared_dir = (
            root_abs
            / "results"
            / tname
            / "replica_0"
            / "shared_HybridTopologyMultiStateSimulationUnit-abc_attempt_0"
        )
        shared_dir.mkdir(parents=True)
        (shared_dir / "simulation_real_time_analysis.yaml").write_text(
            "- iteration: 100\n"
            "  percent_complete: 12.5\n"
            "  timing_data:\n"
            "    estimated_time_remaining: 2 days, 3:00:00\n"
            "- iteration: 200\n"
            "  percent_complete: 25.0\n"
            "  timing_data:\n"
            "    estimated_time_remaining: 1 day, 12:00:00\n"
        )

        result = _run(slurm_env, root=openfe_workspace["root"])
        rows = _parse_rows(result.stdout)
        assert "25.0%" in rows[0]["info"]
        assert "ETA:" in rows[0]["info"]
        assert "1 day" in rows[0]["info"]


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

        _ensure_replica_dir(openfe_workspace["results_dir"], "rbfe_A_complex_B_complex", 0)
        result = _run(slurm_env, root=openfe_workspace["root"])
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
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 2)

        result = _run(slurm_env, root=openfe_workspace["root"])
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
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 2)

        result = _run(slurm_env, "-R", root=openfe_workspace["root"])
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
        _ensure_replica_dir(openfe_workspace["results_dir"], "rbfe_A_complex_B_complex", 0)
        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        sbatch_log = slurm_env["sbatch"].read_text().strip()
        assert sbatch_log == "", "sbatch should not be called without -R"


class TestNoTransformationsDir:
    """Missing transformations directory exits with error."""

    def test_missing_transforms_dir(self, slurm_env: dict[str, Path], tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        result = _run(slurm_env, root=empty)
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


class TestPreemptionDetection:
    """Preempted jobs detected via sacct are treated as active (REQUEUING).

    When SLURM preempts and requeues an array task, the task temporarily
    vanishes from squeue.  The sacct-based Phase 2 in build_active_jobs
    detects this and injects a synthetic REQUEUING entry so the replica
    is classified as "active" instead of "failed".
    """

    @staticmethod
    def _setup_scontrol(
        slurm_env: dict[str, Path], root_abs: Path, tname: str = "rbfe_A_complex_B_complex"
    ) -> None:
        slurm_env["scontrol"].write_text(
            "   SubmitLine=sbatch --array=0 quickrun.sbatch "
            f"{root_abs}/transformations/{tname}.json "
            f"-o {root_abs}/results\n"
        )

    def test_preempted_not_in_squeue_shows_active(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        """A preempted job absent from squeue is classified as active/REQUEUING."""
        root_abs = openfe_workspace["root"].resolve()
        tname = "rbfe_A_complex_B_complex"
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 0)

        # squeue is empty -- the job vanished during requeue transition.
        slurm_env["squeue"].write_text("")
        # sacct reports the job was recently preempted.
        slurm_env["sacct"].write_text(f"100_0|PREEMPTED|{root_abs}\n")
        self._setup_scontrol(slurm_env, root_abs)

        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "active"
        assert "REQUEUING" in rows[0]["info"]

    def test_preempted_already_in_squeue_no_duplicate(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        """A preempted job that reappeared in squeue is not double-counted."""
        root_abs = openfe_workspace["root"].resolve()
        tname = "rbfe_A_complex_B_complex"
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 0)

        # Job is back in squeue after requeue.
        slurm_env["squeue"].write_text(f"100|0|100_0|RUNNING|{root_abs}\n")
        # sacct still reports the earlier preemption event.
        slurm_env["sacct"].write_text(f"100_0|PREEMPTED|{root_abs}\n")
        self._setup_scontrol(slurm_env, root_abs)

        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "active"
        # Should show RUNNING from squeue, not REQUEUING.
        assert "RUNNING" in rows[0]["info"]

    def test_sacct_step_records_are_skipped(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        """Sacct step records (*.batch, *.extern) are filtered out."""
        root_abs = openfe_workspace["root"].resolve()
        tname = "rbfe_A_complex_B_complex"
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 0)

        slurm_env["squeue"].write_text("")
        # Only sub-step records -- should all be filtered.
        slurm_env["sacct"].write_text(
            f"100_0.batch|PREEMPTED|{root_abs}\n100_0.extern|PREEMPTED|{root_abs}\n"
        )

        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"

    def test_sacct_filters_other_workdir(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        """Preempted jobs from a different project workdir are ignored."""
        tname = "rbfe_A_complex_B_complex"
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 0)

        slurm_env["squeue"].write_text("")
        # sacct reports a preemption from a different workdir.
        slurm_env["sacct"].write_text("100_0|PREEMPTED|/some/other/project\n")

        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"

    def test_preempted_does_not_trigger_restart(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        """A preempted-but-requeuing replica is NOT resubmitted with -R."""
        root_abs = openfe_workspace["root"].resolve()
        tname = "rbfe_A_complex_B_complex"
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 0)

        slurm_env["squeue"].write_text("")
        slurm_env["sacct"].write_text(f"100_0|PREEMPTED|{root_abs}\n")
        self._setup_scontrol(slurm_env, root_abs)

        result = _run(slurm_env, "-R", root=openfe_workspace["root"])
        assert result.returncode == 0

        sbatch_log = slurm_env["sbatch"].read_text().strip()
        assert sbatch_log == "", "sbatch must not be called for requeuing replicas"

    def test_multiple_preemptions_deduplicated(
        self, slurm_env: dict[str, Path], openfe_workspace: dict[str, Path]
    ) -> None:
        """Multiple sacct PREEMPTED records for the same task are deduplicated."""
        root_abs = openfe_workspace["root"].resolve()
        tname = "rbfe_A_complex_B_complex"
        _ensure_replica_dir(openfe_workspace["results_dir"], tname, 0)

        slurm_env["squeue"].write_text("")
        # Same task preempted three times -- should be deduplicated by sort -u.
        slurm_env["sacct"].write_text(
            f"100_0|PREEMPTED|{root_abs}\n100_0|PREEMPTED|{root_abs}\n100_0|PREEMPTED|{root_abs}\n"
        )
        self._setup_scontrol(slurm_env, root_abs)

        result = _run(slurm_env, root=openfe_workspace["root"])
        assert result.returncode == 0

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "active"
        assert "REQUEUING" in rows[0]["info"]
