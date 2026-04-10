"""Tests for scripts/gromacs/runtime/check_status.sh."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

CHECK_STATUS = (
    Path(__file__).resolve().parents[4] / "scripts" / "gromacs" / "runtime" / "check_status.sh"
)

pytestmark = pytest.mark.skipif(
    not shutil.which("parallel"),
    reason="GNU parallel not found",
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# ---------------------------------------------------------------------------
# Realistic gmx dump output templates
# ---------------------------------------------------------------------------

# gmx dump -s output: nsteps=50000000, dt=0.002, init_t=0
# target = 0 + 50000000 * 0.002 = 100000 ps = 100 ns
GMX_DUMP_S_DEFAULT = textwrap.dedent("""\
   nsteps                         = 50000000
   init_t                         = 0
   delta_t                        = 0.002
""")

# gmx dump -cp output: checkpoint at 100000 ps = 100 ns (completed)
GMX_DUMP_CP_COMPLETED = textwrap.dedent("""\
   t                              = 100000
""")

# gmx dump -cp output: checkpoint at 50000 ps = 50 ns (incomplete)
GMX_DUMP_CP_INCOMPLETE = textwrap.dedent("""\
   t                              = 50000
""")

# gmx dump -s output with no nsteps/dt (unparseable)
GMX_DUMP_S_UNPARSEABLE = textwrap.dedent("""\
   some_random_field              = 42
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gmx_env(tmp_path: Path, slurm_env: dict[str, Path]) -> dict[str, Path]:
    """Extend slurm_env with a gmx shim and parallel shim.

    Returns a dict with all slurm_env keys plus:
    - ``gmx_dump_s``: write gmx dump -s response content here.
    - ``gmx_dump_cp``: write gmx dump -cp response content here.
    """
    bin_dir = slurm_env["_bin_dir"]

    # Response files for gmx dump
    gmx_dump_s_resp = tmp_path / "gmx_dump_s_response.txt"
    gmx_dump_s_resp.write_text("")
    gmx_dump_cp_resp = tmp_path / "gmx_dump_cp_response.txt"
    gmx_dump_cp_resp.write_text("")

    # gmx shim: routes based on the flag after "dump"
    gmx_shim = bin_dir / "gmx"
    gmx_shim.write_text(
        textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # Shim for gmx: only handles "dump -s" and "dump -cp"
        if [[ "$1" == "dump" ]]; then
            shift
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    -s)
                        cat "{gmx_dump_s_resp}"
                        exit 0
                        ;;
                    -cp)
                        cat "{gmx_dump_cp_resp}"
                        exit 0
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
        fi
        exit 1
        """)
    )
    _make_executable(gmx_shim)

    result = dict(slurm_env)
    result["gmx_dump_s"] = gmx_dump_s_resp
    result["gmx_dump_cp"] = gmx_dump_cp_resp
    return result


def _make_sim_dir(
    root: Path,
    name: str,
    *,
    tpr: bool = True,
    cpt: bool = False,
) -> Path:
    """Create a simulation directory with optional TPR and CPT files."""
    sim = root / name
    sim.mkdir(parents=True, exist_ok=True)
    if tpr:
        (sim / "step5_production.tpr").touch()
    if cpt:
        (sim / "step5_production.cpt").touch()
    return sim


def _run(
    gmx_env: dict[str, Path],
    *extra_args: str,
    root: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = ["bash", str(CHECK_STATUS), "-r", str(root), "-j", "1"]
    cmd += list(extra_args)
    env = os.environ.copy()
    env["PATH"] = f"{gmx_env['_bin_dir']}:{env['PATH']}"
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestCompleted:
    """Simulation with checkpoint time >= target time shows completed."""

    def test_completed_status(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        _make_sim_dir(root, "run1", tpr=True, cpt=True)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_COMPLETED)
        gmx_env["squeue"].write_text("")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        assert "100.000" in rows[0]["progress"]
        assert "checkpoint reached target time" in rows[0]["info"]


class TestActive:
    """Incomplete simulation with matching squeue job shows active."""

    def test_active_status(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        sim = _make_sim_dir(root, "run1", tpr=True, cpt=True)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_INCOMPLETE)

        # squeue output: pipe-delimited jobid|state|workdir
        sim_abs = sim.resolve()
        gmx_env["squeue"].write_text(f"12345|RUNNING|{sim_abs}\n")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "active"
        assert "50.000" in rows[0]["progress"]
        assert "12345" in rows[0]["info"]
        assert "RUNNING" in rows[0]["info"]


class TestFailed:
    """Incomplete simulation with no matching squeue job shows failed."""

    def test_failed_status(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        _make_sim_dir(root, "run1", tpr=True, cpt=True)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_INCOMPLETE)
        gmx_env["squeue"].write_text("")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert "50.000" in rows[0]["progress"]
        assert "incomplete" in rows[0]["info"]


class TestMissingTPR:
    """Simulation directory without step5_production.tpr is not found."""

    def test_no_tpr_means_no_sim_dir(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        # Create a dir but without the TPR file
        _make_sim_dir(root, "run1", tpr=False, cpt=False)

        gmx_env["squeue"].write_text("")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # No sim dirs found, so only the header line
        rows = _parse_rows(result.stdout)
        assert len(rows) == 0


class TestUnparseableTPR:
    """gmx dump returning no nsteps/dt results in error status."""

    def test_unparseable_tpr(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        _make_sim_dir(root, "run1", tpr=True, cpt=True)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_UNPARSEABLE)
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_INCOMPLETE)
        gmx_env["squeue"].write_text("")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert "could not parse target time" in rows[0]["info"]


class TestMultipleJobs:
    """Two squeue jobs matching the same workdir shows error."""

    def test_multiple_jobs_error(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        sim = _make_sim_dir(root, "run1", tpr=True, cpt=True)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_INCOMPLETE)

        sim_abs = sim.resolve()
        gmx_env["squeue"].write_text(f"111|RUNNING|{sim_abs}\n222|PENDING|{sim_abs}\n")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert "multiple active jobs" in rows[0]["info"]


class TestTargetOverride:
    """-t flag overrides TPR-derived target time."""

    def test_target_override_completed(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        _make_sim_dir(root, "run1", tpr=True, cpt=True)

        # The TPR says 100 ns target, but we override to 50 ns
        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        # Checkpoint at 50 ns
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_INCOMPLETE)
        gmx_env["squeue"].write_text("")

        # Override target to 50 ns -- checkpoint is at 50 ns, so completed
        result = _run(gmx_env, "-t", "50", root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        # Progress should show 50.000/50 (target_ns is the raw override value)
        assert "50.000/50" in rows[0]["progress"]

    def test_target_override_failed(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        _make_sim_dir(root, "run1", tpr=True, cpt=True)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        # Checkpoint at 50 ns
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_INCOMPLETE)
        gmx_env["squeue"].write_text("")

        # Override target to 200 ns -- checkpoint at 50 ns, not done, no job
        result = _run(gmx_env, "-t", "200", root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert "50.000/200" in rows[0]["progress"]


class TestHelp:
    """-h flag prints usage and exits 0."""

    def test_help_flag(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        root.mkdir()

        result = _run(gmx_env, "-h", root=root)
        assert result.returncode == 0
        assert "Usage:" in result.stdout


class TestInvalidOption:
    """Invalid option exits with error code 2."""

    def test_invalid_option_exits_2(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        root.mkdir()

        result = _run(gmx_env, "-Z", root=root)
        assert result.returncode == 2
        assert "unknown option" in result.stderr


class TestNoSimDirs:
    """No step5_production.tpr found produces only the header row."""

    def test_empty_root(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        root.mkdir()

        gmx_env["squeue"].write_text("")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 0

        # But we should still see the header
        clean = _strip_ansi(result.stdout).strip()
        assert clean.startswith("directory\tstatus\tprogress\tinfo")


class TestNoCpt:
    """Simulation with TPR but no checkpoint file shows failed (no current time)."""

    def test_no_checkpoint(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        _make_sim_dir(root, "run1", tpr=True, cpt=False)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        gmx_env["squeue"].write_text("")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 1
        # No checkpoint means no current_ps, so cannot be completed.
        # No squeue match either, so failed.
        assert rows[0]["status"] == "failed"
        assert "NA/" in rows[0]["progress"]


class TestMultipleSimDirs:
    """Multiple simulation directories are all reported."""

    def test_multiple_dirs(self, gmx_env: dict[str, Path], tmp_path: Path) -> None:
        root = tmp_path / "sims"
        _make_sim_dir(root, "run1", tpr=True, cpt=True)
        _make_sim_dir(root, "run2", tpr=True, cpt=True)
        _make_sim_dir(root, "run3", tpr=True, cpt=True)

        gmx_env["gmx_dump_s"].write_text(GMX_DUMP_S_DEFAULT)
        gmx_env["gmx_dump_cp"].write_text(GMX_DUMP_CP_COMPLETED)
        gmx_env["squeue"].write_text("")

        result = _run(gmx_env, root=root)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        rows = _parse_rows(result.stdout)
        assert len(rows) == 3
        statuses = {r["status"] for r in rows}
        assert statuses == {"completed"}
