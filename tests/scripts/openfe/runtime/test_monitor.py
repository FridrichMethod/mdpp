"""Tests for scripts/openfe/runtime/monitor.sbatch.

Uses a mock check_status.sh that returns canned TSV output, so no
Slurm/parallel/jq dependencies are needed.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

MONITOR_SBATCH = (
    Path(__file__).resolve().parents[4] / "scripts" / "openfe" / "runtime" / "monitor.sbatch"
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


# ---------------------------------------------------------------------------
# Canned check_status.sh outputs
# ---------------------------------------------------------------------------

_STATUS_ALL_COMPLETED = textwrap.dedent(
    """\
    directory\tstatus\treplica\tinfo
    /res/rbfe_A_B/replica_0\tcompleted\treplica_0\tddG = 1.4 +/- 0.1 kcal/mol
    /res/rbfe_A_B/replica_1\tcompleted\treplica_1\tddG = 1.5 +/- 0.2 kcal/mol
    """
)

_STATUS_WITH_FAILED = textwrap.dedent(
    """\
    directory\tstatus\treplica\tinfo
    /res/rbfe_A_B/replica_0\tcompleted\treplica_0\tddG = 1.4 +/- 0.1 kcal/mol
    /res/rbfe_A_B/replica_1\tfailed\treplica_1\tincomplete and no matching active job
    /res/rbfe_A_B/replica_2\tactive\treplica_2\tjob in squeue: 123 (100_2):RUNNING
    """
)

_STATUS_WITH_FAILED_AND_RESTART = (
    _STATUS_WITH_FAILED
    + "\n"
    + textwrap.dedent(
        """\
        Resubmitting rbfe_A_B replicas [1]
        Submitted batch job 99999
        """
    )
)

_STATUS_WITH_ERROR = textwrap.dedent(
    """\
    directory\tstatus\treplica\tinfo
    /res/rbfe_A_B/replica_0\terror\treplica_0\tmultiple matching jobs: 1:R,2:R
    /res/rbfe_A_B/replica_1\tcompleted\treplica_1\tddG = 1.4 +/- 0.1 kcal/mol
    """
)

_STATUS_MIXED = textwrap.dedent(
    """\
    directory\tstatus\treplica\tinfo
    /res/rbfe_A_B/replica_0\tcompleted\treplica_0\tddG = 1.4 +/- 0.1 kcal/mol
    /res/rbfe_A_B/replica_1\tfailed\treplica_1\tincomplete and no matching active job
    /res/rbfe_C_D/replica_0\tfailed\treplica_0\tresult JSON has null estimate/uncertainty
    /res/rbfe_C_D/replica_1\tactive\treplica_1\tjob in squeue: 456 (400_1):RUNNING
    """
)

_STATUS_MIXED_WITH_RESTART = (
    _STATUS_MIXED
    + "\n"
    + textwrap.dedent(
        """\
        Resubmitting rbfe_A_B replicas [1]
        Submitted batch job 88888
        Resubmitting rbfe_C_D replicas [0]
        Submitted batch job 99999
        """
    )
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def monitor_env(tmp_path: Path) -> dict[str, Path]:
    """Set up a mock environment for monitor.sbatch."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()

    # Mock check_status.sh: prints canned response from file.
    cs_response = tmp_path / "check_status_response.txt"
    cs_response.write_text("")
    mock_cs = scripts_dir / "check_status.sh"
    mock_cs.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            resp="{cs_response}"
            if [[ -f "$resp" ]]; then cat "$resp"; fi
            exit 0
            """
        )
    )
    _make_executable(mock_cs)

    # Copy real monitor.sbatch so SCRIPTS_DIR resolves to our mock dir.
    monitor_copy = scripts_dir / "monitor.sbatch"
    shutil.copy2(MONITOR_SBATCH, monitor_copy)

    # Shim bin dir: sbatch and mail.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    sbatch_log = tmp_path / "sbatch_log.txt"
    sbatch_log.write_text("")
    sbatch_shim = bin_dir / "sbatch"
    sbatch_shim.write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{sbatch_log}"\necho "Submitted batch job 77777"\n'
    )
    _make_executable(sbatch_shim)

    mail_log = tmp_path / "mail_log.txt"
    mail_log.write_text("")
    mail_shim = bin_dir / "mail"
    mail_shim.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            subject=""
            while [[ $# -gt 1 ]]; do
                case "$1" in -s) subject="$2"; shift 2 ;; *) shift ;; esac
            done
            echo "SUBJECT: $subject" >> "{mail_log}"
            cat >> "{mail_log}"
            """
        )
    )
    _make_executable(mail_shim)

    # Fake project directory.
    project = tmp_path / "project"
    (project / "transformations").mkdir(parents=True)
    (project / "results").mkdir(parents=True)

    return {
        "monitor_sbatch": monitor_copy,
        "cs_response": cs_response,
        "sbatch_log": sbatch_log,
        "mail_log": mail_log,
        "_bin_dir": bin_dir,
        "project": project,
    }


def _run(
    env: dict[str, Path],
    *extra_args: str,
    dirs: list[Path] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = ["bash", str(env["monitor_sbatch"])]
    for d in dirs if dirs is not None else [env["project"]]:
        cmd += ["-d", str(d)]
    if dry_run:
        cmd.append("-n")
    cmd += list(extra_args)

    run_env = os.environ.copy()
    run_env["PATH"] = f"{env['_bin_dir']}:{run_env['PATH']}"
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env, check=False)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestStatusCounts:
    """Report shows correct status counts per directory."""

    def test_all_completed(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        result = _run(monitor_env, dry_run=True)
        out = _strip_ansi(result.stdout)
        assert "2/2 completed" in out
        assert "0 active" in out
        assert "0 failed" in out

    def test_mixed_statuses(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        result = _run(monitor_env, dry_run=True)
        out = _strip_ansi(result.stdout)
        assert "1/3 completed" in out
        assert "1 active" in out
        assert "1 failed" in out

    def test_two_directories(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_MIXED_WITH_RESTART)
        project2 = monitor_env["project"].parent / "project2"
        (project2 / "transformations").mkdir(parents=True)
        (project2 / "results").mkdir(parents=True)

        result = _run(monitor_env, dirs=[monitor_env["project"], project2])
        out = _strip_ansi(result.stdout)
        assert out.count("1/4 completed") == 2


class TestRestartReport:
    """Failed replicas are restarted and reported."""

    def test_restart_lines_in_output(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        result = _run(monitor_env)
        out = _strip_ansi(result.stdout)
        assert "Resubmitting rbfe_A_B" in out
        assert "Submitted batch job 99999" in out

    def test_restart_triggers_self_resubmit(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env)
        assert "monitor.sbatch" in monitor_env["sbatch_log"].read_text()

    def test_failed_details_in_output(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        result = _run(monitor_env)
        out = _strip_ansi(result.stdout)
        assert "rbfe_A_B" in out
        assert "replicas [1]" in out


class TestErrorReport:
    """Error entries are reported but not restarted."""

    def test_error_in_report(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_ERROR)
        result = _run(monitor_env, dry_run=True)
        out = _strip_ansi(result.stdout)
        assert "1 error" in out
        assert "multiple matching jobs" in out


class TestDryRun:
    """Dry run: no restarts, no sbatch, no email."""

    def test_shows_would_restart(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        result = _run(monitor_env, dry_run=True)
        assert "[DRY RUN] would restart" in _strip_ansi(result.stdout)

    def test_no_sbatch(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        _run(monitor_env, dry_run=True)
        assert monitor_env["sbatch_log"].read_text().strip() == ""

    def test_no_email(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        _run(monitor_env, dry_run=True)
        assert monitor_env["mail_log"].read_text().strip() == ""


class TestAllDone:
    """All completed: email, no resubmit."""

    def test_no_resubmit(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        _run(monitor_env)
        assert "monitor.sbatch" not in monitor_env["sbatch_log"].read_text()

    def test_email_subject(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        _run(monitor_env)
        assert "All" in monitor_env["mail_log"].read_text()
        assert "completed" in monitor_env["mail_log"].read_text()


class TestSelfResubmit:
    """Self-resubmission with correct interval and --chdir."""

    def test_interval_flag(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env, "-i", "2")
        sbatch_log = monitor_env["sbatch_log"].read_text()
        assert "now+2hour" in sbatch_log
        assert "--dependency=singleton" in sbatch_log

    def test_chdir_to_scripts_dir(self, monitor_env: dict[str, Path]) -> None:
        """Self-resubmit must pass --chdir so the resubmitted job runs in the scripts dir."""
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env)
        sbatch_log = monitor_env["sbatch_log"].read_text()
        scripts_dir = str(monitor_env["monitor_sbatch"].parent)
        assert f"--chdir {scripts_dir}" in sbatch_log


class TestEmailContent:
    """Email subject reflects status summary."""

    def test_subject_has_counts(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env)
        mail = monitor_env["mail_log"].read_text()
        assert "1/3 completed" in mail


class TestMissingDirectory:
    """Non-existent project directory is warned about."""

    def test_warning(self, monitor_env: dict[str, Path]) -> None:
        fake = monitor_env["project"].parent / "nonexistent"
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        result = _run(monitor_env, dirs=[fake, monitor_env["project"]])
        assert "does not exist" in _strip_ansi(result.stdout + result.stderr)


class TestSlurmSpoolResolution:
    """SCRIPTS_DIR resolves correctly when SLURM copies the script to a spool dir."""

    def test_slurm_spool_uses_cwd(self, monitor_env: dict[str, Path]) -> None:
        """Under SLURM (SLURM_JOB_ID set), SCRIPTS_DIR comes from pwd, not BASH_SOURCE."""
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)

        # Copy monitor.sbatch to a fake spool directory (different from scripts_dir).
        spool = monitor_env["monitor_sbatch"].parent.parent / "spool"
        spool.mkdir()
        spool_copy = spool / "slurm_script"
        shutil.copy2(monitor_env["monitor_sbatch"], spool_copy)

        scripts_dir = monitor_env["monitor_sbatch"].parent

        # Run from spool path but with cwd=scripts_dir and SLURM_JOB_ID set.
        cmd = ["bash", str(spool_copy), "-d", str(monitor_env["project"]), "-n"]
        run_env = os.environ.copy()
        run_env["PATH"] = f"{monitor_env['_bin_dir']}:{run_env['PATH']}"
        run_env["SLURM_JOB_ID"] = "12345"
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=run_env, cwd=str(scripts_dir), check=False
        )
        assert result.returncode == 0
        assert "2/2 completed" in _strip_ansi(result.stdout)

    def test_spool_without_slurm_id_misresolves(self, monitor_env: dict[str, Path]) -> None:
        """Without SLURM_JOB_ID, BASH_SOURCE resolves to spool — check_status.sh not found."""
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)

        spool = monitor_env["monitor_sbatch"].parent.parent / "spool"
        spool.mkdir(exist_ok=True)
        spool_copy = spool / "slurm_script"
        shutil.copy2(monitor_env["monitor_sbatch"], spool_copy)

        scripts_dir = monitor_env["monitor_sbatch"].parent

        # Run from spool path with cwd=scripts_dir but NO SLURM_JOB_ID.
        # BASH_SOURCE resolves to spool dir, so check_status.sh won't be found.
        cmd = ["bash", str(spool_copy), "-d", str(monitor_env["project"]), "-n"]
        run_env = os.environ.copy()
        run_env["PATH"] = f"{monitor_env['_bin_dir']}:{run_env['PATH']}"
        run_env.pop("SLURM_JOB_ID", None)
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=run_env, cwd=str(scripts_dir), check=False
        )
        # SCRIPTS_DIR resolves to spool, check_status.sh is not there — warning emitted
        assert "check_status.sh failed" in result.stdout + result.stderr
        # Status counts should be empty (nothing processed)
        assert "0/0 completed" in _strip_ansi(result.stdout)


class TestCLI:
    """Argument parsing edge cases."""

    def test_no_dirs(self, monitor_env: dict[str, Path]) -> None:
        result = _run(monitor_env, dirs=[])
        assert result.returncode == 2
        assert "-d DIR is required" in result.stderr

    def test_help_flag(self, monitor_env: dict[str, Path]) -> None:
        result = _run(monitor_env, "-h")
        assert result.returncode == 0
        assert "Usage:" in result.stdout
