"""Tests for scripts/openfe/runtime/monitor.sh.

Uses a mock check_status.sh that returns canned TSV output, so no
Slurm/parallel/jq dependencies are needed.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

MONITOR_SH = Path(__file__).resolve().parents[4] / "scripts" / "openfe" / "runtime" / "monitor.sh"

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
    """Set up a mock environment for monitor.sh.

    Creates:
    - A mock check_status.sh that prints canned output from a response file.
    - A mock sbatch that logs calls.
    - A mock mail that logs calls.
    - A fake project directory with transformations/ and results/.
    """
    # Scripts dir with mock check_status.sh and monitor.sbatch
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()

    # Response file for the mock check_status.sh
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

    # Copy the real monitor.sh into the mock scripts dir.
    # monitor.sh resolves SCRIPTS_DIR from readlink -f of BASH_SOURCE[0],
    # so the copy's SCRIPTS_DIR will point to our mock dir (with mock check_status.sh).
    import shutil

    monitor_copy = scripts_dir / "monitor.sh"
    shutil.copy2(MONITOR_SH, monitor_copy)

    # Mock monitor.sbatch (for self-resubmit)
    mock_sbatch_file = scripts_dir / "monitor.sbatch"
    mock_sbatch_file.write_text("#!/bin/bash\nexit 0\n")
    _make_executable(mock_sbatch_file)

    # Shim bin dir for sbatch and mail
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    sbatch_log = tmp_path / "sbatch_log.txt"
    sbatch_log.write_text("")
    sbatch_shim = bin_dir / "sbatch"
    sbatch_shim.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            echo "$@" >> "{sbatch_log}"
            echo "Submitted batch job 77777"
            """
        )
    )
    _make_executable(sbatch_shim)

    mail_log = tmp_path / "mail_log.txt"
    mail_log.write_text("")
    mail_shim = bin_dir / "mail"
    # mail is invoked as: mail -s "subject" recipient < body
    # Capture the subject (arg after -s) and body (stdin).
    mail_shim.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            subject=""
            while [[ $# -gt 1 ]]; do
                case "$1" in
                    -s) subject="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            echo "SUBJECT: $subject" >> "{mail_log}"
            cat >> "{mail_log}"
            """
        )
    )
    _make_executable(mail_shim)

    # Fake project directory
    project = tmp_path / "project"
    (project / "transformations").mkdir(parents=True)
    (project / "results").mkdir(parents=True)

    # State file
    state_file = tmp_path / "state.txt"

    return {
        "scripts_dir": scripts_dir,
        "monitor_sh": monitor_copy,
        "cs_response": cs_response,
        "sbatch_log": sbatch_log,
        "mail_log": mail_log,
        "_bin_dir": bin_dir,
        "project": project,
        "state_file": state_file,
    }


_SENTINEL: list[Path] = []


def _run(
    env: dict[str, Path],
    *extra_args: str,
    dirs: list[Path] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = ["bash", str(env["monitor_sh"])]
    for d in dirs if dirs is not None else [env["project"]]:
        cmd += ["-d", str(d)]
    cmd += ["-s", str(env["state_file"])]
    if dry_run:
        cmd.append("-n")
    cmd += list(extra_args)

    run_env = os.environ.copy()
    run_env["PATH"] = f"{env['_bin_dir']}:{run_env['PATH']}"

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        check=False,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestAllCompleted:
    """All replicas completed -- email says all done, no resubmit."""

    def test_all_done_report(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        result = _run(monitor_env, dry_run=True)
        assert result.returncode == 0

        out = _strip_ansi(result.stdout)
        assert "Completed: 2/2" in out
        assert "All jobs completed" in out

    def test_all_done_clears_state(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        monitor_env["state_file"].write_text("5")
        _run(monitor_env, dry_run=True)
        assert not monitor_env["state_file"].exists()


class TestFailedWithRestart:
    """Failed replicas are restarted via check_status.sh -R."""

    def test_restart_count_in_report(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        result = _run(monitor_env)
        assert result.returncode == 0

        out = _strip_ansi(result.stdout)
        assert "Failed: 1" in out
        assert "Resubmitting rbfe_A_B" in out
        assert "Submitted batch job 99999" in out

    def test_restart_triggers_self_resubmit(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        result = _run(monitor_env)
        assert result.returncode == 0

        # sbatch should be called for self-resubmit (monitor.sbatch)
        sbatch_log = monitor_env["sbatch_log"].read_text()
        assert "monitor.sbatch" in sbatch_log


class TestDryRun:
    """Dry run: no restarts, no sbatch calls, no email."""

    def test_dry_run_shows_would_restart(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        result = _run(monitor_env, dry_run=True)
        assert result.returncode == 0

        out = _strip_ansi(result.stdout)
        assert "[DRY RUN] would restart" in out
        assert "Failed: 1" in out

    def test_dry_run_no_sbatch(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        _run(monitor_env, dry_run=True)

        sbatch_log = monitor_env["sbatch_log"].read_text().strip()
        assert sbatch_log == ""

    def test_dry_run_no_email(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        _run(monitor_env, dry_run=True)

        mail_log = monitor_env["mail_log"].read_text().strip()
        assert mail_log == ""


class TestErrorEntries:
    """Error entries are reported but not restarted."""

    def test_error_in_report(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_ERROR)
        result = _run(monitor_env, dry_run=True)
        assert result.returncode == 0

        out = _strip_ansi(result.stdout)
        assert "Error: 1" in out
        assert "Errors (not restarted)" in out
        assert "multiple matching jobs" in out


class TestMixedDirectories:
    """Multiple directories with mixed statuses."""

    def test_two_directories(self, monitor_env: dict[str, Path]) -> None:
        # check_status.sh mock returns the same output for both dirs.
        monitor_env["cs_response"].write_text(_STATUS_MIXED_WITH_RESTART)

        project2 = monitor_env["project"].parent / "project2"
        (project2 / "transformations").mkdir(parents=True)
        (project2 / "results").mkdir(parents=True)

        result = _run(monitor_env, dirs=[monitor_env["project"], project2])
        assert result.returncode == 0

        out = _strip_ansi(result.stdout)
        # Should show stats for both directories.
        assert out.count("Completed: 1/4") == 2


class TestIterationTracking:
    """Iteration counter increments across runs."""

    def test_first_iteration(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        result = _run(monitor_env, dry_run=True)
        assert result.returncode == 0
        assert "Iteration 1" in _strip_ansi(result.stdout)
        assert monitor_env["state_file"].read_text().strip() == "1"

    def test_increments(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["state_file"].write_text("3")
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        result = _run(monitor_env, dry_run=True)
        assert result.returncode == 0
        assert "Iteration 4" in _strip_ansi(result.stdout)
        assert monitor_env["state_file"].read_text().strip() == "4"


class TestSelfResubmit:
    """Self-resubmission when jobs are still running."""

    def test_resubmits_when_active(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        result = _run(monitor_env, "-i", "2")
        assert result.returncode == 0

        sbatch_log = monitor_env["sbatch_log"].read_text()
        assert "now+2hour" in sbatch_log
        assert "--dependency=singleton" in sbatch_log

    def test_no_resubmit_when_all_done(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        _run(monitor_env)

        sbatch_log = monitor_env["sbatch_log"].read_text().strip()
        assert "monitor.sbatch" not in sbatch_log


class TestEmailSending:
    """Email is sent with correct subject and body."""

    def test_email_sent_with_restart_count(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env)

        mail_log = monitor_env["mail_log"].read_text()
        assert "1 restart(s)" in mail_log

    def test_email_all_completed(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        _run(monitor_env)

        mail_log = monitor_env["mail_log"].read_text()
        assert "All jobs completed" in mail_log


class TestMissingDirectory:
    """Non-existent project directory is warned about."""

    def test_missing_dir_warning(self, monitor_env: dict[str, Path]) -> None:
        fake = monitor_env["project"].parent / "nonexistent"
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        result = _run(monitor_env, dirs=[fake, monitor_env["project"]])
        assert result.returncode == 0
        assert "does not exist" in _strip_ansi(result.stdout + result.stderr)


class TestNoDirectories:
    """No -d flag exits with error."""

    def test_no_dirs(self, monitor_env: dict[str, Path]) -> None:
        result = _run(monitor_env, dirs=[])
        assert result.returncode == 2
        assert "at least one -d DIR is required" in result.stderr


class TestHelp:
    """The -h flag prints usage."""

    def test_help_flag(self, monitor_env: dict[str, Path]) -> None:
        result = _run(monitor_env, "-h")
        assert result.returncode == 0
        assert "Usage:" in result.stdout
