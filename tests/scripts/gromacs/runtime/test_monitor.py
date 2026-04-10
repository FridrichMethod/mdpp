"""Tests for scripts/gromacs/runtime/monitor.sbatch.

Uses a mock check_status.sh that returns canned TSV output, so no
Slurm/parallel/gmx dependencies are needed.
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
    Path(__file__).resolve().parents[4] / "scripts" / "gromacs" / "runtime" / "monitor.sbatch"
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
    directory\tstatus\tprogress\tinfo
    /sims/system1\tcompleted\t100.000/100.000\tcheckpoint reached target time
    /sims/system2\tcompleted\t100.000/100.000\tcheckpoint reached target time
    """
)

_STATUS_WITH_FAILED = textwrap.dedent(
    """\
    directory\tstatus\tprogress\tinfo
    /sims/system1\tcompleted\t100.000/100.000\tcheckpoint reached target time
    /sims/system2\tfailed\t50.000/100.000\tincomplete and no matching active job in squeue
    /sims/system3\tactive\t75.000/100.000\tjob in squeue: 12345:RUNNING
    """
)

_STATUS_WITH_FAILED_AND_RESTART = (
    _STATUS_WITH_FAILED
    + "\n"
    + textwrap.dedent(
        """\
        Resubmitting: /sims/system2
        Submitted batch job 99999
        """
    )
)

_STATUS_WITH_ERROR = textwrap.dedent(
    """\
    directory\tstatus\tprogress\tinfo
    /sims/system1\terror\tNA/NA\tmultiple active jobs match this workdir: 1:R,2:R
    /sims/system2\tcompleted\t100.000/100.000\tcheckpoint reached target time
    """
)

_STATUS_MIXED = textwrap.dedent(
    """\
    directory\tstatus\tprogress\tinfo
    /sims/system1\tcompleted\t100.000/100.000\tcheckpoint reached target time
    /sims/system2\tfailed\t50.000/100.000\tincomplete and no matching active job in squeue
    /sims/system3\tfailed\t25.000/100.000\tincomplete and no matching active job in squeue
    /sims/system4\tactive\t80.000/100.000\tjob in squeue: 456:RUNNING
    """
)

_STATUS_MIXED_WITH_RESTART = (
    _STATUS_MIXED
    + "\n"
    + textwrap.dedent(
        """\
        Resubmitting: /sims/system2
        Submitted batch job 88888
        Resubmitting: /sims/system3
        Submitted batch job 99999
        """
    )
)

_STATUS_WITH_UNEXPECTED = textwrap.dedent(
    """\
    directory\tstatus\tprogress\tinfo
    /sims/system1\tcompleted\t100.000/100.000\tcheckpoint reached target time
    /sims/system2\tpending\t0.000/100.000\twaiting for resources
    """
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

    # Fake project directory (GROMACS: just needs to exist as a root).
    project = tmp_path / "project"
    project.mkdir(parents=True)

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


def _write_per_dir_mock(env: dict[str, Path], responses: dict[str, tuple[str, int]]) -> None:
    """Override check_status.sh to return per-directory responses.

    Args:
        env: Monitor environment dict from the ``monitor_env`` fixture.
        responses: Mapping of directory basename to (tsv_output, exit_code).
    """
    resp_dir = env["monitor_sbatch"].parent.parent / "responses"
    resp_dir.mkdir(exist_ok=True)

    for basename, (output, exit_code) in responses.items():
        (resp_dir / f"{basename}.txt").write_text(output)
        (resp_dir / f"{basename}.exit").write_text(str(exit_code))

    mock_cs = env["monitor_sbatch"].parent / "check_status.sh"
    mock_cs.write_text(
        f"""#!/usr/bin/env bash
DIR=""
while getopts ":j:t:r:Rh" opt; do
    case "$opt" in r) DIR="$OPTARG" ;; *) ;; esac
done
dirname=$(basename "$DIR")
resp="{resp_dir}/$dirname.txt"
exit_file="{resp_dir}/$dirname.exit"
if [[ -f "$resp" ]]; then cat "$resp"; fi
if [[ -f "$exit_file" ]]; then exit $(cat "$exit_file"); fi
exit 0
"""
    )
    _make_executable(mock_cs)


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
        project2.mkdir(parents=True)

        result = _run(monitor_env, dirs=[monitor_env["project"], project2])
        out = _strip_ansi(result.stdout)
        assert out.count("1/4 completed") == 2


class TestRestartReport:
    """Failed simulations are restarted and reported."""

    def test_restart_lines_in_output(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        result = _run(monitor_env)
        out = _strip_ansi(result.stdout)
        assert "Resubmitting: /sims/system2" in out
        assert "Submitted batch job 99999" in out

    def test_restart_triggers_self_resubmit(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env)
        assert "monitor.sbatch" in monitor_env["sbatch_log"].read_text()


class TestErrorReport:
    """Error entries are reported but not restarted."""

    def test_error_in_report(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_ERROR)
        result = _run(monitor_env, dry_run=True)
        out = _strip_ansi(result.stdout)
        assert "1 error" in out
        assert "multiple active jobs" in out


class TestDryRun:
    """Dry run: no restarts, no sbatch, no email."""

    def test_shows_would_restart(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED)
        result = _run(monitor_env, dry_run=True)
        out = _strip_ansi(result.stdout)
        assert "[DRY RUN] would restart" in out
        # GROMACS dry run shows simulation name and progress.
        assert "system2" in out
        assert "50.000/100.000" in out

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
        mail = monitor_env["mail_log"].read_text()
        assert "GROMACS Monitor" in mail
        assert "All" in mail
        assert "completed" in mail

    def test_multi_dir_all_completed_stops(self, monitor_env: dict[str, Path]) -> None:
        """When ALL directories are complete, monitor stops."""
        project2 = monitor_env["project"].parent / "project2"
        project2.mkdir(parents=True)

        _write_per_dir_mock(
            monitor_env,
            {
                monitor_env["project"].name: (_STATUS_ALL_COMPLETED, 0),
                project2.name: (_STATUS_ALL_COMPLETED, 0),
            },
        )
        _run(monitor_env, dirs=[monitor_env["project"], project2])
        assert "monitor.sbatch" not in monitor_env["sbatch_log"].read_text()


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


class TestTargetTimePassthrough:
    """The -t flag is forwarded to check_status.sh and preserved on resubmit."""

    def test_target_time_in_resubmit_args(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env, "-t", "500")
        sbatch_log = monitor_env["sbatch_log"].read_text()
        assert "-t 500" in sbatch_log

    def test_target_time_passed_to_check_status(self, monitor_env: dict[str, Path]) -> None:
        """check_status.sh receives -t when the monitor is given -t."""
        cs_log = monitor_env["monitor_sbatch"].parent.parent / "cs_args.log"
        cs_log.write_text("")
        # Replace mock to log the arguments it receives.
        mock_cs = monitor_env["monitor_sbatch"].parent / "check_status.sh"
        cs_response = monitor_env["cs_response"]
        cs_response.write_text(_STATUS_ALL_COMPLETED)
        mock_cs.write_text(
            f"""#!/usr/bin/env bash
echo "$@" >> "{cs_log}"
cat "{cs_response}"
exit 0
"""
        )
        _make_executable(mock_cs)
        _run(monitor_env, "-t", "200", dry_run=True)
        assert "-t 200" in cs_log.read_text()

    def test_invalid_target_time_rejected(self, monitor_env: dict[str, Path]) -> None:
        result = _run(monitor_env, "-t", "abc")
        assert result.returncode == 2
        assert "numeric" in result.stderr


class TestEmailContent:
    """Email subject reflects status summary."""

    def test_subject_has_counts(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env)
        mail = monitor_env["mail_log"].read_text()
        assert "1/3 completed" in mail

    def test_subject_prefix(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        _run(monitor_env)
        mail = monitor_env["mail_log"].read_text()
        assert "[GROMACS Monitor]" in mail


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

        spool = monitor_env["monitor_sbatch"].parent.parent / "spool"
        spool.mkdir()
        spool_copy = spool / "slurm_script"
        shutil.copy2(monitor_env["monitor_sbatch"], spool_copy)

        scripts_dir = monitor_env["monitor_sbatch"].parent

        cmd = ["bash", str(spool_copy), "-d", str(monitor_env["project"]), "-n"]
        run_env = os.environ.copy()
        run_env["PATH"] = f"{monitor_env['_bin_dir']}:{run_env['PATH']}"
        run_env["SLURM_JOB_ID"] = "12345"
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=run_env, cwd=str(scripts_dir), check=False
        )
        assert result.returncode == 0
        assert "2/2 completed" in _strip_ansi(result.stdout)


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


class TestSkippedDirPreventsAllDone:
    """Monitor must not stop when a directory is skipped (GRAND_SKIPPED > 0)."""

    def test_check_status_failure_prevents_stop(self, monitor_env: dict[str, Path]) -> None:
        """check_status.sh fails for one dir -> monitor resubmits even if other dir is done."""
        project1 = monitor_env["project"]
        project2 = project1.parent / "project2"
        project2.mkdir(parents=True)

        _write_per_dir_mock(
            monitor_env,
            {
                project1.name: (_STATUS_ALL_COMPLETED, 0),
                project2.name: ("", 1),
            },
        )
        _run(monitor_env, dirs=[project1, project2])
        assert "monitor.sbatch" in monitor_env["sbatch_log"].read_text()

    def test_missing_dir_prevents_stop(self, monitor_env: dict[str, Path]) -> None:
        """Non-existent directory -> monitor resubmits even if other dir is done."""
        monitor_env["cs_response"].write_text(_STATUS_ALL_COMPLETED)
        fake = monitor_env["project"].parent / "nonexistent"
        _run(monitor_env, dirs=[monitor_env["project"], fake])
        assert "monitor.sbatch" in monitor_env["sbatch_log"].read_text()


class TestCompletedEqualsTotal:
    """ALL_DONE requires COMPLETED == TOTAL, not just zero active/failed/error."""

    def test_unexpected_status_prevents_stop(self, monitor_env: dict[str, Path]) -> None:
        """A job with unrecognized status prevents ALL_DONE."""
        monitor_env["cs_response"].write_text(_STATUS_WITH_UNEXPECTED)
        _run(monitor_env)
        assert "monitor.sbatch" in monitor_env["sbatch_log"].read_text()


class TestRelativePathResolution:
    """Relative -d paths are resolved to absolute before resubmission."""

    def test_relative_paths_resolved_in_resubmit(self, monitor_env: dict[str, Path]) -> None:
        monitor_env["cs_response"].write_text(_STATUS_WITH_FAILED_AND_RESTART)
        project = monitor_env["project"]
        launch_dir = project.parent

        cmd = [
            "bash",
            str(monitor_env["monitor_sbatch"]),
            "-d",
            str(project.relative_to(launch_dir)),
        ]
        run_env = os.environ.copy()
        run_env["PATH"] = f"{monitor_env['_bin_dir']}:{run_env['PATH']}"
        subprocess.run(
            cmd, capture_output=True, text=True, env=run_env, cwd=str(launch_dir), check=False
        )

        sbatch_args = monitor_env["sbatch_log"].read_text()
        match = re.search(r"-d\s+(\S+)", sbatch_args)
        assert match is not None, f"No -d flag in resubmit args: {sbatch_args}"
        dir_arg = match.group(1)
        assert dir_arg.startswith("/"), f"Expected absolute path, got: {dir_arg}"
        assert dir_arg == str(project)
