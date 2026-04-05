"""Tests for scripts/gromacs/data_transfer/dtn_download.sh.

Only argument parsing and validation are tested; actual SSH/rsync
calls are not executed (the script would fail without a real host).
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

DTN_SH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "gromacs"
    / "data_transfer"
    / "dtn_download.sh"
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _setup_shims(tmp_path: Path) -> dict[str, Path]:
    """Create shim executables for ssh, rsync, and parallel.

    ssh shim: prints one subdirectory name per line (simulates remote find).
    rsync shim: no-op (logs call).
    parallel shim: runs the command for each ::: argument sequentially.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    ssh_resp = tmp_path / "ssh_response.txt"
    ssh_resp.write_text("subdir_a\nsubdir_b\n")

    ssh_shim = bin_dir / "ssh"
    ssh_shim.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            # If called with -n (non-interactive), print the canned response
            resp="{ssh_resp}"
            if [[ -f "$resp" ]]; then cat "$resp"; fi
            """
        )
    )

    rsync_log = tmp_path / "rsync_log.txt"
    rsync_log.write_text("")
    rsync_shim = bin_dir / "rsync"
    rsync_shim.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            echo "$@" >> "{rsync_log}"
            """
        )
    )

    for shim in (ssh_shim, rsync_shim):
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)

    return {
        "ssh_response": ssh_resp,
        "rsync_log": rsync_log,
        "_bin_dir": bin_dir,
    }


def _env_with_shims(shims: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{shims['_bin_dir']}:{env['PATH']}"
    return env


def _run(
    shims: dict[str, Path],
    *args: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(DTN_SH), *args],
        capture_output=True,
        text=True,
        env=_env_with_shims(shims),
        cwd=str(cwd),
        check=False,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not Path("/usr/bin/parallel").exists()
    and not any((Path(p) / "parallel").exists() for p in os.environ.get("PATH", "").split(":")),
    reason="GNU parallel not found",
)


class TestNoArguments:
    """No arguments prints usage and exits 1."""

    def test_no_args(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        result = _run(shims, cwd=tmp_path)
        assert result.returncode == 1
        assert "Usage:" in result.stderr


class TestOneArgument:
    """Only one positional arg (need two) prints usage."""

    def test_one_arg(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        result = _run(shims, "/remote/path", cwd=tmp_path)
        assert result.returncode == 1
        assert "Usage:" in result.stderr


class TestDryRunFlag:
    """The -n flag is accepted and passed through to rsync."""

    def test_dry_run(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        local_dir = tmp_path / "local"
        result = _run(shims, "-n", "/remote/path", str(local_dir), cwd=tmp_path)
        assert result.returncode == 0
        rsync_log = shims["rsync_log"].read_text()
        assert "--dry-run" in rsync_log


class TestJobsFlag:
    """The -j flag is accepted."""

    def test_jobs_flag(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        local_dir = tmp_path / "local"
        result = _run(shims, "-j", "2", "/remote/path", str(local_dir), cwd=tmp_path)
        assert result.returncode == 0


class TestCreatesLocalDir:
    """The local directory is created if it doesn't exist."""

    def test_mkdir(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        local_dir = tmp_path / "new_local"
        assert not local_dir.exists()
        result = _run(shims, "/remote/path", str(local_dir), cwd=tmp_path)
        assert result.returncode == 0
        assert local_dir.is_dir()


class TestNoSubdirsFound:
    """SSH returns no subdirectories — exits 1."""

    def test_empty_remote(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        shims["ssh_response"].write_text("")
        local_dir = tmp_path / "local"
        result = _run(shims, "/remote/path", str(local_dir), cwd=tmp_path)
        assert result.returncode == 1
        assert "No subdirectories found" in result.stderr


class TestTransferLogDir:
    """Transfer log directory is created inside the local dir."""

    def test_logdir_created(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        local_dir = tmp_path / "local"
        result = _run(shims, "/remote/path", str(local_dir), cwd=tmp_path)
        assert result.returncode == 0
        assert (local_dir / ".transfer_logs").is_dir()


class TestDiscoveryOutput:
    """Stdout reports the number of discovered subdirectories."""

    def test_discovery_count(self, tmp_path: Path) -> None:
        shims = _setup_shims(tmp_path)
        shims["ssh_response"].write_text("a\nb\nc\n")
        local_dir = tmp_path / "local"
        result = _run(shims, "/remote/path", str(local_dir), cwd=tmp_path)
        assert result.returncode == 0
        assert "Found 3 subdirectories" in result.stdout
