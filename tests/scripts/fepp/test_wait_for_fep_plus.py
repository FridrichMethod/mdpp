"""Check native completion handling without contacting a real Suite or server."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).parents[3] / "examples/fepp/wait_for_fep_plus.py"
SPEC = importlib.util.spec_from_file_location("fepp_wait_example", SCRIPT)
assert SPEC and SPEC.loader
WAITER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WAITER)


def status(**changes: object) -> dict[str, object]:
    values = {
        "name": "trial",
        "complete": True,
        "succeeded": True,
        "downloaded": True,
        "status": "completed",
        "exit_status": "finished",
    }
    values.update(changes)
    return values


def query_result(value: dict[str, object], run_dir: Path) -> subprocess.CompletedProcess[str]:
    value = {"launch_dir": str(run_dir), **value}
    return subprocess.CompletedProcess([], 0, json.dumps(value), "")


def test_waits_through_running_and_pending_download(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "trial_out.fmp").write_text("native output")
    query = Mock(
        side_effect=[
            query_result(status(complete=False, succeeded=None, downloaded=False), tmp_path),
            query_result(status(downloaded=False), tmp_path),
            query_result(status(), tmp_path),
        ]
    )
    monkeypatch.setattr(WAITER.subprocess, "run", query)
    monkeypatch.setattr(WAITER.time, "sleep", lambda _: None)
    receipt = WAITER.wait_for_job(
        "native-id",
        schrodinger=tmp_path,
        run_dir=tmp_path,
        jobname="trial",
    )
    assert query.call_count == 3
    assert receipt["job_id"] == "native-id"
    assert len(receipt["sha256"]) == 64
    assert "scientific_validation" in receipt


@pytest.mark.parametrize("value", [status(succeeded=False), status(name="another")])
def test_terminal_failure_or_wrong_job_rejected(tmp_path: Path, monkeypatch, value) -> None:
    (tmp_path / "trial_out.fmp").write_text("partial output exists")
    monkeypatch.setattr(WAITER.subprocess, "run", Mock(return_value=query_result(value, tmp_path)))
    with pytest.raises(RuntimeError):
        WAITER.wait_for_job("id", schrodinger=tmp_path, run_dir=tmp_path, jobname="trial")


@pytest.mark.parametrize("contents", [None, ""])
def test_downloaded_requires_real_output(tmp_path: Path, monkeypatch, contents) -> None:
    if contents is not None:
        (tmp_path / "trial_out.fmp").write_text(contents)
    monkeypatch.setattr(
        WAITER.subprocess, "run", Mock(return_value=query_result(status(), tmp_path))
    )
    with pytest.raises(FileNotFoundError):
        WAITER.wait_for_job("id", schrodinger=tmp_path, run_dir=tmp_path, jobname="trial")


def test_ambiguous_lookup_does_not_launch_again(tmp_path: Path, monkeypatch) -> None:
    query = Mock(side_effect=subprocess.CalledProcessError(1, ["run"]))
    monkeypatch.setattr(WAITER.subprocess, "run", query)
    with pytest.raises(RuntimeError, match="reconcile"):
        WAITER.wait_for_job("id", schrodinger=tmp_path, run_dir=tmp_path, jobname="trial")
    assert query.call_count == 1


def test_timeout_does_not_stop_job(tmp_path: Path, monkeypatch) -> None:
    query = Mock(return_value=query_result(status(complete=False), tmp_path))
    monkeypatch.setattr(WAITER.subprocess, "run", query)
    monkeypatch.setattr(WAITER.time, "monotonic", Mock(side_effect=[0, 0, 2, 2]))
    monkeypatch.setattr(WAITER.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="not stopped"):
        WAITER.wait_for_job(
            "id",
            schrodinger=tmp_path,
            run_dir=tmp_path,
            jobname="trial",
            timeout_seconds=1,
        )
    assert query.call_count == 1


def test_vendor_query_uses_job_constructor_and_defers_terminal_access(monkeypatch, capsys) -> None:
    class RunningJob:
        Name = "trial"
        Dir = "/example/trial"
        Status = "running"

        def isComplete(self) -> bool:
            return False

        def isDownloaded(self) -> bool:
            return False

        def succeeded(self) -> bool:
            raise AssertionError("must not query running-job exit status")

        @property
        def ExitStatus(self) -> str:
            raise AssertionError("must not query running-job exit status")

    constructor = Mock(return_value=RunningJob())
    vendor = types.ModuleType("schrodinger.job")
    monkeypatch.setattr(vendor, "jobcontrol", types.SimpleNamespace(Job=constructor), raising=False)
    monkeypatch.setitem(sys.modules, "schrodinger.job", vendor)
    monkeypatch.setattr(sys, "argv", ["query", "native-id"])
    exec(WAITER._QUERY, {})
    constructor.assert_called_once_with("native-id")
    report = json.loads(capsys.readouterr().out)
    assert report["succeeded"] is None
    assert report["exit_status"] is None


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_wait_limit(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        WAITER.wait_for_job(
            "id",
            schrodinger=tmp_path,
            run_dir=tmp_path,
            jobname="trial",
            timeout_seconds=timeout,
        )


@pytest.mark.parametrize("directory", ["/another/run", "", None, "relative/run", 123])
def test_same_name_wrong_or_invalid_directory_rejected(
    tmp_path: Path, monkeypatch, directory
) -> None:
    (tmp_path / "trial_out.fmp").write_text("stale output from another same-name run")
    response = query_result(status(launch_dir=directory), tmp_path)
    monkeypatch.setattr(WAITER.subprocess, "run", Mock(return_value=response))
    with pytest.raises(RuntimeError, match="launch directory"):
        WAITER.wait_for_job("id", schrodinger=tmp_path, run_dir=tmp_path, jobname="trial")
