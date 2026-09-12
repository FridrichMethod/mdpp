#!/usr/bin/env python3
"""Wait for native FEP+ completion and downloaded output before scientific export.

Run with the mdpp Python environment. Only the small read-only status query runs
under the requested Suite interpreter; this command never starts or stops jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

_QUERY = """
import json, sys
from schrodinger.job import jobcontrol
job = jobcontrol.Job(sys.argv[1])
complete = job.isComplete()
print(json.dumps({
    "name": job.Name,
    "launch_dir": job.Dir,
    "complete": complete,
    "downloaded": job.isDownloaded(),
    "succeeded": job.succeeded() if complete else None,
    "status": job.Status,
    "exit_status": job.ExitStatus if complete else None,
}))
"""


def _validate_identity(
    status: object, *, job_id: str, jobname: str, run_dir: Path
) -> dict[str, object]:
    """Reject output from another job or a malformed launch-directory record."""
    if not isinstance(status, dict) or status.get("name") != jobname:
        raise RuntimeError(f"Job {job_id} does not match expected job name {jobname}")
    launch_dir = status.get("launch_dir")
    if not isinstance(launch_dir, str) or not Path(launch_dir).is_absolute():
        raise RuntimeError(f"Job {job_id} has no valid absolute launch directory")
    try:
        directory_matches = Path(launch_dir).resolve() == run_dir.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify Job {job_id} launch directory") from exc
    if not directory_matches:
        raise RuntimeError(f"Job {job_id} launch directory differs from {run_dir}")
    return status


def wait_for_job(
    job_id: str,
    *,
    schrodinger: Path,
    run_dir: Path,
    jobname: str,
    timeout_seconds: float = 86400,
    poll_seconds: float = 30,
) -> dict[str, object]:
    """Require Job Control success, download completion and a nonempty FMP.

    Args:
        job_id: Exact JobId reported by the native launcher.
        schrodinger: Suite installation used to launch the job.
        run_dir: Directory receiving the downloaded output.
        jobname: Expected native job name, as recorded in manifest.tsv.
        timeout_seconds: Maximum elapsed wait; timing out does not kill the job.
        poll_seconds: Delay between read-only Job Control queries.

    Returns:
        Terminal Job Control status and the downloaded FMP checksum. This is an
        execution receipt; the separate extractor validates scientific results.

    Raises:
        ValueError: Job name or timing options are invalid.
        RuntimeError: Job lookup fails, identity mismatches, or the job fails.
        FileNotFoundError: Completed/downloaded job lacks its expected output.
        TimeoutError: The job or its download does not finish within the limit.
    """
    if not job_id.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+", jobname):
        raise ValueError("nonempty JobId and safe jobname are required")
    if any(not math.isfinite(v) or v <= 0 for v in (timeout_seconds, poll_seconds)):
        raise ValueError("timeout and poll intervals must be finite and positive")
    output = run_dir.resolve() / f"{jobname}_out.fmp"
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Job {job_id} has not completed and downloaded; it was not stopped")
        try:
            query = subprocess.run(
                [str(schrodinger / "run"), "python3", "-c", _QUERY, job_id],
                capture_output=True,
                text=True,
                check=True,
                timeout=min(30, remaining),
            )
            status = json.loads(query.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot verify Job {job_id}; reconcile it before relaunch") from exc
        status = _validate_identity(status, job_id=job_id, jobname=jobname, run_dir=run_dir)
        if status.get("complete") is True:
            if status.get("succeeded") is not True:
                raise RuntimeError(f"Job {job_id} failed: {status}")
            if status.get("downloaded") is True:
                if not output.is_file() or output.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"Job reported downloaded but output is missing: {output}"
                    )
                return {
                    "job_id": job_id,
                    "job_control": status,
                    "output": str(output),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "scientific_validation": "requires extract_fep_results.py",
                }
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def main() -> None:
    """Read CLI options and print a verified execution receipt as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="JobId from launcher.log; not a Slurm ID.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobname", required=True, help="Exact jobname in manifest.tsv.")
    parser.add_argument(
        "--schrodinger",
        type=Path,
        default=Path(os.environ.get("SCHRODINGER", "/apps/schrodinger2025-4")),
    )
    parser.add_argument("--timeout-seconds", type=float, default=86400)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        receipt = wait_for_job(
            args.job_id,
            schrodinger=args.schrodinger,
            run_dir=args.run_dir,
            jobname=args.jobname,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
