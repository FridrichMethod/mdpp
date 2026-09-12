"""Integration tests for FEPP shell workflows using a fake Schrodinger Suite."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

FEPP_DIR = Path(__file__).parents[3] / "examples" / "fepp"
BUILD_SCRIPT = FEPP_DIR / "build_fepp_inputs.sh"
RUN_SCRIPT = FEPP_DIR / "run_fep_plus.sh"
SINGLE_BUILD_SCRIPT = FEPP_DIR / "rbfe" / "build_inputs.sh"
SINGLE_RUN_SCRIPT = FEPP_DIR / "rbfe" / "run_fep_plus.sh"
PAIRED_BUILD_SCRIPT = FEPP_DIR / "rbfe_open_closed" / "build_inputs.sh"
PAIRED_RUN_SCRIPT = FEPP_DIR / "rbfe_open_closed" / "run_fep_plus.sh"


def write_executable(path: Path, content: str) -> None:
    """Write a test shim and mark it executable."""
    path.write_text(content)
    path.chmod(0o755)


@pytest.fixture
def fake_workflow(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    """Create minimal valid inputs and deterministic Suite command shims."""
    suite = tmp_path / "suite"
    utilities = suite / "utilities"
    utilities.mkdir(parents=True)
    log = tmp_path / "commands.log"
    log.write_text("")
    (suite / "version.txt").write_text("Suite TEST Build 1\n")

    write_executable(
        suite / "run",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'run %s\n' "$*" >>"${FAKE_LOG}"
if [[ "${1-}" == "python3" ]]; then
    if [[ "$(basename "${2-}")" == "extract_edge_mappings.py" ]]; then
        shift 2
        output=""
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == "-o" ]]; then
                output="$2"
                shift 2
            else
                shift
            fi
        done
        printf '{"schema_version":2,"mapping_fingerprint":"%064d","edges":[]}\n' 0 >"${output}"
        exit 0
    fi
    if [[ "$(basename "${2-}")" == "compare_receptor_microstates.py" ]]; then
        shift 2
        output=""
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == "-o" ]]; then
                output="$2"
                shift 2
            else
                shift
            fi
        done
        printf '{"identical_declared_microstates": true}\n' >"${output}"
        exit 0
    fi
    shift
    exec python3 "$@"
fi
if [[ "${1-}" == "-FROM" && "${3-}" == "fep_mapper.py" ]]; then
    shift 3
    pose_viewer="$1"
    shift
    output=""
    topology=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -o) output="$2"; shift 2 ;;
            -t) topology="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    printf 'mapper %s %s\n' "${pose_viewer}" "${topology}" >>"${FAKE_LOG}"
    printf 'fmp %s %s\n' "$(sha256sum "${pose_viewer}" | awk '{print $1}')" "${topology}" >"${output}.fmp"
    printf 'abc:def  # A7_AMP -> TEST_NODE\n' >"${output}.edge"
    exit 0
fi
echo "unsupported fake run invocation: $*" >&2
exit 2
""",
    )
    write_executable(
        utilities / "prepwizard",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'prepwizard %s\n' "$*" >>"${FAKE_LOG}"
args=("$@")
count=${#args[@]}
input=${args[count-2]}
output=${args[count-1]}
{
    printf 'prepared %s\n' "$*"
    sha256sum "${input}"
} >"${output}"
""",
    )
    write_executable(
        utilities / "structcat",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'structcat %s\n' "$*" >>"${FAKE_LOG}"
inputs=()
output=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -isd|-imae) inputs+=("$2"); shift 2 ;;
        -omae) output="$2"; shift 2 ;;
        *) shift ;;
    esac
done
: >"${output}"
for input in "${inputs[@]}"; do
    sha256sum "${input}" >>"${output}"
done
""",
    )
    write_executable(
        suite / "fep_plus",
        r"""#!/usr/bin/env bash
set -euo pipefail
printf 'fep_plus %s\n' "$*" >>"${FAKE_LOG}"
""",
    )

    inputs = tmp_path / "inputs"
    ligand_dir = inputs / "ligands"
    ligand_dir.mkdir(parents=True)
    shutil.copy2(FEPP_DIR / "inputs" / "ligands" / "A7_AMP.sdf", ligand_dir)
    smiles_lines = (FEPP_DIR / "inputs" / "ligands_amp_fep.smi").read_text().splitlines()
    a7_line = next(line for line in smiles_lines if line.endswith(" A7_AMP"))
    (inputs / "ligands_amp_fep.smi").write_text(f"smiles title\n{a7_line}\n")
    (inputs / "protein_open.pdb").write_text("OPEN PROTEIN\n")
    (inputs / "protein_closed.pdb").write_text("CLOSED PROTEIN\n")

    work = tmp_path / "work"
    env = os.environ.copy()
    env.update({
        "SCHRODINGER": str(suite),
        "FEPP_INPUT_DIR": str(inputs),
        "FEPP_WORK_DIR": str(work),
        "FAKE_LOG": str(log),
    })
    return env, inputs, work, log


def run_script(
    script: Path,
    env: dict[str, str],
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """Run one FEPP shell script and capture its output."""
    return subprocess.run(
        ["bash", str(script), *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
        cwd=FEPP_DIR,
    )


def count_commands(log: Path, prefix: str) -> int:
    """Count fake-suite log records beginning with a command label."""
    return sum(line.startswith(prefix) for line in log.read_text().splitlines())


def tree_hashes(root: Path) -> dict[str, str]:
    """Hash every regular file below a generated-artifact tree."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_dependency_fingerprints_and_paired_map_cache(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, inputs, work, log = fake_workflow
    first = run_script(BUILD_SCRIPT, env)
    assert first.returncode == 0, first.stderr
    assert count_commands(log, "prepwizard ") == 2
    assert count_commands(log, "structcat ") == 3
    assert count_commands(log, "mapper ") == 2
    mapper_invocations = [
        line
        for line in log.read_text().splitlines()
        if line.startswith("run -FROM scisol fep_mapper.py ")
    ]
    assert len(mapper_invocations) == 2
    assert all(" -e 1 " in f" {line} " for line in mapper_invocations)

    cached = run_script(BUILD_SCRIPT, env)
    assert cached.returncode == 0, cached.stderr
    assert count_commands(log, "prepwizard ") == 2
    assert count_commands(log, "structcat ") == 3
    assert count_commands(log, "mapper ") == 2
    before_check = tree_hashes(work)
    checked = run_script(BUILD_SCRIPT, env, "--check")
    assert checked.returncode == 0, checked.stderr
    assert tree_hashes(work) == before_check

    ligand = inputs / "ligands" / "A7_AMP.sdf"
    ligand.write_text(ligand.read_text() + "\n")
    rebuilt = run_script(BUILD_SCRIPT, env)
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert count_commands(log, "prepwizard ") == 2
    assert count_commands(log, "structcat ") == 6
    assert count_commands(log, "mapper ") == 4

    (work / "open_map.edge").unlink()
    repaired = run_script(BUILD_SCRIPT, env)
    assert repaired.returncode == 0, repaired.stderr
    assert count_commands(log, "mapper ") == 5

    (work / "closed_map.fmp").write_text("tampered\n")
    stale = run_script(BUILD_SCRIPT, env, "--check")
    assert stale.returncode != 0
    assert "stale or missing artifact: closed FEP map pair" in stale.stderr


def test_launcher_records_explicit_protocol_and_unique_repeat_seeds(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, _, work, log = fake_workflow
    built = run_script(BUILD_SCRIPT, env)
    assert built.returncode == 0, built.stderr
    launched = run_script(
        RUN_SCRIPT,
        env,
        "-c",
        "both",
        "--repeats",
        "2",
        "--seed-base",
        "4000",
        "--jobname-base",
        "trial",
    )
    assert launched.returncode == 0, launched.stderr
    assert "does not establish simulation completion" in launched.stdout
    assert "wait_for_fep_plus.py" in launched.stdout

    fep_lines = [line for line in log.read_text().splitlines() if line.startswith("fep_plus ")]
    assert len(fep_lines) == 4
    for seed, line in zip([4000, 4002, 4001, 4003], fep_lines):
        assert f"-seed {seed}" in line
        assert "-ff OPLS4" in line
        assert "-custom-charge-mode assign" in line
        assert "-water SPC" in line
        assert "-ensemble muVT" in line
        assert "-time 5000" in line
        assert "-equilibration-time 20" in line
        assert "-lambda-windows 12" in line
        assert "-salt 0.0" in line

    run_dirs = sorted((work / "runs").iterdir())
    assert [path.name for path in run_dirs] == [
        "trial_closed_r01",
        "trial_closed_r02",
        "trial_open_r01",
        "trial_open_r02",
    ]
    assert all((path / "manifest.tsv").is_file() for path in run_dirs)
    snapshot_names = {
        "input_map.fmp",
        "input_map.edge",
        "input_map.provenance",
        "input_map_mappings.json",
        "input_ligands.maegz",
        "input_ligand_validation.json",
        "input_receptor_microstates.json",
        "input_paired_inputs.tsv",
    }
    assert all(snapshot_names <= {item.name for item in path.iterdir()} for path in run_dirs)
    assert all(line.endswith(" working_map.fmp") for line in fep_lines)
    assert all(str(work) not in line for line in fep_lines)
    for path in run_dirs:
        snapshot = path / "input_map.fmp"
        working = path / "working_map.fmp"
        assert working.read_bytes() == snapshot.read_bytes()
        assert snapshot.stat().st_mode & 0o222 == 0
        assert working.stat().st_mode & 0o200
        working.write_text("native workflow may return an updated input map")
        assert working.read_bytes() != snapshot.read_bytes()
        manifest = dict(
            line.split("\t", 1) for line in (path / "manifest.tsv").read_text().splitlines()
        )
        assert manifest["launch_map_file"] == "working_map.fmp"
        assert manifest["launch_map_initial_sha256"] == manifest["map_sha256"]
    seeds = {
        line.split("\t", 1)[1]
        for path in run_dirs
        for line in (path / "manifest.tsv").read_text().splitlines()
        if line.startswith("seed\t")
    }
    assert seeds == {"4000", "4001", "4002", "4003"}


def test_standalone_entry_point_needs_only_one_receptor_and_no_paired_artifacts(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, inputs, work, log = fake_workflow
    (inputs / "protein_closed.pdb").unlink()

    built = run_script(SINGLE_BUILD_SCRIPT, env, "-c", "both")
    assert built.returncode == 0, built.stderr
    assert count_commands(log, "prepwizard ") == 1
    assert (work / "state_inputs_open.tsv").is_file()
    assert not (work / "receptor_microstates.json").exists()
    assert not (work / "paired_inputs.tsv").exists()

    launched = run_script(
        SINGLE_RUN_SCRIPT,
        env,
        "--workflow",
        "paired",
        "-c",
        "both",
        "--repeats",
        "2",
        "--seed-base",
        "6000",
        "--jobname-base",
        "standalone",
    )
    assert launched.returncode == 0, launched.stderr
    run_dirs = sorted((work / "runs").iterdir())
    assert [path.name for path in run_dirs] == [
        "standalone_open_r01",
        "standalone_open_r02",
    ]
    for path in run_dirs:
        names = {item.name for item in path.iterdir()}
        assert "input_state_inputs.tsv" in names
        assert "input_receptor_microstates.json" not in names
        assert "input_paired_inputs.tsv" not in names
        manifest = dict(
            line.split("\t", 1) for line in (path / "manifest.tsv").read_text().splitlines()
        )
        assert manifest["schema_version"] == "2"
        assert manifest["workflow_type"] == "single_rbfe"
        assert manifest["state"] == "open"
        assert "state_inputs_sha256" in manifest
        assert "paired_inputs_sha256" not in manifest
        assert "receptor_microstates_sha256" not in manifest


def test_workflow_wrappers_enforce_modes_and_separate_default_namespaces(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, _, work, _ = fake_workflow
    built = run_script(PAIRED_BUILD_SCRIPT, env, "-c", "open")
    assert built.returncode == 0, built.stderr

    single_help = run_script(SINGLE_RUN_SCRIPT, env, "--help")
    assert single_help.returncode == 0, single_help.stderr
    assert "default: single" in single_help.stdout
    assert "default: open" in single_help.stdout
    assert "default: fepp_rbfe" in single_help.stdout
    paired_help = run_script(PAIRED_RUN_SCRIPT, env, "--help")
    assert paired_help.returncode == 0, paired_help.stderr
    assert "default: paired" in paired_help.stdout
    assert "default: both" in paired_help.stdout
    assert "default: fepp_open_closed" in paired_help.stdout

    single = run_script(SINGLE_RUN_SCRIPT, env, "--workflow", "paired", "-c", "both")
    assert single.returncode == 0, single.stderr
    paired = run_script(PAIRED_RUN_SCRIPT, env, "--workflow", "single", "-c", "open")
    assert paired.returncode == 0, paired.stderr

    run_dirs = sorted(path.name for path in (work / "runs").iterdir())
    assert run_dirs == [
        "fepp_open_closed_closed_r01",
        "fepp_open_closed_open_r01",
        "fepp_rbfe",
    ]
    manifests = {
        path.name: dict(
            line.split("\t", 1) for line in (path / "manifest.tsv").read_text().splitlines()
        )
        for path in (work / "runs").iterdir()
    }
    assert manifests["fepp_rbfe"]["workflow_type"] == "single_rbfe"
    assert manifests["fepp_rbfe"]["state"] == "open"
    for name in ("fepp_open_closed_open_r01", "fepp_open_closed_closed_r01"):
        assert manifests[name]["workflow_type"] == "paired_open_closed"


def test_standalone_launcher_rejects_both_conformations(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, _, _, _ = fake_workflow
    result = run_script(RUN_SCRIPT, env, "--workflow", "single", "-c", "both")
    assert result.returncode != 0
    assert "single workflow requires exactly one conformation" in result.stderr


def test_seeds_are_state_stable_across_separate_launches(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, _, work, _ = fake_workflow
    built = run_script(BUILD_SCRIPT, env)
    assert built.returncode == 0, built.stderr
    open_run = run_script(
        RUN_SCRIPT,
        env,
        "-c",
        "open",
        "--repeats",
        "2",
        "--seed-base",
        "5000",
        "--jobname-base",
        "separate_open",
    )
    assert open_run.returncode == 0, open_run.stderr
    closed_run = run_script(
        RUN_SCRIPT,
        env,
        "-c",
        "closed",
        "--repeats",
        "2",
        "--seed-base",
        "5000",
        "--jobname-base",
        "separate_closed",
    )
    assert closed_run.returncode == 0, closed_run.stderr

    seeds_by_state: dict[str, set[str]] = {"open": set(), "closed": set()}
    for path in (work / "runs").iterdir():
        manifest = dict(
            line.split("\t", 1) for line in (path / "manifest.tsv").read_text().splitlines()
        )
        seeds_by_state[manifest["state"]].add(manifest["seed"])
    assert seeds_by_state == {"open": {"5000", "5002"}, "closed": {"5001", "5003"}}


def test_npt_uses_suite_equilibration_default(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, _, work, log = fake_workflow
    built = run_script(BUILD_SCRIPT, env)
    assert built.returncode == 0, built.stderr
    launched = run_script(
        RUN_SCRIPT,
        env,
        "-c",
        "open",
        "--ensemble",
        "NPT",
        "--jobname-base",
        "npt_trial",
    )
    assert launched.returncode == 0, launched.stderr
    fep_line = next(line for line in log.read_text().splitlines() if line.startswith("fep_plus "))
    assert "-ensemble NPT" in fep_line
    assert "-equilibration-time 240" in fep_line
    manifest = dict(
        line.split("\t", 1)
        for line in (work / "runs" / "npt_trial" / "manifest.tsv").read_text().splitlines()
    )
    assert manifest["equilibration_time_ps"] == "240"


def test_launcher_propagates_microstate_mismatch_policy(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
) -> None:
    env, _, work, _ = fake_workflow
    built = run_script(BUILD_SCRIPT, env, "--allow-microstate-mismatch")
    assert built.returncode == 0, built.stderr
    launched = run_script(
        RUN_SCRIPT,
        env,
        "-c",
        "open",
        "--allow-microstate-mismatch",
        "--jobname-base",
        "mismatch_policy",
    )
    assert launched.returncode == 0, launched.stderr
    manifest = dict(
        line.split("\t", 1)
        for line in (work / "runs" / "mismatch_policy" / "manifest.tsv").read_text().splitlines()
    )
    assert manifest["input_allow_microstate_mismatch"] == "1"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--seed-base", "2147483647"), "seed-base must be <= 2147483646"),
        (
            ("--seed-base", "0", "--repeats", "1073741825"),
            "derived repeat seeds must be <= 2147483647",
        ),
    ],
)
def test_launcher_rejects_out_of_range_seed_schedule(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
    arguments: tuple[str, ...],
    message: str,
) -> None:
    env, _, _, _ = fake_workflow
    result = run_script(RUN_SCRIPT, env, *arguments)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    ("version_contents", "message"),
    [
        (None, "Suite release metadata not found"),
        ("\t\r\n", "Suite release metadata is empty"),
    ],
)
def test_workflow_fails_closed_without_suite_release_metadata(
    fake_workflow: tuple[dict[str, str], Path, Path, Path],
    version_contents: str | None,
    message: str,
) -> None:
    env, _, _, _ = fake_workflow
    version_file = Path(env["SCHRODINGER"]) / "version.txt"
    if version_contents is None:
        version_file.unlink()
    else:
        version_file.write_text(version_contents)
    for script in (BUILD_SCRIPT, RUN_SCRIPT):
        result = run_script(script, env)
        assert result.returncode != 0
        assert message in result.stderr


@pytest.mark.parametrize(
    ("helper", "artifact"),
    [
        ("chem/validation.py", "ligand validation report"),
        ("prep/_fepp_receptor_worker.py", "paired receptor microstate report"),
        ("core/_fepp_mapping_worker.py", "open FEP map pair"),
    ],
)
def test_build_invalidates_when_packaged_implementation_changes(
    fake_workflow, tmp_path: Path, helper: str, artifact: str
) -> None:
    env, _, _, _ = fake_workflow
    repository = tmp_path / "checkout"
    examples = repository / "examples" / "fepp"
    examples.mkdir(parents=True)
    for name in (
        "build_fepp_inputs.sh",
        "validate_ligand_inputs.py",
        "compare_receptor_microstates.py",
        "extract_edge_mappings.py",
    ):
        shutil.copy2(FEPP_DIR / name, examples / name)
    source = repository / "src" / "mdpp"
    for relative in (
        "chem/validation.py",
        "prep/_fepp_receptor_worker.py",
        "core/_fepp_mapping_worker.py",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FEPP_DIR.parents[1] / "src" / "mdpp" / relative, target)
    script = examples / "build_fepp_inputs.sh"
    built = run_script(script, env)
    assert built.returncode == 0, built.stderr
    current = run_script(script, env, "--check")
    assert current.returncode == 0, current.stderr
    implementation = source / helper
    implementation.write_text(implementation.read_text() + "\n# changed implementation\n")
    stale = run_script(script, env, "--check")
    assert stale.returncode != 0
    assert f"stale or missing artifact: {artifact}" in stale.stderr
