"""Float32 vs float64 numerical-consistency regression tests.

The package defaults to ``np.float32`` everywhere because mdtraj stores
trajectory coordinates in float32 and forcing fp64 doubles every
``O(N^2)`` intermediate, OOMing realistic-size trajectories.  This
module pins down the precision claims documented in
``src/mdpp/_dtype.py`` so that any future refactor that silently
changes the dtype policy is caught here -- on either side:

* fp32 paths that develop visible drift versus fp64 (the "default
  precision is no longer good enough" failure mode).
* fp64 paths that someone tries to demote to fp32 and lose load-bearing
  precision (the "well, it's fine on my fixture" failure mode).

The agreement tolerances mirror the empirical numbers cited in
``_dtype.py`` and the per-backend docstrings.  When a future
implementation legitimately tightens precision, the tolerance can be
ratcheted down -- but loosening it requires explicit justification and
a comment update.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest
from numpy.typing import NDArray

from mdpp._dtype import get_default_dtype, set_default_dtype
from mdpp.analysis.clustering import KMeans, compute_rmsd_matrix
from mdpp.analysis.contacts import (
    compute_contact_frequency,
    compute_contacts,
    compute_native_contacts,
)
from mdpp.analysis.decomposition import (
    compute_pca,
    compute_tica,
    featurize_ca_distances,
)
from mdpp.analysis.distance import compute_distances, compute_minimum_distance
from mdpp.analysis.fes import compute_fes_2d
from mdpp.analysis.hbond import compute_hbonds
from mdpp.analysis.metrics import (
    average_rmsf_with_sem,
    compute_dccm,
    compute_delta_rmsf,
    compute_radius_of_gyration,
    compute_rmsd,
    compute_rmsf,
    compute_sasa,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_protein_like_trajectory(
    n_frames: int = 100,
    n_residues: int = 24,
    seed: int = 13,
) -> md.Trajectory:
    """Build a simple peptide trajectory with realistic float32 coordinates.

    Each residue carries the four backbone heavy atoms (N, CA, C, O)
    plus a CB sidechain stub.  Coordinates are drawn from a Gaussian
    around an extended-chain template so PCA/TICA see real variance,
    but the dtype is deliberately float32 to mirror what mdtraj
    actually returns from a real trajectory file.
    """
    rng = np.random.default_rng(seed)
    topology = md.Topology()
    chain = topology.add_chain()
    template = []  # (name, element)
    for resseq in range(1, n_residues + 1):
        residue = topology.add_residue("ALA", chain, resSeq=resseq)
        topology.add_atom("N", md.element.nitrogen, residue)
        topology.add_atom("CA", md.element.carbon, residue)
        topology.add_atom("C", md.element.carbon, residue)
        topology.add_atom("O", md.element.oxygen, residue)
        topology.add_atom("CB", md.element.carbon, residue)
        template.extend([
            (resseq, "N"),
            (resseq, "CA"),
            (resseq, "C"),
            (resseq, "O"),
            (resseq, "CB"),
        ])

    n_atoms = topology.n_atoms
    base = np.zeros((n_atoms, 3), dtype=np.float32)
    for atom_index, (resseq, name) in enumerate(template):
        # Crude extended-chain placement so distances are physical.
        offset = {"N": 0.00, "CA": 0.15, "C": 0.30, "O": 0.45, "CB": 0.20}[name]
        base[atom_index, 0] = (resseq - 1) * 0.38 + offset
        base[atom_index, 1] = 0.05 if name == "CB" else 0.0

    # Per-frame Gaussian fluctuations of ~0.1 nm to drive RMSF / DCCM /
    # PCA away from the trivial all-zero answer.
    noise = rng.normal(scale=0.1, size=(n_frames, n_atoms, 3)).astype(np.float32)
    xyz = base[None, :, :] + noise
    time_ps = np.arange(n_frames, dtype=np.float64) * 10.0
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


@pytest.fixture()
def protein_like_trajectory() -> md.Trajectory:
    """Larger trajectory used by the dtype-consistency regression tests."""
    return _make_protein_like_trajectory()


@pytest.fixture()
def small_protein_trajectory() -> md.Trajectory:
    """Compact trajectory for slow-but-still-cheap analyses (PCA/TICA/RMSD matrix)."""
    return _make_protein_like_trajectory(n_frames=60, n_residues=12, seed=29)


# ---------------------------------------------------------------------------
# Default-dtype contract
# ---------------------------------------------------------------------------


def test_default_dtype_is_float32() -> None:
    """Lock in float32 as the package-wide default dtype.

    Defends against a future ``set_default_dtype`` typo silently
    changing the contract documented in ``_dtype.py``.
    """
    assert get_default_dtype() == np.dtype(np.float32)


def test_set_default_dtype_round_trips_to_float32(protein_like_trajectory) -> None:
    """Switching the global default propagates to every ``compute_*`` output.

    This is the only test that mutates the global default; it uses
    try/finally so a failure mid-test does not poison the rest of the
    suite.
    """
    saved = get_default_dtype()
    try:
        set_default_dtype(np.float64)
        rmsf64 = compute_rmsf(protein_like_trajectory, atom_selection="name CA")
        assert rmsf64.rmsf_nm.dtype == np.float64

        set_default_dtype(np.float32)
        rmsf32 = compute_rmsf(protein_like_trajectory, atom_selection="name CA")
        assert rmsf32.rmsf_nm.dtype == np.float32
    finally:
        set_default_dtype(saved)


# ---------------------------------------------------------------------------
# Helper: run the same compute function at fp32 and fp64 and compare
# ---------------------------------------------------------------------------


def _max_abs(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Maximum absolute difference, ignoring NaNs."""
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if diff.size == 0:
        return 0.0
    return float(np.nanmax(diff))


# ---------------------------------------------------------------------------
# Sensitive metrics: large-N reductions
# ---------------------------------------------------------------------------


def test_rmsf_fp32_matches_fp64(protein_like_trajectory) -> None:
    """RMSF squared-displacement reduction over many frames.

    Expected agreement: ~1e-5 nm.  ``_dtype.py`` documents this as the
    empirical worst-case for protein-scale RMSF.  A regression below
    this bound would indicate accumulator drift -- e.g. someone
    replacing ``np.mean`` with a manual loop.
    """
    fp32 = compute_rmsf(protein_like_trajectory, atom_selection="name CA", dtype=np.float32)
    fp64 = compute_rmsf(protein_like_trajectory, atom_selection="name CA", dtype=np.float64)

    assert fp32.rmsf_nm.dtype == np.float32
    assert fp64.rmsf_nm.dtype == np.float64
    assert _max_abs(fp32.rmsf_nm, fp64.rmsf_nm) < 1e-5


def test_dccm_correlation_fp32_matches_fp64(protein_like_trajectory) -> None:
    """DCCM correlation (covariance + outer-std normalization).

    Expected agreement: ~4e-6 on the correlation matrix.  Correlations
    are bounded in [-1, 1] so any drift larger than this would be
    biologically meaningless but is still worth catching as a
    numerical regression.
    """
    fp32 = compute_dccm(protein_like_trajectory, atom_selection="name CA", dtype=np.float32)
    fp64 = compute_dccm(protein_like_trajectory, atom_selection="name CA", dtype=np.float64)

    assert fp32.correlation.dtype == np.float32
    assert fp64.correlation.dtype == np.float64
    assert _max_abs(fp32.correlation, fp64.correlation) < 5e-6


def test_dccm_numba_backend_fp32_matches_fp64(protein_like_trajectory) -> None:
    """Numba DCCM kernel agrees at machine-epsilon between fp32/fp64 outputs.

    The covariance kernel runs in fp64 internally; the only fp32/fp64
    difference visible to the user comes from the wrapper's final cast.
    Agreement is therefore at the fp32-quantisation level (~6e-8 in
    absolute terms for correlations bounded in [-1, 1]).
    """
    fp32 = compute_dccm(
        protein_like_trajectory,
        atom_selection="name CA",
        backend="numba",
        dtype=np.float32,
    )
    fp64 = compute_dccm(
        protein_like_trajectory,
        atom_selection="name CA",
        backend="numba",
        dtype=np.float64,
    )
    assert _max_abs(fp32.correlation, fp64.correlation) < 1e-6


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------


def test_rmsd_fp32_matches_fp64(protein_like_trajectory) -> None:
    """RMSD via mdtraj's optimised C kernel.

    mdtraj returns float32 natively so the ``dtype=np.float64`` path
    casts up at the wrapper.  The expected difference is therefore
    purely the fp32->fp64 cast precision: ~1e-7 nm on RMSD values in
    the 0-2 nm range.
    """
    fp32 = compute_rmsd(protein_like_trajectory, atom_selection="name CA", dtype=np.float32)
    fp64 = compute_rmsd(protein_like_trajectory, atom_selection="name CA", dtype=np.float64)
    assert fp32.rmsd_nm.dtype == np.float32
    assert fp64.rmsd_nm.dtype == np.float64
    assert _max_abs(fp32.rmsd_nm, fp64.rmsd_nm) < 1e-6


def test_radius_of_gyration_fp32_matches_fp64(protein_like_trajectory) -> None:
    """Rg uses mdtraj's compute_rg which returns fp32; cast precision only."""
    fp32 = compute_radius_of_gyration(
        protein_like_trajectory, atom_selection="all", dtype=np.float32
    )
    fp64 = compute_radius_of_gyration(
        protein_like_trajectory, atom_selection="all", dtype=np.float64
    )
    assert _max_abs(fp32.radius_gyration_nm, fp64.radius_gyration_nm) < 1e-6


def test_sasa_fp32_matches_fp64(protein_like_trajectory) -> None:
    """SASA via Shrake-Rupley round-trips through fp32/fp64 unchanged.

    mdtraj's kernel runs in float32; the output cast preserves precision
    to ~1e-6 nm^2 on per-residue values that are typically 0.1-1.0 nm^2.
    """
    fp32 = compute_sasa(protein_like_trajectory, atom_selection=None, dtype=np.float32)
    fp64 = compute_sasa(protein_like_trajectory, atom_selection=None, dtype=np.float64)
    assert _max_abs(fp32.values_nm2, fp64.values_nm2) < 1e-6


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


@pytest.fixture()
def distance_pairs(protein_like_trajectory) -> NDArray[np.int_]:
    """Build a stable set of CA-CA pairs for distance dtype tests."""
    ca_indices = protein_like_trajectory.topology.select("name CA")
    return np.array(
        [
            (int(ca_indices[i]), int(ca_indices[j]))
            for i in range(len(ca_indices) - 1)
            for j in range(i + 1, len(ca_indices))
        ],
        dtype=np.int_,
    )


def test_distances_mdtraj_fp32_matches_fp64(protein_like_trajectory, distance_pairs) -> None:
    """Mdtraj distances are fp32 native; cast precision only."""
    fp32 = compute_distances(
        protein_like_trajectory,
        atom_pairs=distance_pairs,
        backend="mdtraj",
        periodic=False,
        dtype=np.float32,
    )
    fp64 = compute_distances(
        protein_like_trajectory,
        atom_pairs=distance_pairs,
        backend="mdtraj",
        periodic=False,
        dtype=np.float64,
    )
    assert _max_abs(fp32.distances_nm, fp64.distances_nm) < 1e-6


def test_distances_numba_fp32_matches_fp64(protein_like_trajectory, distance_pairs) -> None:
    """Numba distance kernel agrees between fp32 and fp64 outputs.

    The kernel computes in fp64 internally (via ``float()`` promotion)
    and stores fp32; the difference between fp32/fp64 outputs must not
    exceed the float32 storage quantisation.
    """
    fp32 = compute_distances(
        protein_like_trajectory,
        atom_pairs=distance_pairs,
        backend="numba",
        dtype=np.float32,
    )
    fp64 = compute_distances(
        protein_like_trajectory,
        atom_pairs=distance_pairs,
        backend="numba",
        dtype=np.float64,
    )
    assert _max_abs(fp32.distances_nm, fp64.distances_nm) < 1e-6


def test_minimum_distance_fp32_matches_fp64(protein_like_trajectory) -> None:
    """Minimum distance ``argmin`` picks the same pair under fp32 and fp64.

    ``argmin`` over fp32 distances is robust as long as the underlying
    pairs are well-separated.  We verify the picked pair is identical
    across dtypes and the value agrees within float32 resolution.
    """
    fp32 = compute_minimum_distance(
        protein_like_trajectory,
        group1="name N",
        group2="name O",
        periodic=False,
        dtype=np.float32,
    )
    fp64 = compute_minimum_distance(
        protein_like_trajectory,
        group1="name N",
        group2="name O",
        periodic=False,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(fp32.atom_pairs, fp64.atom_pairs)
    assert _max_abs(fp32.distances_nm, fp64.distances_nm) < 1e-6


# ---------------------------------------------------------------------------
# Decomposition (PCA / TICA)
# ---------------------------------------------------------------------------


def test_pca_fp32_matches_fp64_up_to_sign(protein_like_trajectory) -> None:
    """PCA projections match between fp32 and fp64 modulo per-component sign.

    PCA is well-defined only up to a per-component sign flip, so we
    compare ``abs(projections)`` instead of raw values.  With sklearn
    >= 1.8 the input dtype is preserved end-to-end so the difference
    is purely fp32 quantisation on the eigen-decomposition output.
    """
    features = featurize_ca_distances(protein_like_trajectory, backend="mdtraj")
    fp32 = compute_pca(features.values.astype(np.float32), n_components=2, dtype=np.float32)
    fp64 = compute_pca(features.values.astype(np.float64), n_components=2, dtype=np.float64)
    assert _max_abs(np.abs(fp32.projections), np.abs(fp64.projections)) < 5e-4
    assert _max_abs(fp32.explained_variance_ratio, fp64.explained_variance_ratio) < 5e-5


def test_tica_dtype_only_affects_output_cast(small_protein_trajectory) -> None:
    """TICA projections agree at fp32-cast level regardless of input dtype.

    Deeptime upcasts to fp64 internally for covariance estimation, so
    the only fp32/fp64 difference visible to the user is the wrapper's
    final cast on the projection matrix.
    """
    features = featurize_ca_distances(small_protein_trajectory, backend="mdtraj")
    fp32 = compute_tica(
        features.values.astype(np.float32),
        lagtime=2,
        n_components=2,
        dtype=np.float32,
    )
    fp64 = compute_tica(
        features.values.astype(np.float64),
        lagtime=2,
        n_components=2,
        dtype=np.float64,
    )
    # TICA components are also defined up to sign; compare absolute values.
    assert _max_abs(np.abs(fp32.projections), np.abs(fp64.projections)) < 5e-4


# ---------------------------------------------------------------------------
# Free energy surface
# ---------------------------------------------------------------------------


def test_fes_2d_fp32_matches_fp64() -> None:
    """FES energies agree across dtypes despite the ``-RT log p`` sensitivity.

    ``-RT log p`` is sensitive when ``p`` is small, but ``np.histogram2d``
    runs in fp64 internally so the energy values themselves agree to
    machine precision; only the final output cast differs.  Tolerance
    is <1e-3 kJ/mol on energies typically spanning 0-15 kJ/mol.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=20_000).astype(np.float32)
    y = rng.normal(size=20_000).astype(np.float32)

    fp32 = compute_fes_2d(x, y, bins=40, dtype=np.float32)
    fp64 = compute_fes_2d(x.astype(np.float64), y.astype(np.float64), bins=40, dtype=np.float64)

    # Compare only finite, sampled bins -- masked regions are NaN by design.
    mask = fp32.observed_mask & fp64.observed_mask
    diff = np.abs(fp32.free_energy_kj_mol[mask].astype(np.float64) - fp64.free_energy_kj_mol[mask])
    assert float(diff.max()) < 1e-3


# ---------------------------------------------------------------------------
# Multi-replica reductions
# ---------------------------------------------------------------------------


def test_average_rmsf_with_sem_fp32_matches_fp64() -> None:
    """Three-replica RMSF averaging in MSF space + SEM propagation.

    The square -> mean -> sqrt chain is the classic case where fp32
    underflow on tiny variances would matter.  We use four replicas
    with realistic CA fluctuations so the SEM denominator
    ``2 * avg_rmsf`` stays well above zero.
    """
    rmsf_results = []
    for seed in (3, 7, 11, 19):
        traj = _make_protein_like_trajectory(seed=seed)
        rmsf_results.append(compute_rmsf(traj, atom_selection="name CA", dtype=np.float32))
    avg32, sem32 = average_rmsf_with_sem(rmsf_results, dtype=np.float32)

    rmsf_results_64 = []
    for seed in (3, 7, 11, 19):
        traj = _make_protein_like_trajectory(seed=seed)
        rmsf_results_64.append(compute_rmsf(traj, atom_selection="name CA", dtype=np.float64))
    avg64, sem64 = average_rmsf_with_sem(rmsf_results_64, dtype=np.float64)

    assert _max_abs(avg32, avg64) < 1e-5
    assert sem32 is not None and sem64 is not None
    assert _max_abs(sem32, sem64) < 1e-5


def test_delta_rmsf_fp32_matches_fp64() -> None:
    """End-to-end delta-RMSF (B - A) with two replicas per system."""
    a32 = [
        compute_rmsf(
            _make_protein_like_trajectory(seed=s), atom_selection="name CA", dtype=np.float32
        )
        for s in (1, 2)
    ]
    b32 = [
        compute_rmsf(
            _make_protein_like_trajectory(seed=s), atom_selection="name CA", dtype=np.float32
        )
        for s in (4, 5)
    ]
    a64 = [
        compute_rmsf(
            _make_protein_like_trajectory(seed=s), atom_selection="name CA", dtype=np.float64
        )
        for s in (1, 2)
    ]
    b64 = [
        compute_rmsf(
            _make_protein_like_trajectory(seed=s), atom_selection="name CA", dtype=np.float64
        )
        for s in (4, 5)
    ]
    delta32 = compute_delta_rmsf(a32, b32, dtype=np.float32)
    delta64 = compute_delta_rmsf(a64, b64, dtype=np.float64)

    assert _max_abs(delta32.delta_rmsf_nm, delta64.delta_rmsf_nm) < 1e-5
    assert delta32.sem_nm is not None and delta64.sem_nm is not None
    assert _max_abs(delta32.sem_nm, delta64.sem_nm) < 1e-5


# ---------------------------------------------------------------------------
# Contacts and h-bonds (boolean reductions cast to floating)
# ---------------------------------------------------------------------------


def test_contact_frequency_fp32_matches_fp64(protein_like_trajectory) -> None:
    """Contact frequencies agree across dtypes after the boolean-mean cast.

    ``np.mean(distances < cutoff)`` on a boolean array returns fp64
    natively; the wrapper's cast back to ``resolved`` must not introduce
    drift.
    """
    fp32, _ = compute_contact_frequency(protein_like_trajectory, cutoff_nm=0.6, dtype=np.float32)
    fp64, _ = compute_contact_frequency(protein_like_trajectory, cutoff_nm=0.6, dtype=np.float64)
    assert _max_abs(fp32, fp64) < 1e-7


def test_compute_contacts_fp32_matches_fp64(protein_like_trajectory) -> None:
    """Per-frame contact distances round-trip through fp32/fp64 unchanged.

    mdtraj's ``compute_contacts`` returns fp32 natively, so the only
    fp32/fp64 difference is the wrapper cast.
    """
    fp32 = compute_contacts(protein_like_trajectory, periodic=False, dtype=np.float32)
    fp64 = compute_contacts(protein_like_trajectory, periodic=False, dtype=np.float64)
    assert _max_abs(fp32.distances_nm, fp64.distances_nm) < 1e-6


def test_native_contacts_fp32_matches_fp64(protein_like_trajectory) -> None:
    """Native-contact fraction Q(t) = mean over a boolean mask."""
    fp32 = compute_native_contacts(
        protein_like_trajectory, cutoff_nm=0.6, periodic=False, dtype=np.float32
    )
    fp64 = compute_native_contacts(
        protein_like_trajectory, cutoff_nm=0.6, periodic=False, dtype=np.float64
    )
    assert _max_abs(fp32.fraction, fp64.fraction) < 1e-7


def test_hbond_occupancy_fp32_matches_fp64(hbond_trajectory) -> None:
    """Hydrogen-bond occupancy is mean(presence) on a boolean array.

    Occupancy values are exact rationals (k/n_frames), so fp32 and fp64
    should agree to bit precision when n_frames is small.
    """
    fp32 = compute_hbonds(
        hbond_trajectory,
        method="wernet_nilsson",
        exclude_water=False,
        periodic=False,
        dtype=np.float32,
    )
    fp64 = compute_hbonds(
        hbond_trajectory,
        method="wernet_nilsson",
        exclude_water=False,
        periodic=False,
        dtype=np.float64,
    )
    assert _max_abs(fp32.occupancy, fp64.occupancy) < 1e-7


# ---------------------------------------------------------------------------
# RMSD matrix and clustering
# ---------------------------------------------------------------------------


def test_rmsd_matrix_mdtraj_fp32_matches_fp64(small_protein_trajectory) -> None:
    """Mdtraj RMSD matrix is fp32 native; cast precision only."""
    fp32 = compute_rmsd_matrix(
        small_protein_trajectory, atom_selection="name CA", backend="mdtraj", dtype=np.float32
    )
    fp64 = compute_rmsd_matrix(
        small_protein_trajectory, atom_selection="name CA", backend="mdtraj", dtype=np.float64
    )
    assert _max_abs(fp32.rmsd_matrix_nm, fp64.rmsd_matrix_nm) < 1e-6


def test_rmsd_matrix_numba_fp32_matches_fp64(small_protein_trajectory) -> None:
    """Numba QCP runs in fp64 internally then stores fp32; cast precision only."""
    fp32 = compute_rmsd_matrix(
        small_protein_trajectory, atom_selection="name CA", backend="numba", dtype=np.float32
    )
    fp64 = compute_rmsd_matrix(
        small_protein_trajectory, atom_selection="name CA", backend="numba", dtype=np.float64
    )
    assert _max_abs(fp32.rmsd_matrix_nm, fp64.rmsd_matrix_nm) < 1e-6


def test_rmsd_matrix_numba_vs_mdtraj_at_fp32() -> None:
    """Cross-backend agreement at the package default dtype.

    The numba kernel uses fp64 QCP accumulators while mdtraj uses its
    own optimised fp32 SSE superposition.  On a low-noise fixture
    (per-frame perturbations ~0.02 nm matching the existing
    ``backbone_trajectory`` test setup) both are correct to ~5e-5 nm.

    The agreement loosens with noise because float32 cross-covariance
    accumulation diverges between the SSE and QCP paths once the
    cross-correlation magnitude grows; ``test_clustering.py`` uses the
    same 5e-5 tolerance for the same reason.
    """
    rng = np.random.RandomState(42)
    n_frames = 30
    n_atoms = 15
    base = rng.randn(1, n_atoms, 3).astype(np.float32) * 0.15
    perturbation = rng.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.02
    xyz = base + perturbation

    topology = md.Topology()
    chain = topology.add_chain()
    for resseq in range(1, n_atoms + 1):
        residue = topology.add_residue("ALA", chain, resSeq=resseq)
        topology.add_atom("CA", md.element.carbon, residue)
    traj = md.Trajectory(xyz=xyz, topology=topology)

    numba = compute_rmsd_matrix(traj, atom_selection="name CA", backend="numba")
    mdtraj_ref = compute_rmsd_matrix(traj, atom_selection="name CA", backend="mdtraj")
    assert _max_abs(numba.rmsd_matrix_nm, mdtraj_ref.rmsd_matrix_nm) < 5e-5


def test_kmeans_centers_fp32_matches_fp64(protein_like_trajectory) -> None:
    """K-Means cluster centers agree across dtypes for a deterministic fit.

    sklearn's KMeans uses its own internal dtype but the centers we
    expose are cast through ``resolved`` -- the fp32/fp64 difference
    should be limited to the cast itself for a well-converged KMeans
    on the same fixed seed.
    """
    features = featurize_ca_distances(protein_like_trajectory, backend="mdtraj")
    pca = compute_pca(features.values, n_components=3, dtype=np.float32)
    fp32 = KMeans(n_clusters=4, dtype=np.float32)(pca.projections)
    fp64 = KMeans(n_clusters=4, dtype=np.float64)(pca.projections.astype(np.float64))
    # KMeans is deterministic with the seeded random_state so labels match.
    np.testing.assert_array_equal(fp32.labels, fp64.labels)
    # Centers are rounded for fp32 storage so allow a small tolerance.
    assert _max_abs(fp32.cluster_centers, fp64.cluster_centers) < 1e-5


# ---------------------------------------------------------------------------
# Sanity: invalid dtypes are rejected
# ---------------------------------------------------------------------------


def test_compute_rmsd_rejects_invalid_dtype(protein_like_trajectory) -> None:
    """Out-of-scope float dtypes are rejected with a clear ValueError.

    Float16 / float128 are out of scope for this package -- the only
    valid dtypes are float32 and float64.  A typo at the call site
    must surface a clear ValueError instead of silently propagating
    through the pipeline.
    """
    with pytest.raises(ValueError, match="float32 or float64"):
        compute_rmsd(protein_like_trajectory, atom_selection="name CA", dtype=np.float16)
