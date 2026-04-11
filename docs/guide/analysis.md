# Analysis

The `mdpp.analysis` subpackage provides trajectory analysis functions. All compute functions return frozen dataclass results and follow consistent patterns.

## Float Dtype Control

All `compute_*` functions default to **float32**, matching the precision of
MD trajectory coordinates stored by mdtraj. You can override the dtype
globally or per-function call:

```python
import numpy as np
import mdpp

# Check the current default
print(mdpp.get_default_dtype())  # float32

# Override globally
mdpp.set_default_dtype(np.float64)

# Override per-function (takes precedence over the global default)
result = compute_rmsd(traj, dtype=np.float64)

# Reset to float32
mdpp.set_default_dtype(np.float32)
```

Float32 is sufficient for all analysis operations in this package.
MD trajectories are stored in float32 by mdtraj, so there is no extra
precision to recover from float64. Empirical tests confirm negligible
differences: RMSF error ~1e-5 nm, DCCM correlation error ~4e-6, FES
error ~2e-6 kJ/mol -- all well below any physically meaningful threshold.

## RMSD

```python
from mdpp.core import load_trajectory
from mdpp.analysis.metrics import compute_rmsd

traj = load_trajectory("md.xtc", topology_path="topol.gro")
result = compute_rmsd(traj, atom_selection="backbone")

print(result.rmsd_angstrom.mean())  # average RMSD in Å
print(result.time_ns[-1])           # simulation length in ns
```

## RMSF

```python
from mdpp.analysis.metrics import compute_rmsf

result = compute_rmsf(traj, atom_selection="name CA")
# result.rmsf_angstrom: per-residue RMSF
# result.residue_ids: corresponding residue IDs
```

## Delta-RMSF

Compare flexibility between two systems (e.g. apo vs holo):

```python
from mdpp.analysis.metrics import compute_rmsf, compute_delta_rmsf

rmsf_apo = [compute_rmsf(traj) for traj in apo_replicas]
rmsf_holo = [compute_rmsf(traj) for traj in holo_replicas]
delta = compute_delta_rmsf(rmsf_holo, rmsf_apo, compute_sem=True, name_a="holo", name_b="apo")
# delta.delta_rmsf_angstrom: per-residue flexibility change
```

## Dynamic Cross-Correlation Matrix (DCCM)

```python
from mdpp.analysis.metrics import compute_dccm

result = compute_dccm(traj, atom_selection="name CA")
# result.correlation: (n_atoms, n_atoms) correlation matrix
```

## Solvent-Accessible Surface Area (SASA)

```python
from mdpp.analysis.metrics import compute_sasa

result = compute_sasa(traj, atom_selection="protein", mode="residue")
print(result.total_nm2.mean())  # average total SASA
```

## Radius of Gyration

```python
from mdpp.analysis.metrics import compute_radius_of_gyration

result = compute_radius_of_gyration(traj)
# result.radius_gyration_angstrom: Rg per frame in Angstrom
```

## Hydrogen Bonds

```python
from mdpp.analysis.hbond import compute_hbonds, format_hbond_triplets

result = compute_hbonds(traj)  # default method: baker_hubbard
labels = format_hbond_triplets(traj.topology, result.triplets)
# result.occupancy: fraction of frames each bond is present
```

## Contacts

### Inter-residue contacts

```python
from mdpp.analysis.contacts import compute_contacts

result = compute_contacts(traj, scheme="closest-heavy")
# result.distances_nm: (n_frames, n_pairs) distance matrix
```

### Contact frequency matrix

```python
from mdpp.analysis.contacts import compute_contact_frequency

frequency, pairs = compute_contact_frequency(traj, cutoff_nm=0.45, scheme="closest-heavy")
```

### Native contacts (Q value)

```python
from mdpp.analysis.contacts import compute_native_contacts

result = compute_native_contacts(traj, reference_frame=0, cutoff_nm=0.45, scheme="closest-heavy")
# result.fraction: Q(t) per frame
```

## Pairwise Distances

```python
from mdpp.analysis.distance import compute_distances
import numpy as np

pairs = np.array([[0, 100], [50, 200]])
result = compute_distances(traj, atom_pairs=pairs)
# result.distances_angstrom: (n_frames, 2)
```

### Minimum distance between groups

```python
from mdpp.analysis.distance import compute_minimum_distance

result = compute_minimum_distance(traj, group1="resid 10", group2="resid 50")
```

### Compute backend selection

`compute_distances`, `compute_minimum_distance`, and `featurize_ca_distances`
accept a `backend=` argument. Five backends are available:

| Backend | Device | PBC | Install |
| ---------- | --------------- | --- | ------------------------- |
| `"mdtraj"` | CPU (1 thread) | yes | built-in (default) |
| `"numba"` | CPU (all cores) | no | built-in (numba) |
| `"cupy"` | NVIDIA GPU | no | `pip install -e ".[gpu]"` |
| `"torch"` | CUDA GPU / CPU | no | `pip install -e ".[gpu]"` |
| `"jax"` | GPU / TPU / CPU | no | `pip install -e ".[gpu]"` |

**Default is `"mdtraj"` across every analysis function** for API
consistency and correctness (only mdtraj supports periodic boundary
conditions). Opt in to faster backends explicitly when performance
matters:

```python
# Default -- mdtraj (supports periodic boundary conditions)
result = compute_distances(traj, atom_pairs=pairs, periodic=True)

# Numba: 5-10x faster than mdtraj on multi-core CPU (no PBC)
result = compute_distances(traj, atom_pairs=pairs, backend="numba", periodic=False)

# GPU backends for very large trajectories
result = compute_distances(traj, atom_pairs=pairs, backend="cupy", periodic=False)
```

> **Note:** Only the `mdtraj` backend supports periodic boundary
> conditions. For the other four backends the `periodic` flag is
> silently ignored.

#### Cached GPU memory is released automatically

PyTorch, CuPy, and JAX all use caching memory allocators that hold
GPU blocks in a pool so subsequent calls can reuse them without
expensive CUDA `malloc`/`free` round-trips. **mdpp decorates every
GPU-backed kernel with a framework-specific cleanup decorator**, so
pooled memory is returned to the CUDA driver in a `finally` block
as soon as the kernel returns (even on exceptions):

```python
# Applied internally in mdpp.analysis._backends:
@clean_torch_cache
def rmsd_torch(...): ...

@clean_cupy_cache
def distances_cupy(...): ...
```

In practice, `nvidia-smi` reflects the release automatically after
every `compute_rmsd_matrix` / `compute_distances` /
`featurize_ca_distances` call that uses a GPU backend -- no manual
cleanup is needed. If you interleave mdpp with raw
`torch`/`cupy`/`jax` code and need to force a release across
frameworks, use the native APIs:

```python
import torch
import cupy as cp

torch.cuda.empty_cache()
cp.get_default_memory_pool().free_all_blocks()
```

## Secondary Structure (DSSP)

```python
from mdpp.analysis.dssp import compute_dssp

result = compute_dssp(traj, simplified=True)
# result.assignments: (n_frames, n_residues) with "H", "E", "C"
# result.frequency: (n_residues, 3) fraction in each state
```

## PCA / TICA

### Backbone torsion featurization

```python
from mdpp.analysis.decomposition import featurize_backbone_torsions, compute_pca

features = featurize_backbone_torsions(traj, sincos_embedding=True)
pca_result = compute_pca(features.values, n_components=2)
```

### Project new data onto fitted PCA

```python
from mdpp.analysis.decomposition import project_pca

new_features = featurize_backbone_torsions(new_traj, sincos_embedding=True)
projected = project_pca(new_features.values, fitted=pca_result)
```

### TICA (time-lagged independent component analysis)

```python
from mdpp.analysis.decomposition import compute_tica

tica_result = compute_tica(features.values, lagtime=10, n_components=2)
```

## Free Energy Surface

```python
from mdpp.analysis.fes import compute_fes_from_projection

fes = compute_fes_from_projection(pca_result.projections)
# fes.free_energy_kj_mol: 2D free energy in kJ/mol (default T=298.15 K)
```

## Conformational Clustering

```python
from mdpp.analysis.clustering import compute_rmsd_matrix, cluster_conformations

rmsd_mat = compute_rmsd_matrix(traj, atom_selection="backbone")
clusters = cluster_conformations(rmsd_mat.rmsd_matrix_nm, cutoff_nm=0.15)
print(f"Found {clusters.n_clusters} clusters")
print(f"Medoid frames: {clusters.medoid_frames}")
```

### RMSD matrix backend selection

`compute_rmsd_matrix` defaults to the **mdtraj** backend for API
consistency with other analysis functions. Five backends are
available:

| Backend | Method | Device |
| ---------- | ------------------------------- | --------------- |
| `"mdtraj"` | Precentered `md.rmsd` loop | CPU (1 thread) |
| `"numba"` | QCP + Newton-Raphson | CPU (all cores) |
| `"torch"` | Vectorised einsum + batched SVD | CUDA / CPU |
| `"jax"` | Vectorised einsum + batched SVD | GPU / TPU / CPU |
| `"cupy"` | Vectorised einsum + batched SVD | NVIDIA GPU |

```python
# Default -- mdtraj
rmsd_mat = compute_rmsd_matrix(traj, atom_selection="backbone")

# Numba QCP: 50-200x faster than mdtraj on multi-core CPU
rmsd_mat = compute_rmsd_matrix(traj, atom_selection="backbone", backend="numba")

# GPU backend for very large trajectories (>1000 frames)
rmsd_mat = compute_rmsd_matrix(traj, atom_selection="backbone", backend="torch")
```

All backends agree to within ~5e-5 nm and produce symmetric matrices
with a zero diagonal. GPU backends (`torch`, `jax`, `cupy`) require
the optional `[gpu]` extra.
