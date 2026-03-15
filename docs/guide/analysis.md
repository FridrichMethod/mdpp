# Analysis

The `mdpp.analysis` subpackage provides trajectory analysis functions. All compute functions return frozen dataclass results and follow consistent patterns.

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

result = compute_radius_of_gyration(traj, atom_selection="protein")
# result.radius_gyration_angstrom: Rg per frame
```

## Hydrogen Bonds

```python
from mdpp.analysis.hbond import compute_hbonds, format_hbond_triplets

result = compute_hbonds(traj, method="baker_hubbard")
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

frequency, pairs = compute_contact_frequency(traj, cutoff_nm=0.45)
```

### Native contacts (Q value)

```python
from mdpp.analysis.contacts import compute_native_contacts

result = compute_native_contacts(traj, reference_frame=0, cutoff_nm=0.45)
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

features = featurize_backbone_torsions(traj, periodic=True)
pca_result = compute_pca(features.values, n_components=2)
```

### TICA (time-lagged independent component analysis)

```python
from mdpp.analysis.decomposition import compute_tica

tica_result = compute_tica(features.values, lagtime=10, n_components=2)
```

## Free Energy Surface

```python
from mdpp.analysis.fes import compute_fes_from_projection

fes = compute_fes_from_projection(pca_result.projections, temperature_k=300.0)
# fes.free_energy_kj_mol: 2D free energy in kJ/mol
```

## Conformational Clustering

```python
from mdpp.analysis.clustering import compute_rmsd_matrix, cluster_conformations

rmsd_mat = compute_rmsd_matrix(traj, atom_selection="backbone")
clusters = cluster_conformations(rmsd_mat.rmsd_matrix_nm, cutoff_nm=0.15)
print(f"Found {clusters.n_clusters} clusters")
print(f"Medoid frames: {clusters.medoid_frames}")
```
