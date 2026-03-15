# Plots

The `mdpp.plots` subpackage provides visualization functions. All plot functions accept an optional `ax` parameter and return a matplotlib `Axes` object.

## Time Series

### RMSD, RMSF, SASA, Radius of Gyration

```python
from mdpp.plots import plot_rmsd, plot_rmsf, plot_sasa, plot_radius_of_gyration

ax = plot_rmsd(rmsd_result, label="backbone")
ax = plot_rmsf(rmsf_result)
ax = plot_sasa(sasa_result, aggregate="sum")
ax = plot_radius_of_gyration(rg_result)
```

### Hydrogen bond counts and occupancy

```python
from mdpp.plots import plot_hbond_counts, plot_hbond_occupancy

ax = plot_hbond_counts(hbond_result)
ax = plot_hbond_occupancy(hbond_result, top_n=10, labels=hbond_labels)
```

### Distance time series

```python
from mdpp.plots import plot_distances

ax = plot_distances(distance_result, pair_labels=["CA10-CA50", "CA20-CA80"])
```

### Native contacts Q(t)

```python
from mdpp.plots import plot_native_contacts

ax = plot_native_contacts(native_contact_result, label="Q(t)")
```

### Energy plots

Plot energy terms from parsed GROMACS output:

```python
from mdpp.core import read_xvg
from mdpp.plots import plot_energy

df = read_xvg("energy.xvg")
ax = plot_energy(df, columns=["Potential", "Kinetic En."])
```

## Matrix Plots

### DCCM heatmap

```python
from mdpp.plots import plot_dccm

ax = plot_dccm(dccm_result, cmap="RdBu_r")
```

### Contact map

```python
from mdpp.plots import plot_contact_map, contact_frequency_to_matrix

frequency, pairs = compute_contact_frequency(traj)
matrix = contact_frequency_to_matrix(frequency, pairs, n_residues=traj.n_residues)
ax = plot_contact_map(matrix)
```

## Free Energy Surface

```python
from mdpp.plots import plot_fes

ax = plot_fes(fes_result, cmap="coolwarm", contour_levels=15)
```

## Scatter Plots

### PCA / TICA projection

```python
from mdpp.plots import plot_projection
import numpy as np

time = np.arange(len(pca_result.projections))
ax = plot_projection(pca_result, color_by=time, cmap="coolwarm")
```

### Ramachandran plot

```python
from mdpp.analysis.decomposition import featurize_backbone_torsions
from mdpp.plots import plot_ramachandran

torsions = featurize_backbone_torsions(traj, periodic=False)
ax = plot_ramachandran(torsions)
```

## Multi-Panel Figures

All plot functions work with matplotlib subplots:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_rmsd(rmsd_result, ax=axes[0, 0])
plot_rmsf(rmsf_result, ax=axes[0, 1])
plot_dccm(dccm_result, ax=axes[1, 0])
plot_fes(fes_result, ax=axes[1, 1])
fig.tight_layout()
```
