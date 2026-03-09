# Notebook Guide

These notebooks are split by `mdpp` analysis classes so the pipeline logic is explicit.

1. `io_preprocessing.ipynb`
   - load trajectories, select atoms, align, and set up reusable inputs
1. `rmsd_rmsf.ipynb`
   - structure metrics: RMSD, RMSF, SASA, radius of gyration
1. `dccm.ipynb`
   - correlated motion and hydrogen-bond dynamics
1. `fes.ipynb`
   - torsion featurization, PCA/TICA projections, and free-energy surfaces

Plot styling in all notebooks:

- `plt.style.use("mplplots.styles.GraphPadPrism")`
- `from mplplots.utils import auto_ticks`
