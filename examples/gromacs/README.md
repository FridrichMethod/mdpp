# GROMACS analysis assumptions

These notebooks are templates for equilibrated, unsmoothed trajectories. Set the paths and `START_FRAME` after assessing equilibration; `START_FRAME` uses raw frame indices before `STRIDE`. Retain the recorded times and state the atom selection, reference, temperature, and independent replica count with any result.

## Coordinates and periodicity

Keep two coordinate representations. Periodic contacts, distances, torsions and hydrogen bonds use an **unfitted** trajectory with its original unit-cell orientation. RMSF and DCCM use whole molecules aligned to a chosen reference. MDTraj/GROMACS rotational fitting does not rotate the stored cell along with coordinates; applying minimum-image geometry afterwards can change distances incorrectly. The [GROMACS preprocessing workflow](https://manual.gromacs.org/current/user-guide/terminology.html#suggested-workflow) places fitting last.

The structural notebooks call MDTraj `make_molecules_whole` when cell information exists. This requires a correct bond topology and reconstructs each bonded molecule; it does not assemble separate protein chains or associate a ligand with its receptor. Inspect multimer imaging and use deliberate assembly groups before fitting. A typical single-molecule preprocessing step is:

```bash
gmx trjconv -s production.tpr -f production.xtc -o whole.xtc -pbc whole
```

Choose the output group interactively and use a topology containing exactly those atoms in the same order. For complexes, adapt the [GROMACS trjconv options](https://manual.gromacs.org/current/onlinehelp/gmx-trjconv.html) to the intended assembly. The repository postprocessors retain `step5_production_complex_center.xtc` for periodic analyses and `step5_production_complex_fit.xtc` for aligned analyses. `*_smoothed.xtc` is for visualization: coordinate smoothing changes fluctuations, populations and kinetics.

## Meaning of each observable

- **RMSD/RMSF/DCCM:** `compute_rmsd` optimally superposes its selected atoms without modifying the input. RMSF and DCCM require prior alignment. Report the fit selection separately from the measurement selection. DCCM measures normalized coordinate correlations, not causal allosteric communication.
- **Radius of gyration:** the Python wrapper uses equal atom weights about the geometric center ([MDTraj](https://mdtraj.readthedocs.io/en/latest/api/generated/mdtraj.compute_rg.html)). `gmx gyrate` defaults to mass weighting; use `-mode geometry` with the same atoms to compare the geometric observable ([GROMACS](https://manual.gromacs.org/current/onlinehelp/gmx-gyrate.html)). A PDB does not preserve simulation-specific HMR masses.
- **SASA:** the selected atoms define both the surface and its occluders. `atom_selection="protein"` measures isolated-protein SASA; it excludes ligand/membrane shielding. The probe radius is 0.14 nm. Do not interpret this as the protein surface in an intact complex without preserving the relevant occluders.
- **Contacts/Q:** the default is a closest-heavy residue distance and a hard cutoff. MDTraj's `contacts="all"` excludes immediate sequence neighbors and inter-chain pairs; specify explicit pairs for an interface. A contact map's uncomputed entries are not evidence of no contact. The first retained frame defines Q's reference; this is “native” only if that frame represents the intended native state ([MDTraj contacts API](https://mdtraj.readthedocs.io/en/latest/api/generated/mdtraj.compute_contacts.html)).
- **Hydrogen bonds:** Baker-Hubbard uses H–acceptor < 0.25 nm and D–H–A > 120 degrees, and requires explicit bonded hydrogens. GROMACS's donor–acceptor criterion is a different observable. Positive `freq` filters the bond set by occupancy; total geometric counts therefore use `freq=0.0` ([MDTraj](https://www.mdtraj.org/1.9.7/api/generated/mdtraj.baker_hubbard.html)).
- **DSSP/Ramachandran:** DSSP needs a complete backbone; missing assignments should not be interpreted as coil. A Ramachandran point pairs phi and psi of the same residue. Terminal residues often have only one angle. Torsion label suffixes are unique zero-based input topology residue indices, not PDB residue sequence numbers.

## Projections, clustering and free energies

The notebooks use centered, unstandardized sin/cos features for conventional dihedral PCA. Independently standardizing sine and cosine changes the angular metric; it is an alternative weighting choice rather than the same dPCA ([GROMACS dPCA](https://manual.gromacs.org/current/reference-manual/analysis/dihedral-pca.html)). Compare systems in one fitted PCA basis using `project_pca`, after verifying feature/residue correspondence; separately fitted axes cannot be compared directly.

TICA's `lagtime` counts **retained** frames. The examples print its physical duration and check equal frame spacing. `compute_tica` accepts one continuous trajectory. Do not concatenate independent replicas or removed time blocks as though transitions connect them; use deeptime's [multiple-trajectory datasets](https://deeptime-ml.github.io/latest/api/generated/deeptime.util.data.TrajectoryDataset.html) if a common kinetic model is required. Assess lag/feature dependence and convergence; a two-dimensional projection alone does not validate kinetic states. Deeptime defaults to [kinetic-map scaling](https://deeptime-ml.github.io/latest/notebooks/tica.html).

RMSD clustering cutoffs are in nm; feature-clustering distances are in feature/projection units. GROMOS, average linkage and DBSCAN define different neighborhoods even at the same numerical threshold. GROMOS's `medoid_frames` field contains its greedy centers, not necessarily distance-minimizing medoids. Cluster populations depend on cutoff, subsampling and equilibrium sampling; a 2D display is only a projection of the space used for clustering. Ward/centroid/median linkage requires Euclidean distances and is rejected for general optimally fitted RMSD matrices ([SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)).

`compute_fes_2d` estimates `-RT ln p(x,y)` from **unweighted** counts, normalizes by bin area and shifts the minimum to zero. It does not reweight umbrella/metadynamics or hot REST2 states. Choose the sampled ensemble's temperature, retain unsampled-bin masks, and compare time blocks/replicas in shared CV axes with shared edges. Separate plots have arbitrary free-energy zeros; minima cannot be compared as absolute stabilities, and projected barriers are not rates. For biased/multistate data use an appropriate reweighting estimator ([Shirts and Ferguson](https://arxiv.org/abs/2001.01170)).

Check equilibration, sampling, block stability and independent replica agreement before interpreting populations or small differences. Stored frames are correlated; their count is not an independent sample count. Delta-RMSF's propagated SEM assumes independent replicas and independent systems, with equal replica weight, and is not a per-frame SEM or an exact confidence interval ([Grossfield et al.](https://www.nist.gov/publications/best-practices-quantification-uncertainty-and-sampling-quality-molecular-simulations)).
