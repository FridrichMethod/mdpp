# Standard FEP+ RBFE: one conformation, many ligands

This is the one-conformation FEP+ workflow corresponding to the analysis
pattern in `examples/openfe/rbfe.ipynb`. It runs one receptor conformation and
a shared multi-ligand perturbation graph. The default receptor is `open`; set
`FEPP_CONFORMATION=closed` to use the closed input instead.

The versioned ligands, poses, and receptors are a one-state projection of the
shared cohort constructed in `examples/openfe/rbfe_open_closed.ipynb`, not an
exact input match to the standalone OpenFE notebook. The standalone notebook
always uses open-complex ligand poses, whereas the paired cohort selects
closed-complex poses for Ch5 and M5. Keep this distinction explicit in any
OpenFE-versus-FEP+ comparison.

## Estimand

For an explicit reference ligand `r`, the analyzer reports

```math
\Delta\Delta G_{i,r}^{(s)}=
\Delta G_{\mathrm{bind}}^{(s)}(i)-
\Delta G_{\mathrm{bind}}^{(s)}(r).
```

A negative value predicts stronger binding than the reference. The reference
row is `0 +/- 0` because it fixes the graph's additive gauge; it is not an
absolute binding free energy. The complete covariance and every pairwise
contrast are exported so conclusions do not depend on a convenient reference.

## Build and run

From this directory:

```bash
export SCHRODINGER=/apps/schrodinger2025-4

./build_inputs.sh
./build_inputs.sh --check
./run_fep_plus.sh --prepare
./run_fep_plus.sh --repeats 3 --seed-base 2014
```

For the closed receptor instead:

```bash
FEPP_CONFORMATION=closed ./build_inputs.sh
FEPP_CONFORMATION=closed ./run_fep_plus.sh --repeats 3 --seed-base 2014
```

The standalone launcher checks only the selected receptor and snapshots
`input_state_inputs.tsv`; it neither requires nor fabricates the paired
receptor-microstate and `paired_inputs.tsv` artifacts.

## Extract and analyze

Normalize each completed repeat with the exact launch manifest:

```bash
$SCHRODINGER/run python3 ../extract_fep_results.py \
  ../tmp/runs/fepp_rbfe_open_r01/fepp_rbfe_open_r01_out.fmp \
  --manifest ../tmp/runs/fepp_rbfe_open_r01/manifest.tsv \
  --state open -o ../tmp/results/open_r01.csv
```

Then fit all independent repeats:

```bash
conda run -n mdpp python3 ../analyze_rbfe_results.py \
  --input ../tmp/results/open_r01.csv \
  --input ../tmp/results/open_r02.csv \
  --input ../tmp/results/open_r03.csv \
  --state open --reference LA_AMP \
  --output-dir ../tmp/results/open_rbfe
```

The result bundle contains `analysis.json`, `relative_free_energies.csv`,
`pairwise_contrasts.csv`, `edge_residuals.csv`, `cycles.csv`, and an atomic
completion marker. The fit uses raw Bennett edge values, propagates the full
node covariance, inflates pooled edge uncertainty when repeats disagree, and
reports vendor QC without automatically deleting edges.

## State-definition caveat

An initial open or closed structure does not by itself define a thermodynamic
receptor state. Verify receptor CV/RMSD basin occupancy, or use explicit
state-defining restraints with the required free-energy corrections. Otherwise
describe this result as initial-conformation- and protocol-conditioned RBFE.
The full shared assumptions and protocol controls are documented in
[`../README.md`](../README.md).
