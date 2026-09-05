# FEP+ RBFE for matched open and closed conformations

This is the native FEP+ counterpart of
`examples/openfe/rbfe_open_closed.ipynb`. It runs the same ligand graph and atom
mappings against prepared open and closed receptors, then forms a controlled
double difference.

## Estimand

With explicit reference ligand `r`, the identifiable unanchored result is

```math
D_i^{(r)}=
[G_{\mathrm{open},i}-G_{\mathrm{open},r}]
-[G_{\mathrm{closed},i}-G_{\mathrm{closed},r}].
```

Negative values mean ligand `i` is more open-selective than `r`; positive
values mean more closed-selective. This is reference-relative selectivity, not
an absolute open-versus-closed binding preference. An independent reference
anchor is required for the latter.

## Build and run

From this directory:

```bash
export SCHRODINGER=/apps/schrodinger2025-4

./build_inputs.sh
./build_inputs.sh --check
./run_fep_plus.sh --prepare
./run_fep_plus.sh --repeats 3 --seed-base 2014
```

The paired builder requires both receptors, enforces the declared shared
His267 state by default, compares receptor microstates, and requires identical
ligand edge lists and coordinate-independent atom-mapping fingerprints. Every
run snapshots the shared `paired_inputs.tsv` cohort and receptor-microstate
report.

## Extract and analyze

Normalize each open and closed repeat separately:

```bash
conda run -n mdpp python3 ../extract_fep_results.py \
  ../tmp/runs/fepp_open_closed_open_r01/fepp_open_closed_open_r01_out.fmp \
  --state open -o ../tmp/results/open_r01.csv

conda run -n mdpp python3 ../extract_fep_results.py \
  ../tmp/runs/fepp_open_closed_closed_r01/fepp_open_closed_closed_r01_out.fmp \
  --state closed -o ../tmp/results/closed_r01.csv
```

Analyze the matched cohorts:

```bash
conda run -n mdpp python3 ../analyze_fep_results.py \
  --open ../tmp/results/open_r01.csv \
  --open ../tmp/results/open_r02.csv \
  --open ../tmp/results/open_r03.csv \
  --closed ../tmp/results/closed_r01.csv \
  --closed ../tmp/results/closed_r02.csv \
  --closed ../tmp/results/closed_r03.csv \
  --reference LA_AMP \
  --output-dir ../tmp/results/open_closed
```

The analyzer fits each network independently, adds their covariance matrices
under the declared independent-simulation assumption, and rejects mismatched
protocols, maps, edge graphs, atom mappings, ligand inputs, cohort hashes,
seeds, and reused run identities. Optional anchoring and the complete output
contract are documented in [`../README.md`](../README.md).

## State-definition caveat

"Paired conformations" means matched thermodynamic-cycle inputs, not paired
replicates. Each receptor label is physically state-specific only if simulation
remains in its intended basin, or if state-defining restraints and corrections
are applied. Check receptor CV/RMSD occupancy in addition to ligand and REST
diagnostics.
