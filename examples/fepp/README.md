# FEP+ examples: single-state RBFE and open versus closed

This directory has two explicit workflows corresponding to the two OpenFE
analysis patterns:

| Workflow | Receptor calculations | Identifiable result | Entry point |
|---|---:|---|---|
| Standard RBFE | One conformation, many ligands | `G_i - G_r` within that receptor state | [`rbfe/`](rbfe/) |
| Open versus closed | Matched ligand networks in both conformations | `(G_open,i-G_open,r) - (G_closed,i-G_closed,r)` | [`rbfe_open_closed/`](rbfe_open_closed/) |

The workflow directories contain the user-facing commands and focused
documentation. The scripts in this directory are the shared implementation;
they are not duplicated analysis stacks.

Both workflows use the same versioned LplA input cohort. The ligand identities,
chemistry, aligned poses, and receptor coordinates originate from
[`../openfe/rbfe_open_closed.ipynb`](../openfe/rbfe_open_closed.ipynb);
protein preparation, OPLS parameterization, mapping, and simulation are native
FEP+ operations. The one-conformation FEPP workflow is therefore a one-state
projection of that paired cohort. It is not exactly pose-matched to
[`../openfe/rbfe.ipynb`](../openfe/rbfe.ipynb), whose standalone input
construction always takes ligand poses from the open complex; in particular,
the paired cohort selects closed-complex poses for Ch5 and M5.

The workflow produces pre-run input maps, but no completed production
`*_out.fmp` is versioned with this example. Consequently, it is capable of a
correct FEP+ analysis but does not yet contain a numerical open/closed result.

## What the paired calculation identifies

Let `G_{s,i}` be the standard binding free energy of ligand `i` in state `s`,
where `s` is open or closed. A directed FEP+ edge is interpreted as

```math
y_{s,a\to b}=G_{s,b}-G_{s,a}.
```

Separate RBFE graphs have independent additive gauges. With an explicitly
chosen reference ligand `r`, the identifiable paired-state result is

```math
D_i^{(r)}=
\left[G_{\mathrm{open},i}-G_{\mathrm{open},r}\right]
-
\left[G_{\mathrm{closed},i}-G_{\mathrm{closed},r}\right].
```

- `D_i^{(r)} < 0`: ligand `i` is more open-selective than reference `r`.
- `D_i^{(r)} > 0`: ligand `i` is more closed-selective than reference `r`.
- `D_r^{(r)} = 0` exactly because it fixes the gauge. Its zero reported
  uncertainty is not evidence of zero physical open/closed preference.

This is a reference-relative double difference, not an absolute
open-versus-closed binding preference. An independent open-minus-closed anchor
for the reference is required for an absolute quantity. State populations
additionally require the relevant apo conformational free energy.

## Inputs and controls

| Component | Source or treatment |
|---|---|
| 12 acyl-AMP analog identities | OpenFE SMILES provenance table |
| Ligand bond orders, formal charges, and stereochemistry | Validated against `inputs/ligands_amp_fep.smi` |
| 3D ligand poses | One OpenFE-aligned canonical pose per ligand |
| Open and closed receptors | OpenFE template PDBs in a shared aligned frame |
| Protein preparation | PrepWizard at declared pH/RMSD settings |
| His267 | Forced to the same declared state in both receptors; default HIE |
| Ligand force field and charges | Native OPLS4 with FEP+ charge assignment |
| Perturbation graph | Native `fep_mapper.py` |
| Sampling | Native Desmond FEP/REST |

All 12 current ligands have formal charge -1, so the mapped perturbations are
charge-conserving. `validate_ligand_inputs.py` checks the SDF title, ligand set,
formal charge, connectivity, and stereochemistry against the SMILES source.

The HIE default is a controlled-comparison assumption: it matches the current
open preparation and prevents the previous HIE-open/HIP-closed, one-charge-unit
confound. It is not evidence that HIE is biologically correct. If His267 is
coupled to binding, run a justified shared-state sensitivity analysis (for
example, both HIE and both HIP) or formulate an explicit proton-coupled
thermodynamic model. `compare_receptor_microstates.py` derives protonation
signatures from formal charge and H-atom connectivity rather than trusting
residue labels alone.

## Layout

```text
fepp/
|-- inputs/
|   |-- protein_{open,closed}.pdb
|   |-- ligands/*.sdf
|   `-- ligands_amp_fep.smi
|-- validate_ligand_inputs.py
|-- compare_receptor_microstates.py
|-- build_fepp_inputs.sh
|-- run_fep_plus.sh
|-- extract_fep_results.py
|-- analyze_rbfe_results.py
|-- analyze_fep_results.py
|-- rbfe/                        # one-conformation workflow
|-- rbfe_open_closed/            # matched open/closed workflow
|-- extract_edge_mappings.py
|-- plot_edge_mappings.py
|-- plot_fep_map.py
`-- tmp/                         # generated and ignored
    |-- ligand_validation.json
    |-- receptor_{open,closed}.mae
    |-- receptor_microstates.json
    |-- state_inputs_{open,closed}.tsv
    |-- paired_inputs.tsv
    |-- ligands.maegz
    |-- {open,closed}_pv.mae
    |-- {open,closed}_map.{fmp,edge}
    |-- {open,closed}_map_mappings.json
    |-- *.provenance
    `-- runs/<jobname>/
```

## Build provenance-tracked inputs

```bash
export SCHRODINGER=/apps/schrodinger2025-4
cd examples/fepp

./build_fepp_inputs.sh
./build_fepp_inputs.sh --check
```

The build performs the following operations:

1. Validate all ligand SDFs against the SMILES provenance table.
1. PrepWizard both receptors with the same declared His267 state.
1. Compare residue identities, total formal charge, and per-residue
   protonation signatures across the prepared receptors.
1. Assemble the ligand bundle and receptor-first pose-viewer files.
1. Run `fep_mapper.py` with one common environment structure (`-e 1`), extract
   the canonical atom mappings, validate all staged outputs, then publish the
   `.fmp`/`.edge`/mapping-JSON artifact.
1. Confirm that the open and closed `.edge` files are byte-identical and their
   coordinate-independent atom-mapping fingerprints agree.
1. Publish a shared `paired_inputs.tsv` fingerprint containing both receptor
   and map generations plus the common ligand, microstate, edge, and mapping
   lineage.

Every generated artifact has a dependency fingerprint and output hashes in a
`.provenance` sidecar. A changed PDB, SDF, SMILES table, pH, RMSD, His267 state,
map topology, Suite release, missing `.edge`, or modified output invalidates the
appropriate artifact and all downstream dependencies. Old outputs without a
sidecar are intentionally stale.

`--check` is read-only: it requires an existing build and verifies hashes and
fingerprints without creating or refreshing artifacts. The shell workflow is
Linux/GNU-oriented and requires Bash (including `mapfile`), `flock`, GNU
coreutils (`realpath` and `sha256sum`), and GNU `sort`, in addition to the
declared Schrödinger Suite and conda environment.
The builder and extractor fail closed when the Suite's nonempty `version.txt`
release metadata is unavailable.

Useful variants are:

```bash
./build_fepp_inputs.sh --his267-state HIP -f
./build_fepp_inputs.sh --topology star -f
./build_fepp_inputs.sh -c open
```

A one-state build cannot perform the paired microstate comparison and emits a
warning. The direct launcher defaults to the paired-cohort contract, while
`--workflow single` checks and snapshots only the selected state. Prefer the
entry points under `rbfe/` and `rbfe_open_closed/` so this choice is explicit.
The `--allow-microstate-mismatch` escape hatch is deliberately explicit
because a mismatched protonation state changes the scientific comparison. A
launch from such a build must repeat the same option; the policy is then
recorded in the manifest and normalized protocol fingerprint. This option
never permits a missing, added, or changed receptor residue or a heavy-atom
composition/connectivity change; those mismatches always fail.

Inspect a map with a release-independent Suite command:

```bash
$SCHRODINGER/run -FROM scisol fmp_info.py -f tmp/open_map.fmp
```

## Inspect maps and atom mappings

```bash
conda run -n mdpp python3 plot_fep_map.py -c open
conda run -n mdpp python3 plot_fep_map.py -c closed

$SCHRODINGER/run python3 extract_edge_mappings.py -f tmp/open_map.fmp
conda run -n mdpp python3 plot_edge_mappings.py \
  --json tmp/open_map_mappings.json
```

The edge plots expose more than dummy atoms:

- red: atoms deleted from ligand A;
- green: atoms added to ligand B;
- orange: mapped-core element, isotope, formal-charge, or aromaticity changes;
- blue: mapped-core bond presence, order, aromaticity, or stereo changes.

Do not assume two maps have identical mappings merely because they use the same
ligands. Compare both the built `.edge` files and the extracted JSON
`mapping_fingerprint`; the paired builder makes both checks mandatory.

## Run FEP+

First verify that each map prepares successfully:

```bash
./run_fep_plus.sh -c open --prepare
./run_fep_plus.sh -c closed --prepare
```

The launcher makes the default scientific protocol explicit in the recorded
command and `manifest.tsv`:

- OPLS4;
- assigned custom ligand charges;
- SPC water;
- muVT ensemble;
- 5000 ps production;
- ensemble-dependent Suite equilibration default: 20 ps for muVT and 240 ps
  for NPT/NVT, unless explicitly overridden;
- 12 default-protocol lambda windows;
- 0.0 M added salt;
- a recorded state-stable seed schedule starting at 2014.

For a production comparison, use independent repeats for both states:

```bash
./run_fep_plus.sh -c both --repeats 3 --seed-base 2014 \
  -H localhost -S localhost
```

Each repeat receives a unique job name, seed, and atomically published
`tmp/runs/<jobname>/` directory. The schedule is
`open = seed_base + 2(r-1)` and `closed = seed_base + 2(r-1) + 1`, so seeds are
unchanged whether the states are launched together or separately. Every run
directory contains read-only snapshots of the exact map, edge list, canonical
mapping, ligand bundle, ligand-validation report, receptor-microstate report,
shared paired-input manifest, and map provenance used by that job. One launcher
invocation requires every repeat/state snapshot to retain the same paired-input
hash; a concurrent rebuild aborts the remaining cohort. Existing run
directories are never silently overwritten. Three repeats are a starting
point, not a convergence guarantee.
Inspect edge convergence, ligand RMSD, REST exchange, cycle residuals, and
between-repeat heterogeneity; extend inadequate edges or simulations. Cycle
closure alone is not evidence of convergence.

The launcher exposes `--time-ps`, `--equilibration-time-ps`,
`--lambda-windows`, `--salt-molar`, `--water`, `--ensemble`, and
`--forcefield`, plus the explicit `--allow-microstate-mismatch` input-policy
override. Seeds are limited to the reproducible signed 31-bit range. Record and
justify deviations. In particular, use matched ionic conditions when that is
required by the intended cross-protocol comparison. Suite-internal settings
not exposed here remain release-defined defaults, which is why the exact Suite
release is part of the protocol fingerprint.

## Export raw FEP+ results

Normalize each completed job separately. The adapter invokes the supported,
release-independent `fmp2excel.py` entry point and copies only the raw Bennett
binding edge result and its reported uncertainty:

```bash
$SCHRODINGER/run python3 extract_fep_results.py \
  tmp/runs/fepp_open_r01/fepp_open_r01_out.fmp \
  --manifest tmp/runs/fepp_open_r01/manifest.tsv \
  --state open -o tmp/results/open_r01.csv

$SCHRODINGER/run python3 extract_fep_results.py \
  tmp/runs/fepp_closed_r01/fepp_closed_r01_out.fmp \
  --manifest tmp/runs/fepp_closed_r01/manifest.tsv \
  --state closed -o tmp/results/closed_r01.csv
```

Adjust the input path if Job Control stages the returned output elsewhere. The
adapter rejects a pre-run map or any blank, nonfinite, or zero-uncertainty
Bennett result even if `fmp2excel.py` exits successfully.
It also requires the launch manifest, checks its state, job name, seed, Suite
release, and production-versus-prepare status. It verifies every exact run
snapshot against the manifest, cross-checks the edge and canonical-mapping
snapshots, and requires the completed FMP export to contain exactly that edge
set, the exact original core/dummy atom-mapping fingerprint, and the same
ordered receptor/membrane/solvent environment fingerprint as the immutable
input map. The environment fingerprint uses Suite-serialized Maestro CT data,
so a renamed or swapped open/closed output is rejected. It parses and
cross-checks the workflow-specific cohort snapshot: `input_state_inputs.tsv`
for standalone RBFE, or `input_paired_inputs.tsv` plus the receptor-microstate
report for the paired workflow.
It embeds the complete input lineage, canonical scientific-protocol
fingerprint, manifest hash, source-FMP hash, and raw vendor edge-table hash in
the normalized schema. The adjacent `manifest.tsv` is the default when
`--manifest` is omitted.

Publication is serialized by an output lock. All files are staged first, stale
optional vendor siblings are removed, and `<output>.provenance.json` is
atomically installed last as the bundle's completion marker. A missing marker
means publication was interrupted and the sibling bundle should not be used.
The analyzer enforces this contract: it takes the same lock in shared mode,
reads one immutable byte snapshot, and verifies the completion marker, CSV
hash, protocol, run identity, mapping and environment fingerprints, and input
lineage. New publications use commit-marker schema 3. Legacy normalized-v2
exports with a schema-2 marker remain readable for backward compatibility but
do not prove receptor/environment identity; re-extract their source FMP before
production use. Normalized-v3 standalone results require marker schema 3.

The statistical fit intentionally does not use `pred_dg`, `ccc_ddg`, or
`ccc_ddg_error`. The first two are cycle-closure-derived quantities;
`ccc_ddg_error` is a correction/diagnostic magnitude, not an independent
standard error. The adapter preserves the vendor tables as sibling
`*.vendor_{edges,nodes,summary,hysteresis}.csv` files when available. Use the
edge table's convergence, ligand-RMSD, and REST-exchange ratings as diagnostics,
not as replacements for independent repeats. Official `Good`, `Fair`, `Bad`,
and `N/A` ratings are also retained in the normalized rows and summarized in
`analysis.json`; they are warnings, not an automatic edge-exclusion rule.

## Analyze a single receptor state

The standalone workflow fits the raw Bennett network without inventing a
second receptor leg:

```bash
conda run -n mdpp python3 analyze_rbfe_results.py \
  --input tmp/results/open_r01.csv \
  --input tmp/results/open_r02.csv \
  --input tmp/results/open_r03.csv \
  --state open --reference LA_AMP \
  --output-dir tmp/results/open_rbfe
```

It reports `G_i-G_r`, the complete gauge-conditioned covariance, all pairwise
reference-invariant contrasts, edge residuals, cycle diagnostics, repeat
heterogeneity, vendor QC, and exact run/input provenance. A negative value
means stronger predicted binding than the reference. The reference row is
`0 +/- 0` only because it fixes the additive gauge. See [`rbfe/`](rbfe/) for
the complete standalone workflow.

## Analyze paired states

Pass every independent repeat and choose the reference explicitly:

```bash
conda run -n mdpp python3 analyze_fep_results.py \
  --open tmp/results/open_r01.csv \
  --open tmp/results/open_r02.csv \
  --open tmp/results/open_r03.csv \
  --closed tmp/results/closed_r01.csv \
  --closed tmp/results/closed_r02.csv \
  --closed tmp/results/closed_r03.csv \
  --reference LA_AMP \
  --output-dir tmp/results/open_closed
```

For each state, the analyzer fits raw directed Bennett edges by weighted least
squares with the reference fixed to zero. It assumes diagonal within-network
edge covariance because FEP+ does not export edge-edge covariance, combines
matching independent repeats with a fixed-effect inverse-variance mean, and
propagates the complete node covariance. To avoid increased apparent precision
from discordant repeats, each pooled edge standard uncertainty is multiplied by
the Birge factor `sqrt(max(1, Q/(k-1)))`. Open and closed simulations are assumed
independent, so their covariance matrices add in the double difference. Shared
force-field and other systematic errors are not represented by the Bennett
uncertainties. The analyzer rejects reused files/FMPs/manifests/run IDs or
seeds, mixed engines, and nonidentical protocol fingerprints.
It additionally requires all repeats within a state to use one exact input map,
requires open and closed to use distinct state-specific maps, and verifies that
their shared edge list, canonical atom mapping, ligand inputs, ligand-validation
report, and receptor-microstate comparison are identical by content hash.
The shared paired-input hash also binds both state-specific receptor/map
generations, so an old open run cannot be combined with a newly rebuilt closed
run even when the ligand graph is unchanged.

Outputs include:

- `analysis.json`: estimand, assumptions, provenance, state fits, complete
  covariance matrices, and optional anchored result;
- `selectivity.csv`: per-ligand relative open-minus-closed values;
- `{open,closed}_edge_residuals.csv`: observed-minus-fitted edge residuals;
- `{open,closed}_cycles.csv`: a deterministic cycle basis and closure z-scores.
- `analysis.provenance.json`: hashes of the complete analysis bundle, installed
  last as its completion marker.

Analysis outputs are staged and serialized by `<output-dir>.lock`; the marker
is removed before replacement and atomically installed last. Treat a missing
marker as an interrupted/incomplete analysis publication.

The network diagnostic reports `chi^2 = r^T C^{-1} r`, degrees of freedom
`m-n+1`, and a p-value
when cycles exist. It separately reports fixed-effect repeat heterogeneity,
its p-value, and the applied edgewise uncertainty scale. These p-values are
approximate diagnostics: they assume normal errors with calibrated reported
standard uncertainties, while Bennett uncertainties are themselves estimated
and edge independence is an approximation. Individual cycle z-scores are
correlated. Large inconsistency suggests insufficient sampling or
underestimated uncertainties; it does not identify one bad edge automatically.
Tree networks have no cycle-consistency degrees of freedom.

### Optional anchor

An independent reference anchor may be supplied as JSON:

```json
{
  "reference": "LA_AMP",
  "quantity": "binding_free_energy_open_minus_closed",
  "sign_convention": "open_minus_closed",
  "estimate": 0.5,
  "standard_uncertainty": 0.1,
  "unit": "kcal/mol"
}
```

```bash
conda run -n mdpp python3 analyze_fep_results.py \
  --open tmp/results/open_r01.csv \
  --closed tmp/results/closed_r01.csv \
  --reference LA_AMP --anchor anchor.json \
  --output-dir tmp/results/open_closed_anchored
```

If `a_r +/- u_a` is independent of the RBFE networks, the anchored values are
`Q_i = D_i^{(r)} + a_r`, and the covariance gains the required rank-one term
`u_a^2 1 1^T`. An apo conformational free energy
alone is not silently reinterpreted as a ligand binding anchor. Before use,
verify that the anchor matches the receptor basin and microstate, temperature,
standard state, and thermodynamic quantity, and that its simulations are
independent of the RBFE networks. Preserve its citation, method, and state
definition alongside the hash-bound anchor file.

## Scientific limitations to resolve before publication

- **A starting structure is not automatically a thermodynamic state.** Calling
  a run "open" or "closed" is state-specific only if the receptor remains in
  the intended conformational basin, or if explicit state-defining restraints
  and their corrections are used. Measure receptor CV/RMSD basin occupancy;
  ligand RMSD and REST-exchange ratings do not establish receptor-state
  stability. Otherwise interpret results as protocol- and
  initial-conformation-conditioned RBFE values.
- **Protocol comparison, not engine-only validation.** FEP+ and OpenFE use
  different preparation, force fields, charge assignment, solvent/sampling
  protocols, and possibly perturbation graphs. Agreement is useful; a
  discrepancy cannot be attributed to the simulation engine alone.
- **Inherited poses.** The shared ligand poses control one source of variation,
  but their canonical-pose selection and transfer into both conformations are
  modeling assumptions. Inspect pose stability and alternative plausible
  binding modes.
- **Receptor composition.** The current apo templates omit any catalytic metal
  or structured-water model. Establish whether Mg²⁺, cofactors, or conserved
  waters are required for the intended LplA binding state before interpreting
  production affinities.
- **Difficult perturbations.** In the currently generated map, MP5→BCN,
  LA→BCN, and P5→OP5 have low mapper similarities of roughly 0.13, 0.15, and
  0.17. The Suite did not flag the map as formally bad, but these edges deserve
  mapping inspection, repeat agreement, and possibly a denser or redesigned
  graph.
- **Convergence versus closure.** Good cycle closure can coexist with correlated
  or uniformly biased edges. Use independent repeats and trajectory-level
  diagnostics, not closure alone.
- **Absolute selectivity and populations.** The unanchored calculation yields
  only reference-relative ligand selectivity. Absolute open/closed binding
  preferences and conformational populations require additional thermodynamic
  information.
