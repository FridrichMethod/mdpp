# BrownDye2 association-rate example

Estimate the diffusional association rate between two bodies with BrownDye2.
This folder holds **only the BrownDye stage**; the electrostatics it needs are
produced by the separate **APBS stage** in `examples/apbs/`.

## Three-stage pipeline

```
examples/apbs/<name>/<name>_apbs.ipynb   ->  <name>.pqr + <name>.dx + <name>.apbs.log   (APBS stage)
            |
            v
examples/browndye/browndye_prep.ipynb    ->  ${CORE0}_${CORE1}_simulation.xml           (BrownDye prep)
            |
            v
examples/browndye/bdrun.sh               ->  results.xml + rate_constant.txt            (BrownDye run)
```

1. **APBS stage** (`examples/apbs/`). Each component notebook
   (`protein`, `ligand`, `complex`) parameterizes its structure with AmberTools
   and solves APBS, writing `<name>.pqr`, `<name>.dx`, and `<name>.apbs.log`
   under `examples/apbs/<name>/tmp/`. Run these first.
1. **BrownDye prep** (`browndye_prep.ipynb`). Uses the protein and ligand components
   as the two BrownDye bodies (`CORE0`, `CORE1` in the first code cell) and
   **copies** their PQR, DX, log and settings into `tmp/bdprep/intermediate/`
   as a reproducible snapshot, then builds
   `tmp/bdprep/intermediate/${CORE0}_${CORE1}_simulation.xml` via
   `pqr2xml` -> contact types -> `make_rxn_pairs` / `make_rxn_file` ->
   `input.xml` -> `bd_top`. The Debye length is parsed from the APBS logs.
1. **BrownDye run** (`bdrun.sh`). Propagates trajectories
   (`nam_simulation` / `we_simulation`) and computes the rate constant. Kept as a
   shell script because it can run for hours.

To update the complete input snapshot, the everyday loop is: edit a component's
PDB -> re-run its `examples/apbs/<name>` notebook -> re-run `browndye_prep.ipynb`
-> `bdrun.sh`.

## Picking the two bodies

The bundled example supports the disjoint `protein` + `ligand` pair, with
either ordering. Both must remain in the same bound-pose coordinate frame so
`make_rxn_pairs` can find meaningful reference contacts. The supplied PDBs
are split from `complex.pdb`. Pairing `complex` with either component overlaps
atoms and is rejected. Both bodies contribute to relative diffusion; the
receptor is not immobilized. The ligand is a rigid body in this example.

## Running

```bash
conda activate ambertools           # provides mdpp + AmberTools + BrownDye on PATH

# Stage 1: APBS (once per body)
cd examples/apbs/protein && jupyter nbconvert --to notebook --execute --inplace protein_apbs.ipynb
cd ../ligand            && jupyter nbconvert --to notebook --execute --inplace ligand_apbs.ipynb

# Stage 2: BrownDye prep
cd ../../browndye
jupyter lab browndye_prep.ipynb     # or: jupyter nbconvert --to notebook --execute --inplace browndye_prep.ipynb

# Stage 3: BrownDye run (pass the same CORE0/CORE1 as the notebook)
CORE0=protein CORE1=ligand bash bdrun.sh            # standard NAM mode
# or
CORE0=protein CORE1=ligand MODE=we bash bdrun.sh    # weighted-ensemble mode
```

`bdrun.sh` reads `tmp/bdprep/intermediate/${CORE0}_${CORE1}_simulation.xml`
(written by `browndye_prep.ipynb`) and writes `tmp/bdrun/results.xml` plus
`tmp/bdrun/rate_constant.txt`. Step 8 of the notebook turns one reactive
trajectory into a VTF animation for VMD (after `bdrun.sh` has run).

## Important parameters

All knobs live in the first code cell of `browndye_prep.ipynb`:

- `CORE0` / `CORE1`: the two bodies (the disjoint `protein` + `ligand` pair).
- `RXN_SEARCH_DISTANCE`: distance in Å used to find bound-pose contact pairs.
- `RXN_DISTANCE`: BrownDye reaction distance in Å for each selected pair.
- `RXN_NEEDED`: number of contact pairs required for association.
- `N_TRAJECTORIES`: number of BrownDye trajectories (set in `input.xml`,
  consumed by `bdrun.sh`).
- `DEBYE_LENGTH`: inferred from both APBS logs in Step 1; a manual value must agree.

The default reaction criteria are broad, generated from heavy-atom contacts in
the docked pose. Treat them as a starting point and tune them against structural
or experimental knowledge before interpreting association rates.

## Temporary directory layout

`tmp/` is git-ignored. The BrownDye stage owns `tmp/bdprep/` and `tmp/bdrun/`;
the APBS inputs are read directly from `examples/apbs/<name>/tmp/`.

```text
tmp/
  bdprep/
    ${CORE0}_${CORE1}_simulation.xml
    reactions.xml
    reaction_pairs.xml
    intermediate/
  bdrun/
    results.xml
    rate_constant.txt
    intermediate/
```

## Consistency and interpretation

Regenerate both APBS fixtures to create their `<stem>.settings.json` files.
BrownDye consumes the PQR published with each successful APBS map from
`tmp/apbs/`, so a later failed APBS rerun cannot pair an old map with a new
AmberTools PQR. The notebook checks shared ionic strength, solvent dielectric, probe radius,
and temperature, and passes each body's APBS solute dielectric into BrownDye.
Dielectrics are dimensionless relative permittivities. The bundled reusable
fixtures use 298 K to match BrownDye's energy unit and kT=1; changing temperature
requires a consistent potential conversion and solvent viscosity model.
Different or nonfinite Debye lengths are rejected. Fewer reference contacts
than `RXN_NEEDED` aborts preparation instead of producing a meaningless zero rate.

Three heavy-atom pairs within the default 10 Å define a broad encounter, not
necessarily commitment to a bound state. The resulting encounter rate need
not equal experimental binding `k_on`. Compare alternative contact distances
and pair counts, independent seeds, and physically relevant structural states.
Inspect reactive/escaped/stuck counts; increase `MAX_N_STEPS` if trajectories
are censored. Check time-step tolerances, APBS grid resolution/extent and
protonation assumptions for rate stability. Fixed grid padding is not proof
that the long-range field is converged.

Rates are in M^-1 s^-1 with statistical 95% confidence intervals, which exclude
uncertainty in protonation, the force model and the encounter definition.
Weighted-ensemble estimates require equilibration and correlated-flux checks;
they are written to `rate_constant_we.txt` (NAM uses `rate_constant.txt`).
Keep full results XML and input snapshots with the seed and tool versions.

The [BrownDye2 manual](https://browndye.ucsd.edu/browndye2.pdf) specifies these
units, reaction criteria and rate/uncertainty estimators; the original method is
described by [Huber and McCammon (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2994412/).
