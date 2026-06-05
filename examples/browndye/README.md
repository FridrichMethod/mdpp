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
1. **BrownDye prep** (`browndye_prep.ipynb`). Picks any two of those components
   as the two BrownDye bodies (`CORE0`, `CORE1` in the first code cell) and
   **symlinks** their APBS `.pqr`/`.dx` into `tmp/bdprep/intermediate/` (so a
   fresh APBS run is picked up with no copying), then builds
   `tmp/bdprep/intermediate/${CORE0}_${CORE1}_simulation.xml` via
   `pqr2xml` -> contact types -> `make_rxn_pairs` / `make_rxn_file` ->
   `input.xml` -> `bd_top`. The Debye length is parsed from the APBS logs.
1. **BrownDye run** (`bdrun.sh`). Propagates trajectories
   (`nam_simulation` / `we_simulation`) and computes the rate constant. Kept as a
   shell script because it can run for hours.

Because the inputs are symlinks, the everyday loop is just: edit a component's
PDB -> re-run its `examples/apbs/<name>` notebook -> re-run `browndye_prep.ipynb`
-> `bdrun.sh`.

## Picking the two bodies

`CORE0` and `CORE1` may be any of `protein`, `ligand`, `complex`. The two bodies
must be **co-registered in a bound pose** so that `make_rxn_pairs` can find the
bound contacts. The three `examples/apbs/` components are all split from the same
`complex.pdb`, so the default pair models a meaningful association:

- `CORE0 = "protein"` - the receptor (held fixed)
- `CORE1 = "ligand"` - the ligand (diffuses toward its bound site)

`complex` spatially overlaps either single body, so pairings other than
`protein` + `ligand` are for demonstrating the plumbing only.

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

- `CORE0` / `CORE1`: the two bodies (any of `protein`, `ligand`, `complex`).
- `RXN_SEARCH_DISTANCE`: distance used to find bound-pose contact pairs.
- `RXN_DISTANCE`: BrownDye reaction distance for each selected pair.
- `RXN_NEEDED`: number of contact pairs required for association.
- `N_TRAJECTORIES`: number of BrownDye trajectories (set in `input.xml`,
  consumed by `bdrun.sh`).
- `DEBYE_LENGTH`: inferred from the APBS logs in Step 1; set explicitly to override.

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
