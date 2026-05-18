# BrownDye2 Complex-Substrate Example

This example prepares two BrownDye rigid bodies:

- `complex/`: the docked protein-ligand complex from `complex.pdb`, prepared
  with AmberTools/GAFF2 and ff19SB.
- `substrate/`: the protein-only substrate from `substrate.pdb`, prepared with
  PDB2PQR's available AMBER force field and PropKa pH assignment.

The BrownDye setup consumes the two APBS results as `complex` and `substrate`.

## Workflow

```bash
conda activate ambertools
cd examples/browndye
```

1. Run `complex_pqr.ipynb`.
   This writes main outputs to `tmp/complex/prep/` and extracted-chain
   intermediates to `tmp/complex/prep/intermediate/`.

1. Parameterize with AmberTools:

   ```bash
   bash complex/prep.sh
   ```

   This writes main outputs to `tmp/complex/ambertools/` and tool intermediates
   to `tmp/complex/ambertools/intermediate/`.

1. Run APBS:

   ```bash
   bash complex/run_apbs.sh
   ```

   This writes `tmp/complex/apbs/complex.dx` by default, with APBS working files
   in `tmp/complex/apbs/intermediate/`. To also generate component maps for
   inspection:

   ```bash
   PQR_STEMS="complex protein ligand" bash complex/run_apbs.sh
   ```

1. Prepare the substrate body:

   ```bash
   bash substrate/prep.sh
   bash substrate/run_apbs.sh
   ```

   This writes `tmp/substrate/pdb2pqr/substrate.pqr` and
   `tmp/substrate/apbs/substrate.dx`.

1. Build BrownDye inputs:

   ```bash
   bash bdprep.sh
   ```

   By default the script uses `/apps/browndye2`; override with `BD_HOME` or
   `BD_BIN` if your BrownDye executables are elsewhere. This writes main
   outputs to `tmp/bdprep/` and setup intermediates to `tmp/bdprep/intermediate/`.

1. Run BrownDye and compute a rate estimate:

   ```bash
   bash bdrun.sh
   ```

   For weighted-ensemble mode:

   ```bash
   MODE=we bash bdrun.sh
   ```

   This writes final simulation outputs to `tmp/bdrun/` and simulation working
   files to `tmp/bdrun/intermediate/`.

## PyMOL Visualization

After `complex/run_apbs.sh`, view the full complex electrostatic surface and
+/- 1 kT/e mesh contours:

```bash
pymol viz_complex_apbs.pml
```

To inspect protein-only and ligand-only diagnostic maps, generate them first:

```bash
PQR_STEMS="complex protein ligand" bash complex/run_apbs.sh
pymol viz_components_apbs.pml
```

Both scripts assume they are launched from `examples/browndye/`.

## ChimeraX Visualization

The matching ChimeraX command scripts use the same input files and contour
levels as the PyMOL scripts:

```bash
chimerax viz_complex_apbs.cxc
```

For optional component diagnostics:

```bash
PQR_STEMS="complex protein ligand" bash complex/run_apbs.sh
chimerax viz_components_apbs.cxc
```

These scripts also assume they are launched from `examples/browndye/`.

## Temporary Directory Layout

Each workflow stage owns a top-level folder under `tmp/`. Main results are kept
directly in the stage folder; transient files created by external tools are kept
under that stage's `intermediate/` folder.

```text
tmp/
  complex/
    prep/
      protein_fixed.pdb
      ligand.sdf
      ligand_charge.txt
      ligand_resname.txt
      protein_propka.tsv
      intermediate/
    ambertools/
      complex.pqr
      protein.pqr      # optional diagnostic component
      ligand.pqr       # optional diagnostic component
      *.prmtop
      *.rst7
      intermediate/
    apbs/
      complex.in
      complex.dx
      complex.apbs.log
      intermediate/
  substrate/
    pdb2pqr/
      substrate.pqr
      substrate.pdb2pqr.log
      intermediate/
    apbs/
      substrate.in
      substrate.dx
      substrate.apbs.log
      intermediate/
  bdprep/
    complex_substrate_simulation.xml
    reactions.xml
    reaction_pairs.xml
    intermediate/
  bdrun/
    results.xml
    rate_constant.txt
    intermediate/
```

## Important Parameters

- `RXN_SEARCH_DISTANCE`: distance used to find bound-pose contact pairs.
- `RXN_DISTANCE`: BrownDye reaction distance for each selected pair.
- `RXN_NEEDED`: number of contact pairs required for association.
- `N_TRAJECTORIES`: number of BrownDye trajectories.
- `DEBYE_LENGTH`: normally inferred from APBS logs; set explicitly if needed.
- `PQR_STEMS`: APBS input stems, default `complex` for `complex/run_apbs.sh`
  and `substrate` for `substrate/run_apbs.sh`.
- `FINE_SPACING`, `FINE_PADDING`, `COARSE_PADDING`: APBS grid controls. The
  defaults favor BrownDye compatibility by padding the potential maps beyond the
  molecular surface.

The default reaction criteria are broad, generated from heavy-atom contacts in
the docked pose. Treat them as a starting point and tune them against structural
or experimental knowledge before interpreting association rates.
