# BrownDye2 Protein-Ligand Example

This example prepares a docked protein-ligand pose for APBS and BrownDye2.
BrownDye treats each moving body as a rigid core, so the workflow generates
separate PQR, APBS DX, and BrownDye XML files for the protein and ligand.

## Workflow

```bash
conda activate ambertools
cd examples/browndye
```

1. Run `complex_pqr.ipynb`.
   This writes `tmp/protein_fixed.pdb`, `tmp/ligand.sdf`,
   `tmp/ligand_charge.txt`, and `tmp/ligand_resname.txt`.

1. Parameterize with AmberTools:

   ```bash
   bash run_ambertools.sh
   ```

   This writes `tmp/protein.pqr`, `tmp/ligand.pqr`, and `tmp/complex.pqr`.

1. Run APBS:

   ```bash
   bash run_apbs.sh
   ```

   This writes `tmp/protein.dx` and `tmp/ligand.dx`.

1. Build BrownDye inputs:

   ```bash
   bash bdprep.sh
   ```

   By default the script uses `/apps/browndye2`; override with `BD_HOME` or
   `BD_BIN` if your BrownDye executables are elsewhere.

1. Run BrownDye and compute a rate estimate:

   ```bash
   bash bdrun.sh
   ```

   For weighted-ensemble mode:

   ```bash
   MODE=we bash bdrun.sh
   ```

## Important Parameters

- `RXN_SEARCH_DISTANCE`: distance used to find bound-pose contact pairs.
- `RXN_DISTANCE`: BrownDye reaction distance for each selected pair.
- `RXN_NEEDED`: number of contact pairs required for association.
- `N_TRAJECTORIES`: number of BrownDye trajectories.
- `DEBYE_LENGTH`: normally inferred from APBS logs; set explicitly if needed.
- `PQR_STEMS`: APBS input stems, default `protein ligand`.

The default reaction criteria are broad, generated from heavy-atom contacts in
the docked pose. Treat them as a starting point and tune them against structural
or experimental knowledge before interpreting association rates.
