# BrownDye2 Complex-Substrate Example

End-to-end estimate of the association rate between a docked protein-ligand
complex (`complex.pdb`) and a protein substrate (`substrate.pdb`).

## Workflow

```bash
conda activate ambertools
cd examples/browndye
```

1. Open `complex_pqr.ipynb` in JupyterLab and run all cells.
   The notebook performs the full preparation pipeline:

   - **Step 1.** PropKa pKa check + PDBFixer on the complex protein chain.
   - **Step 2.** RDKit SMILES bond-order assignment for the ligand chain;
     writes an SDF for antechamber.
   - **Step 3.** AmberTools/GAFF2 + ff19SB parameterisation
     (`pdb4amber`, `obabel`, `antechamber`, `parmchk2`, `tleap`) and ParmEd
     PQR export for `protein`, `ligand`, and `complex` bodies.
   - **Step 4.** APBS electrostatic map for the complex body.
   - **Step 5.** PDB2PQR + PropKa parameterisation of the protein-only
     substrate body.
   - **Step 6.** APBS electrostatic map for the substrate body.
   - **Step 7.** BrownDye XML generation (`pqr2xml`, contact-type criteria,
     `make_rxn_pairs`, `make_rxn_file`, `bd_top`).

   All configurable knobs (force fields, APBS grid, reaction criteria,
   BrownDye trajectory count, etc.) live in the first code cell of the
   notebook.

1. Run the BrownDye trajectories from a terminal:

   ```bash
   bash bdrun.sh            # standard NAM mode
   # or
   MODE=we bash bdrun.sh    # weighted-ensemble mode
   ```

   `bdrun.sh` is kept as a shell script because trajectory propagation can
   take hours. It reads
   `tmp/bdprep/intermediate/${CORE0}_${CORE1}_simulation.xml` (populated by
   Step 7 of the notebook) and writes `tmp/bdrun/results.xml` plus
   `tmp/bdrun/rate_constant.txt`.

## PyMOL Visualization

After Step 4 of the notebook, view the full complex electrostatic surface
and +/- 1 kT/e mesh contours:

```bash
pymol viz_complex_apbs.pml
```

To inspect protein-only and ligand-only diagnostic maps, first uncomment
the optional `write_apbs_input("protein", ...)` / `("ligand", ...)` calls
in Step 4 of the notebook and rerun the APBS bash cell, then:

```bash
pymol viz_components_apbs.pml
```

Both scripts assume they are launched from `examples/browndye/`.

## ChimeraX Visualization

The matching ChimeraX command scripts use the same input files and contour
levels as the PyMOL scripts:

```bash
chimerax viz_complex_apbs.cxc
```

For optional component diagnostics (after rerunning Step 4 with the
component PQRs):

```bash
chimerax viz_components_apbs.cxc
```

These scripts also assume they are launched from `examples/browndye/`.

## Temporary Directory Layout

Each workflow stage owns a top-level folder under `tmp/`. Main results are
kept directly in the stage folder; transient files created by external tools
are kept under that stage's `intermediate/` folder.

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

All defaults live in the first code cell of `complex_pqr.ipynb`; edit them
there before running. Key knobs:

- `RXN_SEARCH_DISTANCE`: distance used to find bound-pose contact pairs.
- `RXN_DISTANCE`: BrownDye reaction distance for each selected pair.
- `RXN_NEEDED`: number of contact pairs required for association.
- `N_TRAJECTORIES`: number of BrownDye trajectories (driven by Step 7's
  `input.xml`, consumed by `bdrun.sh`).
- `DEBYE_LENGTH`: normally inferred from APBS logs in Step 7; set
  explicitly in cell 1 to override.
- `FINE_SPACING`, `FINE_PADDING`, `COARSE_PADDING`: APBS grid controls.
  The defaults favour BrownDye compatibility by padding the potential maps
  beyond the molecular surface.

The default reaction criteria are broad, generated from heavy-atom contacts
in the docked pose. Treat them as a starting point and tune them against
structural or experimental knowledge before interpreting association rates.
