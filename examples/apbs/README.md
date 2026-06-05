# APBS electrostatics examples

Three self-contained notebooks that compute an APBS electrostatic potential
map (`.dx`) for three input cases, each parameterized end-to-end with
**AmberTools** (never PDB2PQR):

| Folder | Input | Notebook | Force field | Output map |
|-------------|------------------|-----------------------|--------------------|---------------|
| `protein/` | `protein.pdb` | `protein_apbs.ipynb` | ff19SB | `protein.dx` |
| `ligand/` | `ligand.pdb` | `ligand_apbs.ipynb` | GAFF2 / AM1-BCC | `ligand.dx` |
| `complex/` | `complex.pdb` | `complex_apbs.ipynb` | ff19SB + GAFF2 | `complex.dx` |

- `protein.pdb` is a single protein chain (chain A).
- `ligand.pdb` is a single small-molecule ligand (residue `l01`, chain B).
- `complex.pdb` is the docked protein-ligand complex (chain A protein +
  chain B ligand `l01`); `protein.pdb` and `ligand.pdb` are split from it, so
  the three components share one coordinate frame. The `examples/browndye/`
  association example reuses these components' APBS outputs as its bodies.

## Pipeline

Every notebook follows the same shape and gathers all imports + constants +
`tmp/` setup in its **top cell**:

1. **Prepare**
   - Protein: PropKa pKa check at the target pH, then PDBFixer adds missing
     residues/atoms/hydrogens (`mdpp.prep.run_propka`, `mdpp.prep.fix_pdb`).
   - Ligand: RDKit assigns bond orders from a SMILES template and writes an SDF
     (`mdpp.prep.assign_topology`); PDB files carry no bond-order information.
   - Complex: both of the above, after splitting chains with
     `mdpp.prep.ChainSelect`.
1. **Parameterize with AmberTools** -> PQR
   - `pdb4amber` (protein), `obabel` + `antechamber` (AM1-BCC) + `parmchk2`
     (ligand), then `tleap` builds the topology (`combine` for the complex).
   - ParmEd exports each `prmtop`/`rst7` pair to a `.pqr` (per-atom charge +
     radius).
1. **Solve APBS** -> `.dx`
   - `mdpp.prep.write_apbs_input` writes a multigrid `.in` sized from the
     PQR bounding box (physics defaults in `mdpp.prep.apbs`: 0.150 M NaCl,
     pdie 2.0, sdie 78.54), then `apbs` produces the potential map.

### Why AmberTools for every case

Every case uses AmberTools (rather than PDB2PQR) so the PQR charges and radii are
produced by one consistent force field across the protein, ligand, and complex
examples. This APBS stage feeds the BrownDye association example:
`examples/browndye/browndye_prep.ipynb` reuses any two components' `.pqr` / `.dx`
outputs as its bodies, then builds and runs the BrownDye simulation.

## Running

The notebooks call AmberTools (`pdb4amber`, `antechamber`, `parmchk2`,
`tleap`), `obabel`, and `apbs`, so run them in the AmberTools environment
(which also has `mdpp` installed):

```bash
conda activate ambertools
cd examples/apbs/protein   # or ligand / complex
jupyter lab protein_apbs.ipynb
```

Or execute non-interactively:

```bash
cd examples/apbs/protein
jupyter nbconvert --to notebook --execute --inplace protein_apbs.ipynb
```

## Outputs

Each notebook writes into a per-folder `tmp/` workspace, with transient tool
files kept in each stage's `intermediate/` subfolder:

```
<folder>/tmp/
├── prep/         # fixed PDB, PropKa report, ligand SDF
├── ambertools/   # <stem>.prmtop  <stem>.rst7  <stem>.pqr
└── apbs/         # <stem>.in  <stem>.apbs.log  <stem>.dx
```

The complex notebook exports `protein.pqr`, `ligand.pqr`, and `complex.pqr`
but solves APBS for the **complex only** by default; uncomment the diagnostic
lines in its Step 4 to also map the individual bodies.

## Visualization

Each folder ships a PyMOL (`.pml`) and ChimeraX (`.cxc`) script that loads its
component's `tmp/ambertools/<name>.pqr` and `tmp/apbs/<name>.dx`, colors the
molecular surface by electrostatic potential (red -> white -> blue over
-5 .. +5 kT/e), and draws +/- 1 kT/e mesh contours. Run them from the component
folder after the notebook has produced the `tmp/` outputs:

```bash
cd examples/apbs/protein && pymol viz_protein_apbs.pml   # or: chimerax viz_protein_apbs.cxc
cd examples/apbs/ligand  && pymol viz_ligand_apbs.pml    # or: chimerax viz_ligand_apbs.cxc
cd examples/apbs/complex && pymol viz_complex_apbs.pml   # or: chimerax viz_complex_apbs.cxc
```
