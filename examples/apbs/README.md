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
   - Protein: PROPKA pKa predictions at the target pH, then PDBFixer adds missing
     residues/atoms/hydrogens and applies supported PROPKA overrides (`mdpp.prep.run_propka`, `mdpp.prep.fix_pdb`). Explicit Amber residue
     labels preserve protonation states when tleap rebuilds hydrogens.
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
     pdie 2.0, sdie 78.54, 298 K), then `apbs` produces the potential map.
     Each successful run publishes the matching PQR and `<stem>.settings.json`
     with its DX and log in `tmp/apbs/`; BrownDye uses
     it to check solvent consistency and recover matching dielectric values.

### Why AmberTools for every case

Every case uses AmberTools (rather than PDB2PQR) so the PQR charges and radii are
produced by one consistent force field across the protein, ligand, and complex
examples. This APBS stage feeds the BrownDye association example:
`examples/browndye/browndye_prep.ipynb` reuses the disjoint protein and ligand `.pqr` / `.dx`
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

## Scientific interpretation

DX values are potential in kT/e. APBS total electrostatic energies are not
binding free energies; use a consistent thermodynamic cycle and converged,
compatible grids for energy comparisons. The ligand charge comes from the
SMILES microstate, not an automated pH prediction. PROPKA uses the isolated
protein in these examples; inspect its predictions and warnings, particularly
for active sites, residues near the target pH and ligand-induced pKa shifts.

Linear PB, mbondi3 radii and the chosen dielectrics are model assumptions.
Quantitative work should compare finer grids (for example 0.5 versus 0.75 Å),
larger boxes and plausible protonation/dielectric choices. Fixed box padding
does not guarantee coverage of BrownDye's far field. Check repaired residues,
ligand stereochemistry, total charge and Amber parameter warnings. The two
ligand-containing notebooks select GAFF2 explicitly in `parmchk2 -s 2` as well
as in antechamber and tleap.

See the [APBS potential units](https://apbs.readthedocs.io/en/nathan-docs/using/input/elec/write.html),
[APBS grid dimensions](https://apbs.readthedocs.io/en/nathan-docs/using/input/elec/dime.html),
and [OpenMM protonation rules](https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html).
