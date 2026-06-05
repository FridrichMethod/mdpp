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
  chain B ligand `l01`). It is the same complex used in
  `examples/browndye/complex.pdb`.

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

`examples/browndye/complex_pqr.ipynb` parameterizes its complex body with
AmberTools but its protein-only substrate with PDB2PQR (PDB2PQR 3.7.1 has no
ff19SB option). Here every case uses AmberTools so the PQR charges and radii are
produced by one consistent force field across the protein, ligand, and complex
examples. The complex notebook reproduces the first APBS-calculation part of the
browndye notebook (fix protein -> assign ligand -> Amber-parameterize ->
solve APBS); the downstream BrownDye XML / trajectory steps live in
`examples/browndye/`.

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
