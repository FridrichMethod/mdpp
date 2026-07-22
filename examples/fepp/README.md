# FEP+ relative binding: LplA open vs closed

Native Schrodinger **FEP+** relative-binding (RBFE) inputs for the LplA
open/closed conformation comparison, built as a direct counterpart to the
OpenFE notebook [`../openfe/rbfe_open_closed.ipynb`](../openfe/rbfe_open_closed.ipynb).

OpenFE is used here **only as the source of the inputs**; everything downstream
is the native FEP+ pipeline (PrepWizard, OPLS4, `fep_mapper`, `fep_plus` on
Desmond/GPU). This lets the two engines be cross-validated on the same system,
ligand set, poses, and protein conformations.

## What is taken from OpenFE vs what is native FEP+

| Piece | Source |
|---|---|
| Ligand selection (12 acyl-AMP analogs) | OpenFE (`ligands_amp_fep.smi`) |
| Ligand chemistry / bond orders / protonation | OpenFE SMILES templates |
| Ligand 3D **binding poses** | OpenFE aligned poses (one canonical pose per ligand, NTD-superposed into both pockets) |
| Two protein conformations (open=3a7r, closed=1x2h) | OpenFE template proteins, in the shared aligned frame |
| Protein preparation | **native** PrepWizard (pH 7) |
| Ligand partial charges | **native** OPLS4 (OpenFE AM1-BCC charges are intentionally NOT reused) |
| Perturbation map / topology | **native** `fep_mapper` (its own optimized graph; not the OpenFE network) |
| FEP engine / sampling | **native** FEP+ (Desmond FEP/REST, GPU) |

Because the poses are reused verbatim, the open and closed receptors and the
single shared ligand set all live in **one NTD-aligned coordinate frame**, so a
pose-viewer file (receptor + ligands) is valid for both conformations and the
open/closed comparison stays controlled.

All 12 ligands carry a net charge of **-1** (the phosphate), so every
perturbation is charge-conserving and no charge-correction is required.

## Layout

```
fepp/
|-- inputs/
|   |-- protein_open.pdb         # apo template, open (3a7r), NTD-aligned frame
|   |-- protein_closed.pdb       # apo template, closed (1x2h), NTD-aligned frame
|   |-- ligands/*.sdf            # 12 OpenFE-aligned canonical poses (net -1, explicit H)
|   `-- ligands_amp_fep.smi      # SMILES provenance (chemistry source)
|-- build_fepp_inputs.sh         # PrepWizard -> pose-viewer -> fep_mapper
|-- run_fep_plus.sh              # launch / prepare fep_plus per conformation
`-- tmp/                        # generated inputs (below)
    |-- receptor_{open,closed}.mae
    |-- ligands.maegz
    |-- {open,closed}_pv.mae     # pose-viewer: receptor + ligand poses
    `-- {open,closed}_map.fmp    # native FEP map (+ .edge)
```

## Build the inputs

```bash
export SCHRODINGER=/apps/schrodinger2025-4
cd examples/fepp
./build_fepp_inputs.sh                 # both conformations
./build_fepp_inputs.sh -c open         # one conformation
./build_fepp_inputs.sh -t star -f      # different topology, force rebuild
```

Steps performed:

1. **PrepWizard** each apo conformation at pH 7 (fill side chains, disulfides,
   PROPKA, restrained minimization with 0.3 A heavy-atom RMSD so the receptor
   stays in the aligned frame) -> `receptor_<conf>.mae`.
1. **structcat** the 12 aligned ligand poses -> `ligands.maegz`.
1. **structcat** receptor + ligands -> `<conf>_pv.mae` (pose-viewer).
1. **fep_mapper.py** -> native optimized-topology `<conf>_map.fmp`.

Inspect a map:

```bash
$SCHRODINGER/run $SCHRODINGER/mmshare-v7.2/python/scripts/fmp_info.py tmp/open_map.fmp
```

## Visualize the perturbation map

`plot_fep_map.py` renders the ligand network (2D structures on nodes,
perturbations as edges) -- the static analog of OpenFE's
`plot_atommapping_network` -- straight from the `.edge` file:

```bash
conda run -n mdpp python3 plot_fep_map.py -c open     # -> tmp/open_map.png
conda run -n mdpp python3 plot_fep_map.py -c closed   # -> tmp/closed_map.png
```

(Schrodinger's own `fmp2pdf.py` only renders a results report *after* the FEP
run, since it needs simulation-interaction data; use it on the `*_out.fmp`.)

### Per-edge atom mappings

To preview the *actual* atom mapping `fep_mapper` chose for each edge (the
analog of OpenFE's per-edge `mapping` view), extract it from the `.fmp` and
render A|B side by side -- deleted atoms red on A, added atoms green on B, the
mapped core left plain:

```bash
# phase 1: read the .fmp (needs Schrodinger's Python + the fep graph API)
$SCHRODINGER/run python3 extract_edge_mappings.py            # -> tmp/open_map_mappings.json
# phase 2: draw (needs the mdpp RDKit Cairo backend)
conda run -n mdpp python3 plot_edge_mappings.py              # -> tmp/open_map_mappings.{pdf,png}
```

The split is necessary because only Schrodinger's Python can read the `.fmp`,
but its bundled RDKit has no Cairo drawer. The PDF has one full-resolution edge
per page; the PNG is a grid overview. The mapping depends only on the ligand
pair, so the open and closed maps give identical per-edge mappings.

## Run FEP+

```bash
# Validate the map builds a runnable system without launching production:
./run_fep_plus.sh -c open --prepare

# Launch (1 local GPU); each edge runs complex + solvent FEP/REST legs:
./run_fep_plus.sh -c open   -H localhost -S localhost
./run_fep_plus.sh -c closed -H localhost -S localhost
```

A full run is hours-to-days of GPU time per conformation. On a cluster, point
`-S/--subhost` at a GPU queue entry from `$SCHRODINGER/schrodinger.hosts`.

## Comparing to OpenFE

After both engines finish, compare per-ligand relative binding free energies
(open vs closed in each engine, and FEP+ vs OpenFE for each conformation). The
FEP+ results land in `tmp/<jobname>_out.fmp`; summarize with
`fmp_stats.py` / `fmp2excel.py` under `$SCHRODINGER/mmshare-v7.2/python/scripts/`.
