# Core: Trajectory I/O and Parsers

The `mdpp.core` subpackage handles trajectory loading, atom selection, and parsing of MD engine output files.

## Loading Trajectories

### Single trajectory

```python
from mdpp.core import load_trajectory

traj = load_trajectory("md.xtc", topology_path="topol.gro")
```

With atom selection (loads all atoms, then slices):

```python
traj = load_trajectory(
    "md.xtc",
    topology_path="topol.gro",
    atom_selection="protein",
    stride=10,
)
```

### Multiple trajectories

```python
from mdpp.core import load_trajectories

trajs = load_trajectories(
    ["run1.xtc", "run2.xtc", "run3.xtc"],
    topology_paths=["topol.gro", "topol.gro", "topol.gro"],
    stride=5,
)
```

## Atom Selection

Use MDTraj's DSL selection syntax:

```python
from mdpp.core import select_atom_indices

indices = select_atom_indices(traj.topology, "name CA and resid 1 to 50")
```

Map atom indices back to residue IDs:

```python
from mdpp.core import residue_ids_from_indices

res_ids = residue_ids_from_indices(traj.topology, indices)
```

## Trajectory Alignment

Superpose all frames onto a reference frame:

```python
from mdpp.core import align_trajectory

aligned = align_trajectory(traj, atom_selection="name CA", reference_frame=0)
```

## Parsing GROMACS Files

### XVG files

Parse any GROMACS `.xvg` file (from `gmx energy`, `gmx rms`, etc.):

```python
from mdpp.core import read_xvg

df = read_xvg("rmsd.xvg")
# DataFrame columns are labeled from XVG legend entries
print(df.head())
```

### EDR files

Parse binary energy files using `panedr`:

```python
from mdpp.core import read_edr

df = read_edr("ener.edr")
print(df.columns.tolist())
# ['Time', 'Bond', 'Angle', 'Proper Dih.', 'LJ-14', ...]
```
