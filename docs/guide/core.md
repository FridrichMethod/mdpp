# Core: Trajectory I/O and Parsers

The `mdpp.core` subpackage handles trajectory loading, atom selection, and parsing of MD engine output files.

## Loading Trajectories

### Single trajectory

```python
from mdpp.core import load_trajectory

traj = load_trajectory("md.xtc", topology_path="topol.gro")
```

With atom selection and striding:

```python
traj = load_trajectory(
    "md.xtc",
    topology_path="topol.gro",
    atom_selection="protein",
    stride=10,
)
```

Use `start` and `stop` to skip equilibration or limit the frame range.
Frame selection follows Python's `range(start, stop, stride)` convention:

```python
# Skip the first 1000 frames, read up to frame 5000, every 10th frame
traj = load_trajectory(
    "md.xtc",
    topology_path="topol.gro",
    start=1000,
    stop=5000,
    stride=10,
)
```

When `atom_selection` is provided, only the selected atoms are loaded
(atom indices are passed directly to mdtraj's reader, avoiding a
full-atom load followed by slicing).

### Multiple trajectories

```python
from mdpp.core import load_trajectories

trajs = load_trajectories(
    ["run1.xtc", "run2.xtc", "run3.xtc"],
    topology_paths=["topol.gro", "topol.gro", "topol.gro"],
    stride=5,
)
```

For parallel loading, set `max_workers` to use `multiprocessing.Pool`.
Processes are used instead of threads because mdtraj's C-level parsers
hold the GIL during decoding:

```python
trajs = load_trajectories(
    ["run1.xtc", "run2.xtc", "run3.xtc"],
    topology_paths=["topol.gro", "topol.gro", "topol.gro"],
    stride=10,
    max_workers=3,
)
```

| Method | Time | Speedup | Parent RSS delta |
| --- | --- | --- | --- |
| Sequential | 9.7 s | 1.0x | +16.8 MB |
| Threads (6) | 4.5 s | 2.2x | +7.7 MB |
| mp.Pool (6) | 0.9 s | 11.2x | +0.0 MB |

Worker processes allocate trajectory data in their own address space.
When the pool closes, that memory is fully released to the OS, leaving
zero RSS growth in the parent process.

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

## Time Extraction

Get the raw time array in picoseconds:

```python
from mdpp.core import trajectory_time_ps

time_ps = trajectory_time_ps(traj)
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
