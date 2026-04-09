# Cheminformatics

The `mdpp.chem` subpackage provides RDKit-based utilities for small-molecule analysis: descriptor calculation, structural filters, fingerprint generation, similarity computation, and file I/O.

## Molecule File I/O

### MolSupplier

Iterate over molecules from SDF, SMILES, or MOL2 files:

```python
from mdpp.chem import MolSupplier

for mol in MolSupplier("compounds.sdf"):
    print(mol.GetNumAtoms())
```

For large files, `MolSupplier` streams molecules lazily and skips unparseable entries.

## Molecular Descriptors

### Calculate descriptors

```python
from mdpp.chem import calc_descs, COMMON_DESC_NAMES

descs = calc_descs(mol)  # default: MW, LogP, HBA, HBD, Fsp3, RotBonds, Rings, TPSA, QED
mw, logp = calc_descs(mol, desc_names=("MolWt", "MolLogP"))
```

### Filter by descriptor ranges

```python
from mdpp.chem import filt_descs

passes = filt_descs(mol, filt={
    "MolWt": (200, 500),
    "MolLogP": (-1, 5),
    "NumHDonors": (0, 5),
})
```

## Structural Filters

### PAINS filter

Check whether a molecule matches Pan Assay Interference Compounds (PAINS) patterns:

```python
from mdpp.chem import is_pains

if is_pains(mol):
    print("PAINS alert detected")
```

### Murcko scaffold

Extract the Murcko framework:

```python
from mdpp.chem import get_framework

scaffold = get_framework(mol)
generic_scaffold = get_framework(mol, generic=True)

# Also works with SMILES strings
smiles_scaffold = get_framework("c1ccc(NC(=O)c2ccccc2)cc1")
```

## Fingerprints

### Generate fingerprints

```python
from mdpp.chem import gen_fp, FP_GENERATORS

fp = gen_fp(mol, fp_type="ecfp4")

# Available types
print(list(FP_GENERATORS.keys()))
# ['morgan', 'ecfp2', 'ecfp4', 'ecfp6', 'maccs', 'rdkit', 'atom_pair', 'topological_torsion']
```

### Cluster fingerprints (RDKit bulk similarity)

```python
from mdpp.chem import cluster_fps

fps = [gen_fp(mol) for mol in MolSupplier("compounds.sdf")]
result = cluster_fps(fps, cutoff=0.6, similarity_metric="tanimoto")
print(f"{result.n_clusters} clusters")
print(f"Largest cluster: {len(result.clusters[0])} molecules")
```

### Cluster fingerprints (Numba-parallel)

For large datasets, use the Numba-accelerated path with numpy bit arrays:

```python
import numpy as np
from mdpp.chem import cluster_fps_parallel

fps_array = np.array([list(fp.ToBitString()) for fp in fps], dtype=np.int8)
result = cluster_fps_parallel(fps_array, cutoff=0.6, similarity_metric="tanimoto")
```

## Similarity

### Pairwise similarity

```python
from mdpp.chem import calc_sim

sim = calc_sim(fp1, fp2, similarity_metric="tanimoto")
```

### Bulk similarity (one vs many)

```python
from mdpp.chem import calc_bulk_sim

sims = calc_bulk_sim(query_fp, database_fps, similarity_metric="tanimoto")
```

### Available metrics

```python
from mdpp.chem import SIM_FUNCS, CLUSTERING_SIM_METRICS

print(list(SIM_FUNCS.keys()))        # all similarity metrics
print(list(CLUSTERING_SIM_METRICS))   # metrics valid for clustering (self-sim = 1)
```
