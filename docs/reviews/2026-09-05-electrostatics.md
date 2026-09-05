# APBS / BrownDye scientific and implementation review

Reviewed on 2026-09-05 against baseline `eecb097`, on branch
`science-review/electrostatics`. Scope: all three APBS notebooks, the BrownDye
preparation notebook, their READMEs and visualization/runner scripts, and the
APBS/BrownDye/protein preparation helpers they call. The review prioritized
ordinary workflow and scientific interpretation errors, rather than rare input
corner cases. Primary literature and upstream manuals/source were checked online
and, where available, against installed programs.

## Findings and changes

| ID | Severity | Confirmed problem | Change and practical consequence |
|---|---|---|---|
| E1 | High for protonation-sensitive use | Protein preparation added pH-dependent hydrogens, but `pdb4amber -y` removed them before tleap selected its residue templates. PROPKA was reported but not applied. A reported pH or pKa check therefore did not establish the charge state of the final Amber topology. | `fix_pdb(..., residue_names="amber")` now encodes the actual hydrogen topology as ASH/GLH/LYN/HID/HIE/HIP and distinguishes free CYM from disulfide CYX by S-S connectivity. Both protein-containing notebooks explicitly select the PROPKA policy and Amber names. Hydrogen rebuilding then preserves those template choices. |
| E2 | Medium | APBS maps used solute dielectric 2.0 and solvent 78.54, whereas the BrownDye example used body dielectric 4.0 and default solvent 78.0. Helper documentation incorrectly described dielectric constants as kT units. The first available Debye value was accepted even if the other map came from different salt conditions. | Dielectrics are documented as dimensionless relative permittivities. Each successful APBS run publishes the settings used to generate its map; BrownDye reads each body's solute dielectric and checks shared solvent settings. All supplied Debye values must be finite, positive and mutually consistent. Exponent notation is parsed correctly. |
| E3 | Medium | Antechamber/tleap selected GAFF2 but `parmchk2` omitted its force-field selector; the installed upstream help confirms that omission selects GAFF. This can mix bonded parameter families in the exported topologies. | Both ligand-containing notebooks use `parmchk2 ... -s 2`. No claim is made that this flag alone changes the current PQR electrostatic charges: it fixes the topology parameterization workflow. |
| E4 | Medium | The grid-size helper accepted low-level `c*2**n+1` counts such as 3, 17 or 81 even though the default four-level multigrid solver requires `c*32+1`; APBS can adjust unsuitable counts downward and coarsen the requested mesh. A fixed maximum candidate could also violate the requested spacing. | Round up directly to a positive multiple of 32 plus one and check physical spacing arguments. Tests enforce the spacing contract across box sizes; real APBS confirms the generated 33-cubed grid runs with four levels. Existing ordinary default-size maps are not asserted to have been incorrect. |
| E5 | Medium | Preparation allowed a complex paired with one of its overlapping components, and only printed the number of reference pairs even if fewer than the reaction required. Live PQR/DX symlinks could change after `atoms.xml` and simulation inputs were generated. | The bundled example accepts only the disjoint protein/ligand pair, aborts when too few reference pairs exist, and snapshots the input PQR/DX/log/settings. PQR is published with each successful APBS map and consumed from that completed bundle; a failed APBS rerun cannot mix a new AmberTools PQR with an old map. Re-running APBS no longer changes an already prepared BrownDye input beneath it. |
| E6 | Medium, execution | The trajectory visualization cell used obsolete `vtf_trajectory -mol0/-mol1/-trajf/-trial/-traj` arguments. The installed BrownDye2 converter reads the decompressed trajectory from stdin. | Use `vtf_trajectory <trajectory.xml >trajectory.vtf`. A real one-atom, two-frame fixture converts successfully. |
| E7 | Interpretation / best practice | The receptor was described as held fixed; the generated two-body model does not immobilize it. Fixed padding was described as guaranteeing BrownDye far-field coverage. The broad 10 Å / three-contact reaction could be read as a validated binding boundary. | Correct the relative-diffusion and grid-coverage statements. Explain encounter rate versus experimental binding `k_on`, units, confidence intervals, rigid-ligand assumptions, censored/stuck trajectories and sensitivity checks. The reaction threshold remains an explicitly unvalidated example choice. |

E1 is a workflow correctness finding, not evidence of a particular wrongly
charged residue in the bundled protein. A real PROPKA 3.5.1 calculation found
85 titratable groups and **zero nonstandard model-pKa disagreements at pH 7.4**;
it also warned about the generated interaction atoms at **ARG 263, chain A**.
That warning should be inspected before scientific use. The upstream OpenMM
`CYX` *hydrogen variant* explicitly covers both thiolate and disulfide, so its
existing use in `_propka_variants` was **not** classified as a bug or changed.
Only the downstream Amber residue-name distinction was added.

The reusable notebook fixtures now use **298 K**, matching BrownDye's base
energy unit and kT=1. The standalone APBS helper retains its public 298.15 K
default. The former 0.15 K difference was not treated as a material scientific
error; the explicit convention prevents larger user temperature changes from
silently mixing incompatible potential units. General non-298-K BrownDye
potential conversion and solvent-viscosity modeling are outside this example.

## Literature and upstream evidence

- [OpenMM Modeller API](https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html): protonation variants, hydrogen rebuilding and cysteine/disulfide semantics. Installed `openmm/app/modeller.py` and `data/hydrogens.xml` were also inspected.
- [Olsson et al., PROPKA3, JCTC 2011, DOI 10.1021/ct100578z](https://pubmed.ncbi.nlm.nih.gov/26596171/): environment-dependent pKa prediction is an empirical model; it is not a replacement for inspecting the selected state and structural assumptions.
- [Amber developers' TYK2 parameterization tutorial](https://ambertutorials-rutgerslbsr-c744272d5a9c1169e0dc9e19b8d800019105.gitlab.io/workshop/03_nonStandardPars/TYK2/tyk2_ligand_setup.html): GAFF2 parameter generation uses `parmchk2 -s 2`. Local `parmchk2 -h` independently confirmed GAFF is the default and 2 selects GAFF2.
- [APBS grid dimensions](https://apbs.readthedocs.io/en/nathan-docs/using/input/elec/dime.html): four multigrid levels imply counts of the form `c*32+1`; inappropriate counts can be reduced, losing resolution.
- [APBS potential output](https://apbs.readthedocs.io/en/nathan-docs/using/input/elec/write.html): DX potential units are kT/e at the calculation temperature.
- [APBS solvation-energy example](https://apbs.readthedocs.io/en/latest/using/examples/solvation-energies.html): quantitative energy interpretation requires appropriate reference calculations; a total potential/energy calculation alone is not a binding free energy.
- [BrownDye2 manual, January 2026](https://browndye.ucsd.edu/browndye2.pdf): relative permittivities, 298-K energy unit, encounter criteria, relative diffusion, NAM/weighted-ensemble estimators, trajectory conversion and stuck-trajectory termination. Installed `/apps/browndye2/doc/browndye2.tex` and `aux/vtf_trajectory.ml` were checked as well.
- [Huber and McCammon, CPC 2010, DOI 10.1016/j.cpc.2010.07.022](https://pmc.ncbi.nlm.nih.gov/articles/PMC2994412/): diffusional encounters and reaction-rate estimation; an encounter criterion is part of the modeled observable.

## Validation and reproducibility

- `conda run -n mdpp pytest tests/prep tests/examples -n 0`: **81 passed** after final fixes and formatting. Tests exercise final Amber labels, free thiolate versus S-S connectivity, finite/consistent Debye lengths, APBS spacing, notebook solvent checks, stable snapshots, impossible-reaction rejection, and Python/Bash syntax of every reviewed notebook code cell.
- Real Amber smoke in `/tmp/mdpp-electrostatics-smoke`: construct ALA-GLU-ALA, apply a controlled pKa=8.0 override to internal GLU at pH 7, write GLH labels, run `pdb4amber -y`, then load with ff19SB in tleap. The resulting topology charge is **0.000000 e**, preserving neutral GLH; tleap reported **zero errors and warnings**. A charged GLU in this peptide would give -1 e. This is a controlled integration fixture, not a measurement of the supplied protein's pKa.
- Real **APBS 3.4.1** smoke on one ion: generated `dime 33 33 33`, solver reported four multigrid levels, Debye length 7.8566 Å, wrote a nonempty DX map and exited 0. It also printed `asc_getToken` / `Vio_scanf` input warnings despite normal completion; the smoke establishes compatibility, not a numerically converged reference solution.
- Real installed BrownDye `pqr2xml` plus `vtf_trajectory` converted a one-atom trajectory with positions (0,0,0) and (1,0,0) Å into two VTF frames through stdin. No BrownDye association trajectories were propagated.
- Raw coordinate registration: the 2,465 protein atoms and 38 ligand atoms exactly match their respective chains in `complex.pdb` (maximum coordinate difference **0.0 Å**). Prepared Amber coordinate invariance was not measured for the full example.
- Environment: Python 3.13.12, OpenMM 8.4.0, PDBFixer 1.12.0, PROPKA 3.5.1, ParmEd 4.3.1, pytest 9.0.3. Regression tests use controlled synthetic settings; the Amber fixture controls its PROPKA result explicitly. BrownDye example seed remains 11111111.
- `conda run -n mdpp pre-commit run --all-files`: all hooks passed, including ruff, mypy, shellcheck and notebook checks. Independent source/notebook review found and closed the incomplete-APBS-rerun PQR/map consistency issue; final review reported no remaining HIGH or CRITICAL findings. The parent review report records the final integrated test suite.

## No confirmed error / remaining scientific limits

The supplied structures share a coordinate frame; Amber/ParmEd PQR export uses
atomic charges and radii, and the visualizations label potential consistently
in kT/e. The NAM and weighted-ensemble runner commands agree with the upstream
manual. No change to those commands or to stored PDB coordinates was needed.

The full APBS parameterization notebooks and production BrownDye simulations
were not re-run. No conclusion is claimed about converged potentials, rates,
force-field quality, ligand microstate or the physically correct contact
criterion. Linear PB, mbondi3 radii, dielectric choices, repaired protein
regions and a rigid ligand are assumptions to test for the intended system.
Unsupported PROPKA states and ligand-induced pKa shifts still require explicit
assessment; isolated-protein predictions do not determine bound-state titration.
Statistical rate intervals exclude those modeling uncertainties.
