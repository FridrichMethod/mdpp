# Visualize optional protein-only and ligand-only APBS maps in PyMOL.
#
# First generate component maps:
#   PQR_STEMS="complex protein ligand" bash complex/run_apbs.sh
#
# Then run from this directory:
#   pymol viz_components_apbs.pml

load tmp/complex/ambertools/protein.pqr, protein
load tmp/complex/apbs/protein.dx, protein_potential
load tmp/complex/ambertools/ligand.pqr, ligand
load tmp/complex/apbs/ligand.dx, ligand_potential

show cartoon, protein
show surface, protein
show sticks, ligand
show surface, ligand

ramp_new protein_esp, protein_potential, [-5, 0, 5], [red, white, blue]
ramp_new ligand_esp, ligand_potential, [-5, 0, 5], [red, white, blue]
set surface_color, protein_esp, protein
set surface_color, ligand_esp, ligand

isomesh protein_pos_mesh, protein_potential, 1.0
isomesh protein_neg_mesh, protein_potential, -1.0
isomesh ligand_pos_mesh, ligand_potential, 1.0
isomesh ligand_neg_mesh, ligand_potential, -1.0

color blue, protein_pos_mesh
color red, protein_neg_mesh
color blue, ligand_pos_mesh
color red, ligand_neg_mesh

orient protein or ligand
zoom protein or ligand, 8
