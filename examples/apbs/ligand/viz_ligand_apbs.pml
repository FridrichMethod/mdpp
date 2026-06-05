# Visualize the ligand APBS map in PyMOL.
#
# Run from this directory (examples/apbs/ligand):
#   pymol viz_ligand_apbs.pml

load tmp/ambertools/ligand.pqr, ligand
load tmp/apbs/ligand.dx, ligand_potential

show sticks, ligand
show surface, ligand

ramp_new ligand_esp, ligand_potential, [-5, 0, 5], [red, white, blue]
set surface_color, ligand_esp, ligand

isomesh ligand_pos_mesh, ligand_potential, 1.0
isomesh ligand_neg_mesh, ligand_potential, -1.0
color blue, ligand_pos_mesh
color red, ligand_neg_mesh

orient ligand
zoom ligand, 8
