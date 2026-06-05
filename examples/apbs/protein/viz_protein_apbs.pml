# Visualize the protein APBS map in PyMOL.
#
# Run from this directory (examples/apbs/protein):
#   pymol viz_protein_apbs.pml

load tmp/ambertools/protein.pqr, protein
load tmp/apbs/protein.dx, protein_potential

show cartoon, protein
show surface, protein

ramp_new protein_esp, protein_potential, [-5, 0, 5], [red, white, blue]
set surface_color, protein_esp, protein

isomesh protein_pos_mesh, protein_potential, 1.0
isomesh protein_neg_mesh, protein_potential, -1.0
color blue, protein_pos_mesh
color red, protein_neg_mesh

orient protein
zoom protein, 8
