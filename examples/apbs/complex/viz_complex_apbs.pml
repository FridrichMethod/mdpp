# Visualize the full complex APBS map in PyMOL.
#
# Run from this directory (examples/apbs/complex):
#   pymol viz_complex_apbs.pml

load tmp/ambertools/complex.pqr, complex
load tmp/apbs/complex.dx, complex_potential

show cartoon, complex
show sticks, organic
show surface, complex

ramp_new complex_esp, complex_potential, [-5, 0, 5], [red, white, blue]
set surface_color, complex_esp, complex

isomesh complex_pos_mesh, complex_potential, 1.0
isomesh complex_neg_mesh, complex_potential, -1.0
color blue, complex_pos_mesh
color red, complex_neg_mesh

orient complex
zoom complex, 8
