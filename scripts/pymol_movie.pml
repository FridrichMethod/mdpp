mset; rewind
mset 1x300
mview store, 1, state=1, object=step5_1_complex_fit
mview store, 300, state=-1, object=step5_1_complex_fit
intra_fit step5_1_complex_fit
smooth