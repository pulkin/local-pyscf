from pyscf import scf

from dchf import DCHF

import numpy

from plot_common import load_pyscf_cluster_model, draw_cluster_model
import fake
pyplot = fake.pyplot()

tolerance = 1e-5

draw_cluster_model("alkane-8", width=800, height=200)

dchf = load_pyscf_cluster_model("alkane-8", isolated_cluster=True)
dchf.__mol__.verbose = 4
dchf.kernel()

hf = scf.RHF(dchf.__mol__)
hf.kernel()
dm_ref = hf.make_rdm1()

print "DM error:", abs(dchf.dm - dm_ref).max()
mask = numpy.logical_not(dchf.domains_pattern(2))
print "DM intrinsic error:", abs(dm_ref*mask).max()
print "Energy diff:", abs(dchf.e_tot - hf.e_tot)

pyplot.figure(figsize=(12, 4.8))
for hf, subplot, title in (
        (hf, 121, "HF"),
        (dchf, 122, "DC-HF"),
):
    pyplot.subplot(subplot)
    pyplot.imshow(abs(hf.dm if isinstance(hf, DCHF) else hf.make_rdm1()))
    pyplot.title(title)
    pyplot.colorbar()

pyplot.show()