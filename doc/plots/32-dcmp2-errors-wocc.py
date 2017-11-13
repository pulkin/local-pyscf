import time

from pyscf import scf, mp

import dchf
from test_common import hydrogen_dimer_chain
from test_dchf import assign_domains

import fake
pyplot = fake.pyplot()

N = 24
model = hydrogen_dimer_chain(N)
ref_hf = scf.RHF(model)
ref_hf.conv_tol = 1e-12
ref_hf.kernel()

ref_mp2 = mp.MP2(ref_hf)
ref_mp2.kernel()
ref_e = ref_hf.e_tot - model.energy_nuc() + ref_mp2.emp2
ref_e2 = ref_mp2.emp2

configurations = [
    (2, 0),
    (4, 2),
    (6, 4),
    (12, 6),
]
x = list(range(len(configurations)))

mf_cache = {}

for wocc in (1, 0.5, 0):
    print "wocc={:.1f}".format(wocc)
    y = []
    y2 = []
    for c in configurations:
        print "  config={}".format(repr(c))

        if c in mf_cache:
            hf = mf_cache[c]
        else:
            hf = dchf.DCHF(model)
            assign_domains(hf, *c)
            hf.kernel(tolerance=1e-12, maxiter=30)
            mf_cache[c] = hf

        mp = dchf.DCMP2(hf, w_occ=wocc)
        mp.kernel()

        error = abs(ref_e - hf.hf_energy - mp.e2)
        y.append(error)
        y2.append(abs(ref_e2 - mp.e2))
        print "    result: {:.3e}".format(error)

    pyplot.semilogy(x, y, marker="o", label="w_occ={:.1f}".format(wocc))
    pyplot.semilogy(x, y2, marker="x", ls="--", label="w_occ={:.1f} (e2 error only)".format(wocc))

pyplot.xticks(x, list("{:d}({:d})".format(*i) for i in configurations))
pyplot.ylabel("Absolute error (Ry)")
pyplot.xlabel("Domain size (buffer size)")
pyplot.legend()

pyplot.show()
