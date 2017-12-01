from pyscf import scf, mp, cc

import dchf
from test_common import hydrogen_dimer_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()

N = 24
tolerance = 1e-12

model = hydrogen_dimer_chain(N)
ref_hf = scf.RHF(model)
ref_hf.conv_tol = tolerance
ref_hf.kernel()

ref_mp2 = mp.MP2(ref_hf)
ref_mp2.kernel()
ref_mp2_e2 = ref_mp2.emp2

ref_ccsd = cc.CCSD(ref_hf)
ref_ccsd.kernel()
ref_ccsd_e2 = ref_ccsd.e_corr

configurations = [
    (2, 0),
    (4, 2),
    (8, 4),
#    (12, 6),
]
x = list(range(len(configurations)))

mf_cache = {}

for driver, subplot, title, reference in (
        (dchf.DCMP2, 121, "DC-MP2", ref_mp2_e2),
        (dchf.DCCCSD, 122, "DC-CCSD", ref_ccsd_e2),
):
    print "==", title, "=="
    pyplot.subplot(subplot)

    for wocc in (1, 0.5, 0):
        print "wocc={:.1f}".format(wocc)
        y = []
        for c in configurations:
            print "  config={}".format(repr(c))

            if c in mf_cache:
                hf = mf_cache[c]
            else:
                hf = dchf.DCHF(model)
                assign_chain_domains(hf, *c)
                hf.kernel(tolerance=tolerance)
                mf_cache[c] = hf

            mp = driver(hf, w_occ=wocc)
            mp.kernel()

            y.append(abs(reference - mp.e2))
            print "  {:.3e}".format(y[-1])

        pyplot.semilogy(x, y, marker="x", label="w_occ={:.1f}".format(wocc))

    pyplot.xticks(x, list("{:d}({:d})".format(*i) for i in configurations))
    pyplot.ylabel("Absolute error (Ry)")
    pyplot.xlabel("Domain size (buffer size)")
    pyplot.legend()
    pyplot.title(title)

pyplot.show()
