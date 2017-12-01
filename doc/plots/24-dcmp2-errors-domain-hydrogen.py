import time

from pyscf import scf, mp, cc

import dchf
from test_common import hydrogen_dimer_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()

N = 24
model = hydrogen_dimer_chain(N)
print "HF ..."
ref_hf = scf.RHF(model)
ref_hf.conv_tol = 1e-12
ref_hf.kernel()
ref_e = ref_hf.e_tot

print "MP2 ..."
ref_mp2 = mp.MP2(ref_hf)
ref_mp2.kernel()
ref_e2 = ref_mp2.e_corr

print "CCSD ..."
ref_ccsd = cc.CCSD(ref_hf)
ref_ccsd.kernel()
ref_e2_cc = ref_ccsd.e_corr

times_hf = []
times_mp2 = []
times_ccsd = []
titles = []
errors_hf = []
errors_mp2 = []
errors_ccsd = []

pyplot.figure(figsize=(20, 4.8))
for domain_size in [2, 4, 6, 8, 12, 24]:
    print "Size={:d}".format(domain_size)
    x = list(sorted(set([0, 2, (domain_size//4)*2, domain_size])))
    for buffer_size in x:
        print "  buff={:d}".format(buffer_size)

        t = time.time()
        hf = dchf.DCHF(model)
        assign_chain_domains(hf, domain_size, buffer_size)
        hf.kernel(tolerance=1e-9, maxiter=30)
        t = time.time()-t
        times_hf.append(t)

        titles.append("{:d}({:d})".format(domain_size, buffer_size))
        errors_hf.append(abs(ref_e - hf.e_tot))

        t = time.time()
        mp = dchf.DCMP2(hf)
        mp.kernel()
        t = time.time()-t
        times_mp2.append(t)

        errors_mp2.append(abs(ref_e2 - mp.e2))

        t = time.time()
        ccsd = dchf.DCCCSD(hf)
        ccsd.kernel()
        t = time.time() - t
        times_ccsd.append(t)

        errors_ccsd.append(abs(ref_e2_cc - ccsd.e2))

        if buffer_size+domain_size >= N:
            break

pyplot.subplot(131)
pyplot.semilogy(times_hf, errors_hf, marker="o", ls="None", label="HF")
for x, y, t in zip(times_hf, errors_hf, titles):
    pyplot.annotate(
        t, xy=(x, y), xytext=(5, 0), textcoords='offset points', ha='left', va='center',
    )
pyplot.xlabel("Run time (s)")
pyplot.ylabel("Absolute error (Ry)")
pyplot.legend()

pyplot.subplot(132)
pyplot.semilogy(times_mp2, errors_mp2, marker="o", ls="None", label="MP2")
for x, y, t in zip(times_mp2, errors_mp2, titles):
    pyplot.annotate(
        t, xy=(x, y), xytext=(5, 0), textcoords='offset points', ha='left', va='center',
    )
pyplot.xlabel("Run time (s)")
pyplot.ylabel("Absolute error (Ry)")
pyplot.legend()

pyplot.subplot(133)
pyplot.semilogy(times_ccsd, errors_ccsd, marker="o", ls="None", label="CCSD")
for x, y, t in zip(times_ccsd, errors_ccsd, titles):
    pyplot.annotate(
        t, xy=(x, y), xytext=(5, 0), textcoords='offset points', ha='left', va='center',
    )
pyplot.xlabel("Run time (s)")
pyplot.ylabel("Absolute error (Ry)")
pyplot.legend()

pyplot.show()
