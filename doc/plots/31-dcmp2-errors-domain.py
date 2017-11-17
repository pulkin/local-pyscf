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

times = []
titles = []
errors = []

pyplot.figure(figsize=(12, 4.8))
pyplot.subplot(121)
for domain_size in [2, 4, 6, 8, 12, 24]:
    print "Size={:d}".format(domain_size)
    x = list(sorted(set([0, 2, (domain_size//4)*2, domain_size])))
    y = []
    for buffer_size in x:
        print "  buff={:d}".format(buffer_size)

        t = time.time()

        hf = dchf.DCHF(model)
        assign_domains(hf, domain_size, buffer_size)
        hf.kernel(tolerance=1e-12, maxiter=30)

        mp = dchf.DCMP2(hf)
        mp.kernel()

        times.append(time.time()-t)

        error = abs(ref_e - hf.hf_energy - mp.e2)

        titles.append("{:d}({:d})".format(domain_size, buffer_size))
        errors.append(error)
        y.append(error)
        print "    result: {:.3e}".format(error)

        if buffer_size+domain_size >= N:
            break

    pyplot.semilogy(x[:len(y)], y, marker="o", label="domain size {:d}".format(domain_size))

pyplot.grid(axis='y')
pyplot.xlabel("Buffer size (atoms)")
pyplot.ylabel("Absolute error (Ry)")
pyplot.legend()

pyplot.subplot(122)
pyplot.semilogy(times, errors, marker="o", ls="None", label="domain size (core size)")
for x, y, t in zip(times, errors, titles):
    pyplot.annotate(
        t, xy=(x, y), xytext=(5, 0), textcoords='offset points', ha='left', va='center',
    )
pyplot.xlabel("Run time (s)")
pyplot.ylabel("Absolute error (Ry)")
pyplot.legend()
pyplot.show()
