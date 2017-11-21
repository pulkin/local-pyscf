from pyscf import scf, mp, cc

import dchf
from test_common import helium_chain, hydrogen_dimer_chain, hydrogen_distant_dimer_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()


def calculate(model, domain_size, buffer_size=0):
    hf = dchf.DCHF(model)
    assign_chain_domains(hf, domain_size, buffer_size)
    hf.kernel(tolerance=1e-12)

    mp = dchf.DCMP2(hf)
    mp.kernel()

    cc = dchf.DCCCSD(hf)
    cc.kernel()
    return hf.hf_energy, mp.e2, cc.e2

models = [
    (helium_chain(6), dict(domain_size=1), "He chain N=6"),
    (hydrogen_distant_dimer_chain(12), dict(domain_size=2), "H chain N=12 d=6.0"),
    (hydrogen_dimer_chain(12), dict(domain_size=2), "H dim chain N=12"),
    (hydrogen_dimer_chain(12), dict(domain_size=2, buffer_size=2), "... 2(2) config"),
    (hydrogen_dimer_chain(12), dict(domain_size=4, buffer_size=2), "... 4(2) config"),
]


names = []
e = []
e2_mp = []
e2_cc = []

for model, options, label in models:
    print label
    mf = scf.RHF(model)
    mf.conv_tol = 1e-12
    mf.kernel()

    mp2 = mp.MP2(mf)
    mp2.kernel()

    ccsd = cc.CCSD(mf)
    ccsd.kernel()

    e_hf, e_mp, e_cc = calculate(model, **options)
    e.append(abs(mf.e_tot - model.energy_nuc() - e_hf)/model.natm)
    e2_mp.append(abs(mp2.emp2 - e_mp) / model.natm)
    e2_cc.append(abs(ccsd.e_corr - e_cc) / model.natm)
    names.append(label)

x = list(range(len(names)))
pyplot.figure(figsize=(8, 4.8))
pyplot.semilogy(x, e, marker="_", ls="None", markersize=30, label="HF", mew=5)
pyplot.semilogy(x, e2_mp, marker="_", ls="None", markersize=30, label="MP2", mew=5)
pyplot.semilogy(x, e2_cc, marker="_", ls="None", markersize=30, label="CCSD", mew=5)

pyplot.xticks(x, names, rotation=15)
pyplot.ylabel("Energy difference per atom (Ry)")
pyplot.grid(axis='y')
pyplot.legend()
pyplot.show()
