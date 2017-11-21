from pyscf import scf

import dchf
from test_common import helium_chain, hydrogen_dimer_chain, hydrogen_distant_dimer_chain, atomic_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()


def calculate(model, domain_size, buffer_size=0):
    hf = dchf.DCHF(model)
    assign_chain_domains(hf, domain_size, buffer_size)

    hf.kernel(tolerance=1e-12)
    return hf.hf_energy, hf.dm

models = [
    (helium_chain(6), dict(domain_size=1), "He chain N=6"),
    (hydrogen_distant_dimer_chain(12), dict(domain_size=2), "H chain N=12 d=6.0"),
    (hydrogen_dimer_chain(12), dict(domain_size=2), "H dim chain N=12"),
    (hydrogen_dimer_chain(12), dict(domain_size=4, buffer_size=2), "... 3(0) config"),
    (hydrogen_dimer_chain(12), dict(domain_size=2, buffer_size=1), "... 2(1) config"),
    (hydrogen_dimer_chain(12), dict(domain_size=2, buffer_size=2), "... 2(2) config"),
    (hydrogen_dimer_chain(12), dict(domain_size=4, buffer_size=2), "... 4(2) config"),
]


names = []
e = []
dm = []

for model, options, label in models:
    print label
    mf = scf.RHF(model)
    mf.conv_tol = 1e-12
    mf.kernel()
    e_tot, hf_dm = calculate(model, **options)
    e.append(abs(mf.e_tot - model.energy_nuc() - e_tot)/model.natm)
    dm.append(abs(hf_dm - mf.make_rdm1()).max())
    names.append(label)

x = list(range(len(names)))
pyplot.figure(figsize=(16, 4.8))

pyplot.subplot(121)
pyplot.semilogy(x, e, marker="_", ls="None", markersize=30, mew=5)
pyplot.xticks(x, names, rotation=15)
pyplot.ylabel("Energy difference per atom (Ry)")
pyplot.grid(axis='y')

pyplot.subplot(122)
pyplot.semilogy(x, dm, marker="_", ls="None", markersize=30, mew=5)
pyplot.xticks(x, names, rotation=15)
pyplot.ylabel("Maximal density matrix deviation")
pyplot.grid(axis='y')

pyplot.show()
