from pyscf import scf, mp

import lmp2
from test_common import helium_chain, hydrogen_dimer_chain, hydrogen_distant_dimer_chain

import fake
pyplot = fake.pyplot()

models = (
    ("He chain N=6", helium_chain(6)),
    ("H dim chain N=6 d=2.3", hydrogen_dimer_chain(6)),
    ("H dim chain N=6 d=6.0", hydrogen_distant_dimer_chain(6)),
)

names = []
e = []

for name, model in models:
    mf = scf.RHF(model)
    mf.kernel()
    a = lmp2.LMP2(mf)
    a.kernel()
    b = mp.MP2(mf)
    b.kernel()
    e.append(1e3*(a.emp2 - b.emp2)/model.natm)
    names.append(name)

x = list(range(len(names)))
pyplot.bar(x, e)

pyplot.xticks(x, names)
pyplot.ylabel("Energy difference per atom, mRy")
pyplot.grid(axis='y')
pyplot.axhline(y=0, color="black")
pyplot.show()
