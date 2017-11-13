from pyscf import scf

import lmp2
from test_common import helium_chain, hydrogen_dimer_chain, hydrogen_distant_dimer_chain
from test_lmp2 import iter_local_dummy

import fake
pyplot = fake.pyplot()

# Maximal number of iterations
maxiter = 30


def calculate(model, **kwargs):
    mf = scf.RHF(model)
    mf.kernel()
    mp2 = lmp2.LMP2(mf, **kwargs)

    # Since the convergence criterion is never met an exception will be raised
    try:
        mp2.kernel(tolerance=0, maxiter=maxiter)
    except RuntimeError:
        pass
    return mp2.convergence_history


# Chain of He atoms spaced by 6A
he6 = calculate(helium_chain(6))
# Chain of dimerized H atoms
h6 = calculate(hydrogen_dimer_chain(6))
# Chain of dimerized H atoms with a larger spacing
h6_d2 = calculate(hydrogen_distant_dimer_chain(6))
# LMP2 run resembling conventional MP2 with localization transformation
h6_d2_ns = calculate(hydrogen_distant_dimer_chain(6), local_space_provider=iter_local_dummy)
# LMP2 run resembling conventional MP2
h6_d2_ns_nl = calculate(
    hydrogen_dimer_chain(6),
    local_space_provider=iter_local_dummy,
    localization_provider=None,
)


x = list(range(1, maxiter+1))
pyplot.semilogy(x, he6, marker="o", label="He chain N=6")
pyplot.semilogy(x, h6, marker="o", label="H chain N=6 d=2.3")
pyplot.semilogy(x, h6_d2, marker="o", label="H chain N=6 d=6.0")
pyplot.semilogy(x, h6_d2_ns, marker="o", label="H chain N=6 non-sparse")
pyplot.semilogy(x, h6_d2_ns_nl, marker="o", label="H chain N=6 non-sparse non-localized")

pyplot.xlabel("Step")
pyplot.ylabel("Maximal absolute change in amplitudes")
pyplot.grid(axis='y')
pyplot.legend()
pyplot.show()
