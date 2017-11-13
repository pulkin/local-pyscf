from pyscf import scf

import dchf
from test_common import helium_chain, hydrogen_dimer_chain, hydrogen_distant_dimer_chain
from test_dchf import assign_domains

import fake
pyplot = fake.pyplot()

# Maximal number of iterations
maxiter = 30


def calculate(model, domain_size, buffer_size=0):
    hf = dchf.DCHF(model)
    assign_domains(hf, domain_size, buffer_size)

    # Since the convergence criterion is never met an exception will be raised
    try:
        hf.kernel(tolerance=0, maxiter=maxiter)
    except RuntimeError:
        pass
    return hf.convergence_history


models = [
    (helium_chain(6), dict(domain_size=1), "He chain N=6"),
    (hydrogen_distant_dimer_chain(12), dict(domain_size=2), "H chain N=12 d=6.0"),
    (hydrogen_dimer_chain(12), dict(domain_size=2), "H dim chain N=12"),
    (hydrogen_dimer_chain(12), dict(domain_size=2, buffer_size=2), "+ buffer"),
    (hydrogen_dimer_chain(12), dict(domain_size=4, buffer_size=2), "+ larger domain size"),
]


x = list(range(1, maxiter+1))
for model, options, label in models:
    y = calculate(model, **options)
    pyplot.semilogy(x, y, marker="o", label=label)

pyplot.xlabel("Step")
pyplot.ylabel("Maximal absolute change in amplitudes")
pyplot.grid(axis='y')
pyplot.legend()
pyplot.show()
