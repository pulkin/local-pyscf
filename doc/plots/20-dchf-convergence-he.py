import dchf
from test_common import helium_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()

maxiter=30

def calculate(model, domain_size, buffer_size=0, diis=True):
    hf = dchf.DCHF(model)
    assign_chain_domains(hf, domain_size, buffer_size)

    # Since the convergence criterion is never met an exception will be raised
    try:
        hf.kernel(tolerance=0, maxiter=maxiter, fock_hook="diis" if diis else None)
    except RuntimeError:
        pass
    return hf.convergence_history


x = list(range(1, maxiter+1))

for i in (6, 12, 18):
    y = calculate(helium_chain(i), 1)
    pyplot.semilogy(x, y, marker="o", label="N={:d} DIIS".format(i))
    y = calculate(helium_chain(i), 1, diis=False)
    pyplot.semilogy(x, y, marker="o", label="N={:d} plain".format(i))

pyplot.xlabel("Step")
pyplot.ylabel("Error in the density matrix")
pyplot.grid(axis='y')
pyplot.legend()
pyplot.show()
