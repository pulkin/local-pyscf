import time
from scipy.optimize import curve_fit
import numpy

from pyscf import scf

import dchf
from test_common import helium_chain, hydrogen_dimer_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()

# Chains of He atoms spaced by 6A

chain_size = [8, 12, 16, 24, 32, 48, 64, 96, 128]
plot_chain_size_range = numpy.array([1, 2*max(chain_size)])


def setup(model):
    hf = dchf.DCHF(model)
    if model.atom[0][0] == "He":
        assign_chain_domains(hf, 1, 0)
    elif model.atom[0][0] == "H":
        assign_chain_domains(hf, 4, 2)
    else:
        raise ValueError("Model not recognized")
    return hf


def setup_full(model):
    hf = dchf.DCHF(model)
    assign_chain_domains(hf, model.natm, 0)
    return hf


def get_scaling(x, y):
    while True:
        try:
            return curve_fit(lambda x, a, p: a * (x ** p), x, y, p0=(5, y[-1]/x[-1]**5))[0]
        except RuntimeError:
            x, y = x[1:], y[1:]
            if len(x)<2:
                return float("nan")


for label, constructor in (
        ("He chains", helium_chain),
        ("H chains", hydrogen_dimer_chain),
):
    for driver, label2, marker, ls in (
        (scf.RHF, "pyscf HF", "x", "--"),
        (setup, "DCHF", "o", None),
        (setup_full, "custom HF", "+", "--")
    ):
        dt = None

        times = []

        for N in chain_size:

            if dt is None or dt < 10.0:

                print "Calculating {}[{:d}] {} ...".format(label, N, label2)

                model = constructor(N, max_memory=15e3)

                t = time.time()
                dr = driver(model)
                dr.kernel()
                dt = time.time() - t

                times.append(dt)
                print "  time {:.3f}s".format(times[-1])

        popt = get_scaling(chain_size[:len(times)][-3:], times[-3:])

        pyplot.loglog(
            chain_size[:len(times)],
            times,
            label=label+" ("+label2+") s={:.1f}".format(popt[1]),
            marker=marker,
            ls=ls,
        )

pyplot.xlabel("Model size")
pyplot.ylabel("Run time (s)")
pyplot.grid()
pyplot.legend()
pyplot.show()
