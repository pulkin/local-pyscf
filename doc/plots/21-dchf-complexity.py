import time
from scipy.optimize import curve_fit
import numpy

from pyscf import scf

import dchf
from test_common import helium_chain, hydrogen_dimer_chain
from test_dchf import assign_domains

import fake
pyplot = fake.pyplot()

# Chains of He atoms spaced by 6A

chain_size = [8, 12, 16, 24, 32, 48, 64, 96, 128]
plot_chain_size_range = numpy.array([1, 2*max(chain_size)])


def setup(model):
    hf = dchf.DCHF(model)
    if model.atom[0][0] == "He":
        assign_domains(hf, 1, 0)
    elif model.atom[0][0] == "H":
        assign_domains(hf, 4, 2)
    else:
        raise ValueError("Model not recognized")
    return hf


def setup_full(model):
    hf = dchf.DCHF(model)
    assign_domains(hf, model.natm, 0)
    return hf


for label, constructor in (
        ("He chains", helium_chain),
        ("H chains", hydrogen_dimer_chain),
):
    for driver, label2, marker in (
        (scf.RHF, "pyscf HF", "x"),
        (setup, "DCHF", "o"),
        (setup_full, "custom HF", "+")
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

        f = lambda x, a, p: a*(x**p)
        try:
            popt = curve_fit(f, chain_size[:len(times)], times, p0=(5, times[-1]/chain_size[len(times)-1]**5))[0]
        except RuntimeError:
            popt = [float("nan"), float("nan")]

        pyplot.loglog(
            chain_size[:len(times)],
            times,
            label=label+" ("+label2+") s={:.1f}".format(popt[1]),
            marker=marker,
        )

pyplot.xlabel("Model size")
pyplot.ylabel("Run time (s)")
pyplot.grid()
pyplot.legend()
pyplot.show()
