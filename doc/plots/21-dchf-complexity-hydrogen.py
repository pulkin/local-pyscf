import time
from scipy.optimize import curve_fit
import numpy

from pyscf import scf

import dchf
from test_common import hydrogen_dimer_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()

# Dimerized chains of hydrogen atoms

chain_size = [8, 12, 16, 24, 32, 48, 64, 96, 128]
plot_chain_size_range = numpy.array([1, 2*max(chain_size)])


def setup(model):
    hf = dchf.DCHF(model)
    assign_chain_domains(hf, 4, 2)
    return hf


def setup_full(model):
    hf = dchf.DCHF(model)
    assign_chain_domains(hf, model.natm, 0)
    return hf


def get_scaling(x, y):
    while True:
        try:
            return curve_fit(lambda x, a, p: a * (x ** p), x, y, p0=(4, y[-1]/x[-1]**4))[0]
        except RuntimeError:
            print "Re-trying fit with {:d} points".format(len(x))
            x, y = x[1:], y[1:]
            if len(x) < 2:
                return float("nan")


for driver, label2, marker in (
    (setup, "DCHF", "o"),
    (scf.RHF, "pyscf HF", "x"),
    (setup_full, "custom HF", "+")
):
    dt = None

    times = []

    for N in chain_size:

        if dt is None or dt < 10.0:

            print "Calculating [{:d}] {} ...".format(N, label2)

            model = hydrogen_dimer_chain(N, max_memory=15e3)

            t = time.time()
            dr = driver(model)
            dr.kernel()
            dt = time.time() - t

            times.append(dt)
            print "  {:.3f} s".format(dt)

    popt = get_scaling(chain_size[:len(times)][-3:], times[-3:])

    pyplot.loglog(
        chain_size[:len(times)],
        times,
        label=label2+" pow={:.1f}".format(popt[1]),
        marker=marker,
    )

pyplot.xlabel("Model size")
pyplot.ylabel("Run time (s)")
pyplot.legend()
pyplot.show()
