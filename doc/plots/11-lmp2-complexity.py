import time
from scipy.optimize import curve_fit
import numpy

from pyscf import scf, mp

import lmp2
from test_common import helium_chain, hydrogen_dimer_chain

import fake
pyplot = fake.pyplot()

# Chains of He atoms spaced by 6A

chain_size = [8, 12, 16, 24, 32, 48, 64, 96, 128]
plot_chain_size_range = numpy.array([1, 2*max(chain_size)])

datasets = []

for label, constructor in (
        ("He chains", helium_chain),
        ("H chains", hydrogen_dimer_chain),
):
    # Calculate mean-field
    mf_models = []
    for N in chain_size:
        print "Calculating {}[{:d}] HF ...".format(label, N)
        model = constructor(N, max_memory=15e3)
        mf_models.append(scf.RHF(model))
        mf_models[-1].kernel()

    for driver, kwargs, label2, marker in (
        (mp.MP2, {}, "conventional MP2", "x"),
        (lmp2.LMP2, {}, "LMP2", "o"),
    ):
        times = []

        dt = None
        for mf in mf_models:
            if dt is None or dt < 100.0:
                print "Calculating {}[{:d}] {} ...".format(label, mf.mol.natm, label2)
                mp2 = driver(mf, **kwargs)

                t = time.time()
                mp2.kernel()
                dt = time.time() - t
                times.append(dt)
                print "  time {:.3f}s".format(times[-1])
                if "initialized_local_integral_provider" in dir(mp2) and "cache_factor" in dir(mp2.initialized_local_integral_provider):
                    print "  cache {:.1f}".format(mp2.initialized_local_integral_provider.cache_factor())

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
        datasets.append(dict(
            chain_size=chain_size[:len(times)],
            times=times,
            label=label + " (" + label2 + ") s={:.1f}".format(popt[1]),
            marker=marker,
        ))

pyplot.xlabel("Model size")
pyplot.ylabel("Run time (s)")
pyplot.grid()
pyplot.legend()
pyplot.show()
