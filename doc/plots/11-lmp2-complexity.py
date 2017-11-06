import time
from scipy.optimize import curve_fit
from matplotlib import pyplot

from pyscf import scf, mp

import lmp2
from test_common import atomic_chain

# Chains of He atoms spaced by 6A

chain_size = [8, 12, 16, 24, 32, 48, 64, 96, 128]

time_mp2_conventional = []
time_lmp2 = []
energy_mp2_conventional = []
energy_lmp2 = []

for N in chain_size:
    print N
    model = atomic_chain(N, alt_spacing=2.3)
    model.verbose = 3

    mf = scf.RHF(model)

    t = time.time()
    mf.kernel()
    print "MF done"
    mf_time.append(time.time() - t)

    mp2 = lmp2.LMP2(mf)

    t = time.time()
    mp2.kernel()
    print "LMP2 done"
    time_lmp2.append(time.time() - t)
    energy_lmp2.append(mp2.emp2)

    mp2 = mp.MP2(mf)

    if N<65:
        t = time.time()
        mp2.kernel()
        print "MP2 done"
        time_mp2_conventional.append(time.time() - t)
        energy_mp2_conventional.append(mp2.emp2)

print chain_size
print time_mp2_conventional
print time_lmp2

# He
chain_size = [16, 24, 32, 48, 64, 96, 128]
time_mp2_conventional = [0.24232983589172363, 1.7183690071105957, 7.13835597038269, 96.93113303184509, 457.7666389942169]
time_lmp2 = [1.0663118362426758, 2.8978991508483887, 6.474092960357666, 22.09022092819214, 51.9230899810791, 177.98335003852844, 447.09027194976807]

# H
# sizes = [16, 24, 32, 48, 64, 96, 128]
# mp2_time = [0.11144304275512695, 0.7975170612335205, 3.6849119663238525, 52.98084616661072, 208.3274049758911]
# lmp2_time = [4.574473857879639, 10.74223780632019, 19.900940895080566, 13.833334922790527, 29.11626696586609, 87.67919707298279, 213.73089504241943]


f = lambda x, a, p: a*(x**p)
popt_mp2 = curve_fit(f, chain_size[:len(time_mp2_conventional)], time_mp2_conventional)[0]
popt_lmp2 = curve_fit(f, chain_size[:len(time_lmp2)], time_lmp2)[0]

pyplot.figure()
#pyplot.loglog(sizes)], mf_time, label="HF")
pyplot.loglog(chain_size[:len(time_mp2_conventional)], time_mp2_conventional, label="MP2 p={:.1f}".format(popt_mp2[1]), ls="None", marker='x', color="#5555FF")
pyplot.loglog([10, 150], [f(10, *popt_mp2), f(150, *popt_mp2)], color="#5555FF")
pyplot.loglog(chain_size[:len(time_lmp2)], time_lmp2, label="LMP2 p={:.1f}".format(popt_lmp2[1]), ls="None", marker='x', color="#55FF55")
pyplot.loglog([10, 150], [f(10, *popt_lmp2), f(150, *popt_lmp2)], color="#55FF55")
pyplot.xlim(10, 150)
pyplot.legend()
pyplot.xlabel("N (atoms)")
pyplot.ylabel("Time (s)")
pyplot.savefig("He.pdf")