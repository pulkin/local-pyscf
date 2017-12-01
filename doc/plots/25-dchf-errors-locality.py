from pyscf import scf

import dchf
from test_common import atomic_chain
from test_dchf import assign_chain_domains

import numpy

import fake
pyplot = fake.pyplot()

N = 24
tolerance = 1e-5

spacing = numpy.concatenate((
    numpy.linspace(2.3, 1.5, 8, endpoint=False),
    numpy.linspace(1.5, 1.4, 4, endpoint=False),
))
hf_energy = []
ref_hf_energy = []
errors_dm = []
errors_dm_intrinsic = []

for alt in spacing:

    print "Alt =", alt

    model = atomic_chain(N, alt_spacing=alt)
    ref_hf = scf.RHF(model)
    ref_hf.conv_tol = tolerance
    ref_hf.kernel()
    dm_ref = ref_hf.make_rdm1()

    hf = dchf.DCHF(model)
    assign_chain_domains(hf, 4, 2)
    hf.kernel(tolerance=tolerance)

    mask = numpy.logical_not(hf.domains_pattern(2))

    hf_energy.append(hf.e_tot)
    ref_hf_energy.append(ref_hf.e_tot)
    errors_dm.append(abs(dm_ref - hf.dm).max())
    errors_dm_intrinsic.append(abs(dm_ref*mask).max())

pyplot.figure(figsize=(12, 4.8))
pyplot.subplot(121)
pyplot.scatter(spacing - 1.4, hf_energy, label="DCHF")
pyplot.scatter(spacing - 1.4, ref_hf_energy, label="HF")

pyplot.xlabel("Dimerization parameter (A)")
pyplot.ylabel("Total energy (Ry)")
pyplot.legend()

pyplot.subplot(122)
pyplot.plot(spacing-1.4, errors_dm, label="Absolute")
pyplot.plot(spacing-1.4, errors_dm_intrinsic, label="Intrinsic")
pyplot.xlabel("Dimerization parameter (A)")
pyplot.ylabel("Error in the density matrix")
pyplot.legend()
pyplot.show()
