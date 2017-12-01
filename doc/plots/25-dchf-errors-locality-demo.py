from pyscf import scf

from dchf import DCHF

from test_common import atomic_chain
from test_dchf import assign_chain_domains

import fake
pyplot = fake.pyplot()

N = 24
tolerance = 1e-5

model = atomic_chain(N, alt_spacing=2.3)
hf_dimer = scf.RHF(model)
hf_dimer.conv_tol = tolerance
hf_dimer.kernel()

dchf_dimer = DCHF(model)
assign_chain_domains(dchf_dimer, 2, 2)
dchf_dimer.kernel(tolerance=tolerance)

model = atomic_chain(N, alt_spacing=1.53)
hf_uniform = scf.RHF(model)
hf_uniform.conv_tol = tolerance
hf_uniform.kernel()

dchf_uniform = DCHF(model)
assign_chain_domains(dchf_uniform, 2, 2)
dchf_uniform.kernel(tolerance=tolerance)

pyplot.figure(figsize=(12, 10))
for hf, subplot, title in (
        (hf_dimer, 221, "HF dimerized chain"),
        (hf_uniform, 222, "HF uniform chain"),
        (dchf_dimer, 223, "DC-HF dimerized chain"),
        (dchf_uniform, 224, "DC-HF uniform chain"),
):
    pyplot.subplot(subplot)
    # for wf, occ in zip(hf.mo_coeff.T, hf.mo_occ):
    #     if occ >= 1:
    #         wf = list(
    #             ((wf[ip.get_block(i)])**2).sum() for i in range(model.natm)
    #         )
    #         pyplot.plot(numpy.arange(len(wf)), wf)
    pyplot.imshow(abs(hf.dm if isinstance(hf, DCHF) else hf.make_rdm1()), vmin=0, vmax=0.25)
    pyplot.title(title)
    pyplot.colorbar()

    # pyplot.xlabel("Atom in the chain")
    # pyplot.ylabel("Occupied MO weight")

pyplot.show()