#!/usr/bin/env python
from test_common import hydrogen_dimer_chain
from test_dchf import assign_chain_domains
import dchf
from pyscf.scf import RHF

model = hydrogen_dimer_chain(24)
model.verbose = 4

mf = RHF(model)
mf.kernel()

mf = dchf.DCHF(model)
assign_chain_domains(mf, 2, 2)
mf.kernel()

# print "Custom MP2", dchf.energy_2(mf.domains, 1.0, with_t2=False)[0]
# print "Pyscf MP2", dchf.energy_2(mf.domains, 1.0, amplitude_calculator=dchf.pyscf_mp2_amplitude_calculator, with_t2=False)[0]
# print "Pyscf CCSD", dchf.energy_2(mf.domains, 1.0, amplitude_calculator=dchf.pyscf_ccsd_amplitude_calculator, with_t2=False)[0]
