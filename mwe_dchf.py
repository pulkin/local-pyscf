#!/usr/bin/env python
from test_common import hydrogen_dimer_chain
from test_dchf import assign_domains
import dchf

model = hydrogen_dimer_chain(24)
mf = dchf.DCHF(model)
assign_domains(mf, 4, 2)
mf.kernel()
