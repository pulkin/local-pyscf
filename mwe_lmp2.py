#!/usr/bin/env python
from test_common import atomic_chain
from pyscf import scf
import lmp2

model = atomic_chain(16, spacing=6, name="He")
mf = scf.RHF(model)
mf.kernel()
pt = lmp2.LMP2(mf)
print pt.kernel()[0]
