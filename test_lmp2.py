#!/usr/bin/env python
from pyscf import gto, scf, mp, lib
import lmp2

import unittest
from numpy import testing


def atomic_chain(n, name='H', spacing=1.4, basis='cc-pvdz', alt_spacing=None):
    if alt_spacing is None:
        alt_spacing = spacing
    a = 0.5*(spacing+alt_spacing)
    b = 0.5*(spacing-alt_spacing)
    return gto.M(
        atom=';'.join(list(
            '{} 0 0 {:.1f}'.format(name, a*i + (i%2)*b) for i in range(n)
        )),
        basis=basis,
        verbose=0,
    )


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h6chain = atomic_chain(6, alt_spacing=2.3)
        cls.h6mf = scf.RHF(cls.h6chain)
        cls.h6mf.kernel()
        cls.h6mp2 = mp.MP2(cls.h6mf)
        cls.h6mp2.kernel()

    def test_h6(self):
        e_ref = self.h6mp2.emp2
        h6lmp2 = lmp2.LMP2(self.h6mf)
        h6lmp2.kernel()
        e = h6lmp2.emp2
        testing.assert_allclose(e, e_ref, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()