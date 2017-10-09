from pyscf import gto, scf, mp, lib
import lmp2

import unittest


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
        t2_ref = self.h6mp2.t2
        h6lmp2 = lmp2.LMP2(self.h6mf)
        h6lmp2.kernel()


if __name__ == "__main__":
    print "An example LMP2 run"
    model = atomic_chain(4, alt_spacing=2.3)
    # model = atomic_chain(20, name='He', spacing=20)
    model.verbose = 4
    mf = scf.RHF(model)
    mf.kernel()
    print mp.MP2(mf).kernel()[0]

    lmp2 = lmp2.LMP2(mf)#, local_integral_provider=lmp2.DummyLMP2IntegralProvider)#, local_space_provider=lmp2.iter_local_dummy)
    lmp2.kernel(mixer=lib.diis.DIIS())
