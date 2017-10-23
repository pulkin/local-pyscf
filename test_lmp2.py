#!/usr/bin/env python
from pyscf import scf, mp
import numpy
import lmp2

import unittest
from numpy import testing
from test_common import atomic_chain


class DummyLMP2IntegralProvider(object):
    def __init__(self, mol):
        """
        A dummy provider for 4-center integrals in local MP2 which does not take advantage of matrix/tensor sparsity.
        Args:
            mol (pyscf.Mole): the Mole object;
        """
        integrals = mol.intor("int2e_sph")
        n = int(integrals.shape[0]**.5)
        self.oovv = integrals.reshape((n,) * 4).swapaxes(1, 2)

    def get_lmo_pao_block(self, atoms, orbitals, lmo1, lmo2, pao):
        """
        Retrieves a block of 4-center integrals in the localized molecular orbitals / projected atomic orbitals basis
        set.
        Args:
            atoms (list): a list of atoms within this space;
            orbitals (list): a list of orbitals within this space;
            lmo1 (numpy.ndarray): localized molecular orbital 1;
            lmo2 (numpy.ndarray): localized molecular orbital 2;
            pao (numpy.ndarray): projected atomic orbitals;

        Returns:
            A block of 4-center integrals.
        """
        result = lmp2.transform(
            lmp2.transform(
                lmp2.transform(self.oovv, lmo1[:, numpy.newaxis], axes=0),
                lmo2[:, numpy.newaxis],
                axes=1,
            ),
            pao,
            axes='l2',
        )
        return result[0, 0][numpy.ix_(orbitals, orbitals)]


def iter_local_dummy(mol, mo_occ):
    """
    Performs iterations over dummy pair subspaces without any reduction.
    Args:
        mol (pyscf.Mole): a Mole object;
        mo_occ (numpy.ndarray): occupied molecular orbitals;

    Returns:
        For each pair, returns molecular orbital indexes i, j, a set of atomic indexes corresponding to this pair and
        a numpy array with atomic orbital indexes corresponding to this pair.
    """

    for i in range(mo_occ.shape[1]):
        for j in range(mo_occ.shape[1]):
            yield i, j, numpy.arange(mol.natm), numpy.arange(mo_occ.shape[0])


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h6chain = atomic_chain(6, alt_spacing=2.3)
        cls.h6mf = scf.RHF(cls.h6chain)
        cls.h6mf.kernel()
        cls.h6mp2 = mp.MP2(cls.h6mf)
        cls.h6mp2.kernel()

    def test_h6(self):
        """
        Tests the method with default parameters of the local MP2.
        """
        e_ref = self.h6mp2.emp2

        h6lmp2 = lmp2.LMP2(self.h6mf)
        h6lmp2.kernel()
        e = h6lmp2.emp2
        testing.assert_allclose(e, e_ref, rtol=2e-2)

    def test_h6_dummy_integrals(self):
        """
        Tests the method with exact transformed integrals.
        """
        e_ref = self.h6mp2.emp2

        h6lmp2 = lmp2.LMP2(self.h6mf, local_integral_provider=DummyLMP2IntegralProvider)
        h6lmp2.kernel()
        e = h6lmp2.emp2
        testing.assert_allclose(e, e_ref, rtol=2e-2)

    def test_h6_no_sparsity(self):
        """
        Tests the implementation equivalence to the conventional MP2 when the full LMO/PAO space is employed.
        """
        e_ref = self.h6mp2.emp2

        h6lmp2 = lmp2.LMP2(self.h6mf, local_space_provider=iter_local_dummy)
        h6lmp2.kernel()
        e = h6lmp2.emp2
        testing.assert_allclose(e, e_ref, rtol=1e-6)

    def test_tolerance(self):
        """
        Tests convergence with a tight tolerance setting.
        """
        e_ref = self.h6mp2.emp2

        h6lmp2 = lmp2.LMP2(self.h6mf)
        h6lmp2.kernel(tolerance=1e-8, maxiter=30)
        e = h6lmp2.emp2
        testing.assert_allclose(e, e_ref, rtol=2e-2)


class HeliumChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.he6chain = atomic_chain(6, name="He", spacing=6)
        cls.he6mf = scf.RHF(cls.he6chain)
        cls.he6mf.kernel()
        cls.he6mp2 = mp.MP2(cls.he6mf)
        cls.he6mp2.kernel()

    def test_he6(self):
        """
        Tests the method with default parameters of the local MP2.
        """
        e_ref = self.he6mp2.emp2

        he6lmp2 = lmp2.LMP2(self.he6mf)
        he6lmp2.kernel()
        e = he6lmp2.emp2
        testing.assert_allclose(e, e_ref, rtol=2e-2)
        testing.assert_equal(he6lmp2.iterations, 2)


if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(HydrogenChainTest("test_h6_no_sparsity"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
