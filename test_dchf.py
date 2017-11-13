#!/usr/bin/env python
from pyscf import scf, mp
import dchf

import numpy

import unittest
from numpy import testing
from test_common import helium_chain, hydrogen_dimer_chain


def assign_domains(dchf, domain_size, buffer_size):
    """
    Assigns domains given the size of the domain core region and buffer region.
    Args:
        dchf (dchf.DCHF): divide-conquer Hartree-Fock setup;
        domain_size (int): size of domains' cores;
        buffer_size (int): size of the domains' buffer regions
    """
    dchf.domains_erase()
    for i in range(0, dchf.mol.natm, domain_size):
        dchf.add_domain(numpy.arange(
            max(i - buffer_size, 0),
            min(i + domain_size + buffer_size, dchf.mol.natm),
        ), domain_core=numpy.arange(i, min(i + domain_size, dchf.mol.natm)))


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h6chain = hydrogen_dimer_chain(6)
        cls.h6mf = scf.RHF(cls.h6chain)
        cls.h6mf.kernel()

        cls.total_energy = cls.h6mf.e_tot - cls.h6mf.energy_nuc()
        cls.dm = cls.h6mf.make_rdm1()

        cls.h6dchf = dchf.DCHF(cls.h6chain)

    def test_fock(self):
        """
        Tests the Fock matrix.
        """
        a1 = [1, 2]
        a2 = [1, 3, 4]
        testing.assert_allclose(
            self.h6mf.get_fock(dm=self.dm)[self.h6dchf.get_block(a1, a2)],
            self.h6dchf.get_fock(self.dm, a1, a2),
            atol=1e-14,
        )

    def test_orb_energies(self):
        """
        Tests exact orbital energies.
        """
        e, _ = self.h6dchf.get_orbs(self.dm, None)
        testing.assert_allclose(e, self.h6mf.mo_energy, rtol=1e-4)

    def test_util(self):
        """
        Test utility functions.
        """
        t = dchf.DCHF(self.h6chain)
        t.add_domain([0, 1], domain_core=[0, 1])
        with self.assertRaises(ValueError):
            t.domains_cover(r=True)

    def test_iter(self):
        """
        Tests DCHF iterations.
        """
        assign_domains(self.h6dchf, 2, 2)
        e = self.h6dchf.kernel()
        testing.assert_allclose(self.h6dchf.dm, self.dm, atol=1e-2)
        testing.assert_allclose(self.h6mf.e_tot-self.h6chain.energy_nuc(), e, rtol=1e-4)

        mp2 = dchf.DCMP2(self.h6dchf)
        mp2.kernel()
        e_ref, _ = mp.MP2(self.h6mf).kernel()
        testing.assert_allclose(e_ref, mp2.e2, atol=1e-4)


class HydrogenChain12Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h12chain = hydrogen_dimer_chain(16)
        cls.h12mf = scf.RHF(cls.h12chain)
        cls.h12mf.kernel()

        cls.total_energy = cls.h12mf.e_tot - cls.h12mf.energy_nuc()
        cls.dm = cls.h12mf.make_rdm1()

        cls.h12dchf = dchf.DCHF(cls.h12chain)

    def test_iter(self):
        """
        Tests DCHF iterations.
        """
        assign_domains(self.h12dchf, 4, 2)
        e = self.h12dchf.kernel(tolerance=1e-9)
        testing.assert_allclose(self.h12dchf.dm, self.dm, atol=1e-2)
        testing.assert_allclose(self.h12mf.e_tot - self.h12chain.energy_nuc(), e, rtol=1e-4)

        mp2 = dchf.DCMP2(self.h12dchf)
        mp2.kernel()
        e_ref, _ = mp.MP2(self.h12mf).kernel()
        testing.assert_allclose(e_ref, mp2.e2, atol=1e-4)


class HeliumChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.he10chain = helium_chain(10)

        cls.he10mf = scf.RHF(cls.he10chain)
        cls.he10mf.kernel()
        cls.he10mp2 = mp.MP2(cls.he10mf)
        cls.he10mp2.kernel()

        cls.he10dchf = dchf.DCHF(cls.he10chain)
        assign_domains(cls.he10dchf, 1, 3)
        cls.he10dchf.kernel()
        cls.he10dcmp2 = dchf.DCMP2(cls.he10dchf)
        cls.he10dcmp2.kernel()

    def test_results(self):
        testing.assert_allclose(self.he10mp2.e_corr, self.he10dcmp2.e2)
