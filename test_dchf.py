#!/usr/bin/env python
from pyscf import scf, mp, cc, gto
import dchf

import numpy

import unittest
from numpy import testing
from test_common import helium_chain, hydrogen_dimer_chain


def assign_chain_domains(dchf, domain_size, buffer_size, reset_dm=True):
    """
    Assigns domains given the size of the domain core region and buffer region.
    Args:
        dchf (dchf.DCHF): divide-conquer Hartree-Fock setup;
        domain_size (int): size of domains' cores;
        buffer_size (int): size of the domains' buffer regions
        reset_dm (bool): resets the density matrix;
    """
    if reset_dm:
        dchf.dm = None
    dchf.domains_erase()
    for i in range(0, dchf.__mol__.natm, domain_size):
        dchf.add_domain(numpy.arange(
            max(i - buffer_size, 0),
            min(i + domain_size + buffer_size, dchf.__mol__.natm),
        ), core=numpy.arange(i, min(i + domain_size, dchf.__mol__.natm)))


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Solve HF
        cls.h6chain = hydrogen_dimer_chain(6)
        cls.h6mf = scf.RHF(cls.h6chain)
        cls.h6mf.kernel()
        cls.total_energy = cls.h6mf.e_tot - cls.h6mf.energy_nuc()
        cls.dm = cls.h6mf.make_rdm1()

        # Solve MP2
        cls.h6mp2 = mp.MP2(cls.h6mf)
        cls.h6mp2.kernel()

        # Solve CCSD
        cls.h6ccsd = cc.CCSD(cls.h6mf)
        cls.h6ccsd.kernel()

        # Single-domain DCHF
        cls.h6dchf_1 = dchf.DCHF(cls.h6chain)
        assign_chain_domains(cls.h6dchf_1, 6, 0)
        cls.h6dchf_1.kernel(tolerance=1e-10)
        cls.h6dcmp2_1 = dchf.DCMP2(cls.h6dchf_1)
        cls.h6dcmp2_1.kernel()
        cls.h6dcccsd_1 = dchf.DCCCSD(cls.h6dchf_1)
        cls.h6dcccsd_1.kernel()

        # Three-domain DCHF
        cls.h6dchf_3 = dchf.DCHF(cls.h6chain)
        assign_chain_domains(cls.h6dchf_3, 2, 2)
        cls.h6dchf_3.kernel(fock_hook=None)
        cls.h6dcmp2_3 = dchf.DCMP2(cls.h6dchf_3)
        cls.h6dcmp2_3.kernel()
        cls.h6dcccsd_3 = dchf.DCCCSD(cls.h6dchf_3)
        cls.h6dcccsd_3.kernel()

    def test_fock(self):
        """
        Tests the Fock matrix.
        """
        a1 = [1, 2]
        a2 = [1, 3, 4]
        testing.assert_allclose(
            self.h6mf.get_fock(dm=self.dm)[self.h6dchf_1.get_block(a1, a2)],
            self.h6dchf_1.get_fock(self.dm, a1, a2),
            atol=1e-14,
        )

    def test_orb_energies(self):
        """
        Tests exact orbital energies.
        """
        e, _ = self.h6dchf_1.get_orbs(self.dm, None)
        testing.assert_allclose(e, self.h6mf.mo_energy, rtol=1e-4)

    def test_util(self):
        """
        Test utility functions.
        """
        t = dchf.DCHF(self.h6chain)
        t.add_domain([0, 1], core=[0, 1])
        with self.assertRaises(ValueError):
            t.domains_cover(r=True)

    def test_hf_single_domain(self):
        """
        A single-domain HF test.
        """
        testing.assert_allclose(self.h6dchf_1.dm, self.dm, atol=1e-6)
        testing.assert_allclose(self.h6dchf_1.hf_energy, self.total_energy, rtol=1e-8)

    def test_mp2_single_domain(self):
        """
        A single-domain MP2 test.
        """
        testing.assert_allclose(self.h6dcmp2_1.e2, self.h6mp2.e_corr, atol=1e-8)

    def test_ccsd_single_domain(self):
        """
        A single-domain CCSD test.
        """
        testing.assert_allclose(self.h6dcccsd_1.e2, self.h6ccsd.e_corr, atol=1e-8)

    def test_hf(self):
        """
        HF test.
        """
        testing.assert_allclose(self.h6dchf_3.dm, self.dm, atol=1e-2)
        testing.assert_allclose(self.h6dchf_3.hf_energy, self.total_energy, rtol=1e-4)

    def test_mp2(self):
        """
        MP2 test.
        """
        testing.assert_allclose(self.h6dcmp2_3.e2, self.h6mp2.e_corr, atol=1e-4)

    def test_ccsd(self):
        """
        CCSD test.
        """
        testing.assert_allclose(self.h6dcccsd_3.e2, self.h6ccsd.e_corr, atol=1e-3)


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
        assign_chain_domains(self.h12dchf, 4, 2)
        e = self.h12dchf.kernel(tolerance=1e-9, fock_hook=None)
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
        assign_chain_domains(cls.he10dchf, 1, 1)
        cls.he10dchf.kernel()
        cls.he10dcmp2 = dchf.DCMP2(cls.he10dchf)
        cls.he10dcmp2.kernel()

    def test_results(self):
        testing.assert_allclose(self.he10mf.e_tot, self.he10dchf.e_tot)
        testing.assert_allclose(self.he10mp2.e_corr, self.he10dcmp2.e2)
