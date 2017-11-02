#!/usr/bin/env python
from pyscf import scf
import dchf

import numpy

import unittest
from numpy import testing
from test_common import atomic_chain


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h6chain = atomic_chain(6, alt_spacing=2.3)
        cls.h6mf = scf.RHF(cls.h6chain)
        cls.h6mf.kernel()

        cls.total_energy = cls.h6mf.e_tot - cls.h6mf.energy_nuc()
        cls.dm = cls.h6mf.make_rdm1()

        cls.h6dchf = dchf.DCHF(cls.h6chain)

    def assign_domains(self, domain_size, buffer_size):
        """
        Assigns domains given the size of the domain core region and buffer region.
        Args:
            domain_size (int): size of domains' cores;
            buffer_size (int): size of the domains' buffer regions
        """
        self.h6dchf.domains_erase()
        for i in range(0, self.h6chain.natm, domain_size):
            self.h6dchf.add_domain(numpy.arange(
                max(i - buffer_size, 0),
                min(i + domain_size + buffer_size, self.h6chain.natm),
            ), domain_core=numpy.arange(i, min(i + domain_size, self.h6chain.natm)))

    def test_fock(self):
        """
        Tests the Fock matrix.
        """
        a1 = [1, 2]
        a2 = [1, 3, 4]
        testing.assert_allclose(
            self.h6mf.get_fock(dm=self.dm)[self.h6dchf.get_block(a1, a2)],
            self.h6dchf.get_fock(self.dm, a1, a2),
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
        self.assign_domains(2, 2)
        e = self.h6dchf.iter_hf()
        testing.assert_allclose(self.h6dchf.dm, self.dm, atol=1e-2)
        testing.assert_allclose(self.h6mf.e_tot-self.h6chain.energy_nuc(), e, rtol=1e-4)
