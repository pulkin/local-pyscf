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

        cls.h6lhf = dchf.DCHF(cls.h6chain)

    def test_fock(self):
        """
        Tests the Fock matrix.
        """
        a1 = [1, 2]
        a2 = [1, 3, 4]
        testing.assert_allclose(
            self.h6mf.get_fock(dm=self.dm)[self.h6lhf.get_block(a1, a2)],
            self.h6lhf.get_fock(self.dm, a1, a2),
        )

    def test_orb_energies(self):
        """
        Tests exact orbital energies.
        """
        e, _ = self.h6lhf.get_orbs(self.dm, None)
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
        self.h6lhf.domains_erase()
        domain_size = 2
        buffer_size = 2
        for i in range(0, self.h6chain.natm, domain_size):
            self.h6lhf.add_domain(numpy.arange(
                max(i - buffer_size, 0),
                min(i + domain_size + buffer_size, self.h6chain.natm),
            ), domain_core=numpy.arange(i, min(i + domain_size, self.h6chain.natm)))
        e = self.h6lhf.iter_hf()
        testing.assert_allclose(self.h6lhf.dm, self.dm, atol=1e-2)
        testing.assert_allclose(self.h6mf.e_tot-self.h6chain.energy_nuc(), e, rtol=1e-4)
