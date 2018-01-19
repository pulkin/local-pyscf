#!/usr/bin/env python
from pyscf.scf import RHF
import numpy
import dmet
import common

import unittest
from numpy import testing
from test_common import hubbard_model_driver, hydrogen_dimer_chain


class HubbardModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 12
        cls.u = 8.0
        cls.nelec = 10
        cls.mf = hubbard_model_driver(cls.u, cls.n, cls.nelec)

    def __test_dummy__(self, sc, fragment_size, **kwargs):
        """
        A generic routine for self-consistency tests where the local fragment is treated with RHF.
        Args:
            sc: the self-consistency class;
            fragment_size: the size of the fragment;
            **kwargs: keyword arguments to DMET constructor;
        """
        self.mf.kernel()
        e_ref = self.mf.e_tot
        dummy_dmet = dmet.DMET(
            self.mf,
            common.ModelRHF,
            sc,
            numpy.arange(self.n).reshape(-1, fragment_size),
            **kwargs
        )
        dummy_dmet.kernel()

        testing.assert_allclose(e_ref, dummy_dmet.e_tot, rtol=1e-6)
        if dummy_dmet.conv_tol is not None:
            testing.assert_allclose([0], dummy_dmet.convergence_history)

    def test_dummy_interacting(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF. The fragment size is 1.
        """
        self.__test_dummy__(
            dmet.FragmentFullDMETUSC,
            1,
            associate='all',
            style="interacting-bath",
        )

    def test_dummy_non_interacting(self):
        """
        Test a dummy non-interacting-bath setup where the fragment solver is RHF. The fragment size is 1.
        """
        self.__test_dummy__(
            dmet.FragmentFullDMETUSC,
            1,
            associate='all',
            style="non-interacting-bath",
        )

    def test_dummy_2(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF. The fragment size is 2.
        """
        self.__test_dummy__(
            dmet.FragmentFullDMETUSC,
            2,
            associate='all',
            style="interacting-bath",
        )

    def test_dummy_local(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF and only the fragment part of the density
        matrix is fitted. The fragment size is 1.
        """
        self.__test_dummy__(
            dmet.FragmentFragmentDMETUSC,
            1,
            associate='all',
            style="interacting-bath",
        )

    def test_dummy_mu(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF and only the potential is fitted. The
        fragment size is 1.
        """
        self.__test_dummy__(
            dmet.MuFragmentDMETUSC,
            1,
            associate='all',
            style="interacting-bath",
        )

    def test_dummy_non_self_consistent(self):
        """
        Test a dummy interacting-bath non-self-consistent setup where the fragment solver is RHF. The fragment size is
        1.
        """
        self.__test_dummy__(
            dmet.FragmentFragmentDMETUSC,
            1,
            associate='all',
            style="interacting-bath",
            conv_tol=None,
        )

    def test_fci_mu(self):
        """
        Test whether chemical potential-based self-consistency converges immediately with FCI solver.
        """
        self.mf.kernel()
        e_ref = self.mf.e_tot

        fci_dmet = dmet.DMET(
            self.mf,
            common.ModelFCI,
            dmet.MuFragmentDMETUSC,
            numpy.arange(self.n).reshape(-1, 2),
            associate="all",
            style="interacting-bath",
        )
        fci_dmet.kernel()

        # A single iteration is required to fit the non-zero chemical potential of the mean-field solver.
        # The second iteration should converge to zero
        testing.assert_equal(2, len(fci_dmet.convergence_history))
        testing.assert_array_less(0, e_ref)
        testing.assert_array_less(fci_dmet.e_tot, 0)

    def test_fci_1(self):
        """
        Performs an FCI test with a fragment size equal to 1.
        """
        self.mf.kernel()
        e_ref = self.mf.e_tot
        testing.assert_array_less(0, e_ref)

        fci_dmet = dmet.DMET(
            self.mf,
            common.ModelFCI,
            dmet.FragmentFullDMETUSC,
            numpy.arange(self.n).reshape(-1, 2),
            associate="all",
            style="interacting-bath",
        )
        fci_dmet.kernel()
        testing.assert_array_less(fci_dmet.e_tot, 0)
        testing.assert_array_less(2, len(fci_dmet.convergence_history))


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mf = RHF(hydrogen_dimer_chain(10))

    def __test_dummy__(self, sc, fragment_size, **kwargs):
        """
        A generic routine for self-consistency tests where the local fragment is treated with RHF.
        Args:
            sc: the self-consistency class;
            fragment_size: the size of the fragment;
            **kwargs: keyword arguments to DMET constructor;
        """
        self.mf.kernel()
        e_ref = self.mf.e_tot
        dummy_dmet = dmet.AbInitioDMET(
            self.mf,
            common.ModelRHF,
            sc,
            numpy.arange(self.mf.mol.natm).reshape(-1, fragment_size),
            **kwargs
        )
        dummy_dmet.kernel(maxiter=2)

        testing.assert_allclose(e_ref, dummy_dmet.e_tot, rtol=1e-6)
        if dummy_dmet.conv_tol is not None:
            testing.assert_allclose([0], dummy_dmet.convergence_history)

    def test_dummy_interacting(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF. The fragment size is 1.
        """
        self.__test_dummy__(
            dmet.FragmentFullDMETUSC,
            1,
            style="interacting-bath",
        )

    def test_dummy_2(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF. The fragment size is 2.
        """
        self.__test_dummy__(
            dmet.FragmentFullDMETUSC,
            2,
            style="interacting-bath",
        )

    def test_dummy_local(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF and only the fragment part of the density
        matrix is fitted. The fragment size is 1.
        """
        self.__test_dummy__(
            dmet.FragmentFragmentDMETUSC,
            1,
            style="interacting-bath",
        )

    def test_dummy_mu(self):
        """
        Test a dummy interacting-bath setup where the fragment solver is RHF and only the potential is fitted. The
        fragment size is 1.
        """
        self.__test_dummy__(
            dmet.MuFragmentDMETUSC,
            1,
            style="interacting-bath",
        )

    def test_dummy_non_self_consistent(self):
        """
        Test a dummy interacting-bath non-self-consistent setup where the fragment solver is RHF. The fragment size is
        1.
        """
        self.__test_dummy__(
            dmet.FragmentFragmentDMETUSC,
            1,
            style="interacting-bath",
            conv_tol=None,
        )
