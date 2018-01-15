#!/usr/bin/env python
import numpy
import dmet
import common

import unittest
from numpy import testing
from test_common import hubbard_model_driver


class HubbardModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 12
        cls.u = 8.0
        cls.nelec = 6
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

        testing.assert_allclose(e_ref, dummy_dmet.e_tot)
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

    def test_non_self_consistent(self):
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

