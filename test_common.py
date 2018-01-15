from pyscf import gto, scf
import common

import unittest
import numpy
from numpy import testing
import random


def assert_eye(a, **kwargs):
    """
    Tests whether the matrix is equal to the unity matrix.
    Args:
        a (numpy.ndarray): a 2D matrix;
        **kwargs: keyword arguments to `numpy.testing.assert_allclose`;
    """
    testing.assert_equal(a.shape[0], a.shape[1])
    testing.assert_allclose(a, numpy.eye(a.shape[0]), **kwargs)


def assert_basis_orthonormal(a, **kwargs):
    """
    Tests orthonormality of the basis set.
    Args:
        a (numpy.ndarray): a 2D matrix with basis coefficients;
        **kwargs: keyword arguments to `numpy.testing.assert_allclose`;
    """
    assert_eye(a.conj().T.dot(a), **kwargs)


def atomic_chain(n, name='H', spacing=1.4, alt_spacing=None, rndm=0.0, **kwargs):
    """
    Creates a Mole object with an atomic chain of a given size.
    Args:
        n (int): the size of an atomic chain;
        name (str): atom caption;
        spacing (float): spacing between atoms;
        alt_spacing (float): alternating spacing, if any;
        rndm (float): random displacement of atoms;

    Returns:
        A Mole object with an atomic chain.
    """
    default = dict(
        basis='cc-pvdz',
        verbose=0,
    )
    default.update(kwargs)
    if alt_spacing is None:
        alt_spacing = spacing
    a = 0.5*(spacing+alt_spacing)
    b = 0.5*(spacing-alt_spacing)
    random.seed(0)
    return gto.M(
        atom=';'.join(list(
            '{} 0 0 {:.1f}'.format(name, a*i + (i % 2)*b + random.random()*rndm - rndm/2) for i in range(n)
        )),
        **default
    )


def helium_chain(n, **kwargs):
    return atomic_chain(n, name="He", spacing=6, **kwargs)


def hydrogen_dimer_chain(n, **kwargs):
    return atomic_chain(n, alt_spacing=2.3, **kwargs)


def hydrogen_distant_dimer_chain(n, **kwargs):
    return atomic_chain(n, alt_spacing=6, **kwargs)


def hubbard_model_driver(u, n, nelec, pbc=True, t=-1, driver=common.ModelRHF):
    """
    Sets up the Hubbard model.
    Args:
        u (float): the on-site interaction value;
        n (int): the number of sites;
        nelec (int): the number of electrons;
        pbc (bool): closes the chain if True;
        t (float): the hopping term value;
        driver: a supported driver;

    Returns:
        The Hubbard model.
    """
    hcore = t * (numpy.eye(n, k=1) + numpy.eye(n, k=-1))
    if pbc:
        hcore[0, n-1] = hcore[n-1, 0] = t
    eri = numpy.zeros((n, n, n, n), dtype=numpy.float)
    for i in range(n):
        eri[i, i, i, i] = u
    result = driver(
        hcore,
        eri,
        nelectron=nelec,
        verbose=4,
    )
    return result


class DummyIntegralProvider(common.AbstractIntegralProvider):
    def get_ovlp(self, atoms1, atoms2):
        """
        Retrieves an overlap matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the column basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the column basis functions reside (row index);

        Returns:
            A rectangular matrix with overlap integral values.
        """
        return self.__mol__.intor_symmetric('int1e_ovlp')[self.get_block(atoms1, atoms2)]

    def get_kin(self, atoms1, atoms2):
        """
        Retrieves a kinetic energy matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the column basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the column basis functions reside (row index);

        Returns:
            A rectangular matrix with kinetic energy matrix values.
        """
        return self.__mol__.intor_symmetric('int1e_kin')[self.get_block(atoms1, atoms2)]

    def get_ext_pot(self, atoms1, atoms2):
        """
        Retrieves an external potential energy matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the column basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the column basis functions reside (row index);

        Returns:
            A rectangular matrix with external potential matrix values.
        """
        return self.__mol__.intor_symmetric('int1e_nuc')[self.get_block(atoms1, atoms2)]

    def get_hcore(self, atoms1, atoms2):
        """
        Retrieves a core part of the Hamiltonian.
        Args:
            atoms1 (list, tuple): a subset of atoms where the column basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the column basis functions reside (row index);

        Returns:
            A rectangular matrix with the core Hamiltonian.
        """
        return self.get_kin(atoms1, atoms2) + self.get_ext_pot(atoms1, atoms2)

    def get_eri(self, atoms1, atoms2, atoms3, atoms4):
        """
        Retrieves a subset of electron repulsion integrals corresponding to a given subset of atomic basis functions.
        Args:
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (first index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (second index);
            atoms3 (list, tuple): a subset of atoms where the basis functions reside (third index);
            atoms4 (list, tuple): a subset of atoms where the basis functions reside (fourth index);

        Returns:
            A four-index tensor with ERIs belonging to a given subset of atoms.
        """
        eri = self.__mol__.intor('int2e_sph').view()
        n = int(eri.shape[0]**.5)
        eri.shape = (n,)*4
        return eri[self.get_block(atoms1, atoms2, atoms3, atoms4)]


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h6chain = hydrogen_dimer_chain(6)
        cls.h6ip = common.IntegralProvider(cls.h6chain)
        cls.h6dip = DummyIntegralProvider(cls.h6chain)

    def test_ovlp(self):
        """
        Tests the overlap matrix.
        """
        testing.assert_allclose(self.h6ip.get_ovlp([2], [3, 4]), self.h6dip.get_ovlp([2], [3, 4]))

    def test_hcore(self):
        """
        Tests the core Hamiltonian matrix.
        """
        testing.assert_allclose(self.h6ip.get_kin([2], [3, 4]), self.h6dip.get_kin([2], [3, 4]))
        testing.assert_allclose(self.h6ip.get_ext_pot([0, 1, 2, 5], [3, 4]), self.h6dip.get_ext_pot([0, 1, 2, 5], [3, 4]))
        testing.assert_allclose(self.h6ip.get_hcore(None, [3, 4]), self.h6dip.get_hcore(None, [3, 4]))

    def test_eri(self):
        """
        Tests electron repulsion integrals.
        """
        testing.assert_allclose(self.h6ip.get_eri([0], [1, 2], [1, 3], [4, 5]), self.h6dip.get_eri([0], [1, 2], [1, 3], [4, 5]))


class ThresholdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h6chain = hydrogen_dimer_chain(6)
        cls.t = 1e-5
        cls.h6ip = common.IntegralProvider(cls.h6chain)
        cls.h6dip = DummyIntegralProvider(cls.h6chain)
        cls.sparse = common.get_sparse_eri(cls.h6ip, threshold=cls.t)

    def test_eri(self):
        """
        Tests electron repulsion integrals.
        """
        t1 = False
        t2 = False
        for q in (
                (0, 0, 0, 0),
                (0, 1, 0, 1),
                (0, 0, 0, 4),
                (3, 3, 3, 3),
                (0, 1, 2, 3),
                (0, 1, 3, 2),
                (1, 0, 2, 3),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 0, 1),
                (3, 2, 1, 0),
        ):
            if q in self.sparse:
                testing.assert_allclose(self.sparse[q], self.h6dip.get_eri(*q), atol=self.t)
                t1 = True
            else:
                testing.assert_allclose(self.h6dip.get_eri(*q), 0, atol=self.t)
                t2 = True

        assert t1
        assert t2


class UtilityTest(unittest.TestCase):
    def test_frozen(self):
        h6chain = hydrogen_dimer_chain(6)
        mf = scf.RHF(h6chain)
        mf.conv_tol = 1e-10
        mf.kernel()
        en, dm, eps = mf.e_tot, mf.make_rdm1(), mf.mo_energy

        common.NonSelfConsistentMeanField(mf)
        mf.kernel()
        testing.assert_allclose(en, mf.e_tot)
        testing.assert_allclose(dm, mf.make_rdm1(), atol=1e-8)
        testing.assert_allclose(eps, mf.mo_energy, atol=1e-8)
