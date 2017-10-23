from pyscf import gto
import common

import unittest
from numpy import testing


def atomic_chain(n, name='H', spacing=1.4, basis='cc-pvdz', alt_spacing=None):
    """
    Creates a Mole object with an atomic chain of a given size.
    Args:
        n (int): the size of an atomic chain;
        name (str): atom caption;
        spacing (float): spacing between atoms;
        basis (str): basis string;
        alt_spacing (float): alternating spacing, if any;

    Returns:
        A Mole object with an atomic chain.
    """
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
        return self.mol.intor_symmetric('int1e_ovlp')[self.get_block(atoms1, atoms2)]

    def get_kin(self, atoms1, atoms2):
        """
        Retrieves a kinetic energy matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the column basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the column basis functions reside (row index);

        Returns:
            A rectangular matrix with kinetic energy matrix values.
        """
        return self.mol.intor_symmetric('int1e_kin')[self.get_block(atoms1, atoms2)]

    def get_ext_pot(self, atoms1, atoms2):
        """
        Retrieves an external potential energy matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the column basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the column basis functions reside (row index);

        Returns:
            A rectangular matrix with external potential matrix values.
        """
        return self.mol.intor_symmetric('int1e_nuc')[self.get_block(atoms1, atoms2)]

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
        eri = self.mol.intor('int2e_sph').view()
        n = int(eri.shape[0]**.5)
        eri.shape = (n,)*4
        return eri[self.get_block(atoms1, atoms2, atoms3, atoms4)]


class HydrogenChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h6chain = atomic_chain(6, alt_spacing=2.3)
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
