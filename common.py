import itertools

from pyscf import gto, scf
from pyscf.lib import logger
from pyscf.scf.hf import RHF
from pyscf.fci.direct_spin0 import FCISolver
from pyscf.ao2mo import restore

import numpy
from scipy import special, linalg


def transform(o, psi, axes="all", mode="fast"):
    """
    A generic transform routine using numpy.einsum.
    Args:
        o (numpy.ndarray): a vector/matrix/tensor to transform;
        psi (numpy.ndarray): a basis to transform to;
        axes (list, str): dimensions to transform along;
        mode (str): mode, either 'onecall', calls numpy.einsum once, or 'fast' transforming one axis at a time.

    Returns:
        A transformed array.
    """
    n = len(o.shape)
    if axes == "all":
        axes = range(n)
    elif axes == "f2":
        axes = (0, 1)
    elif axes == "l2":
        axes = (n-2, n-1)
    elif isinstance(axes, int):
        axes = (axes,)
    else:
        axes = tuple(axes)
    if mode == "fast":
        result = o
        for a in axes:
            result = transform(result, psi, axes=a, mode='onecall')
        return result
    elif mode == "onecall":
        letters = "abcdefghijklmnopqrstuvwxyz"
        o_subscripts = letters[:n]
        output_subscripts = str(o_subscripts)
        letters = letters[n:]
        p_subscripts = ""
        for i, ax in enumerate(axes):
            p_subscripts += ","+o_subscripts[ax]+letters[i]
            output_subscripts = output_subscripts[:ax]+letters[i]+output_subscripts[ax+1:]
        subscripts = o_subscripts+p_subscripts+"->"+output_subscripts
        return numpy.einsum(subscripts, o, *((psi,)*len(axes)))
    else:
        raise ValueError("Unknown mode: {}".format(mode))


class AbstractIntegralProvider(object):
    def __init__(self, mol):
        """
        A local integral provider.
        Args:
            mol (pyscf.gto.mole.Mole): the Mole object;
        """
        self.__mol__ = mol
        self.__ao_ownership__ = numpy.array(tuple(i[0] for i in self.__mol__.ao_labels(fmt=False)))

    def get_atom_basis(self, atoms, domain=None):
        """
        Retrieves basis function indices corresponding to the list of atoms.
        Args:
            atoms (list, tuple): a subset of atoms where the basis functions reside;
            domain (list, tuple): the parent domain, if any;

        Returns:
            A list of basis functions' indices.
        """
        ao = self.__ao_ownership__
        atoms = self.__dressed_atoms__(atoms)
        if domain is not None:
            mask = (ao[:, numpy.newaxis] == numpy.array(domain)[numpy.newaxis, :]).sum(axis=1)
            ao = ao[numpy.argwhere(mask)]
        if atoms is None:
            return numpy.arange(len(ao))
        else:
            mask = (ao[:, numpy.newaxis] == numpy.array(atoms)[numpy.newaxis, :]).sum(axis=1)
            return numpy.nonzero(mask)[0]

    def get_block(self, *atoms):
        """
        Retrieves a block slice corresponding to given atoms sets.
        Args:
            atoms (list, tuple): subsets of atoms where the basis functions of each dimension reside;
            dims (int): the number of dimensions;

        Returns:
            A slice for the diagonal block.
        """
        return numpy.ix_(*tuple(self.get_atom_basis(i) for i in atoms))

    def __dressed_atoms__(self, atoms):
        if atoms is None:
            return tuple(range(self.__mol__.natm))
        elif isinstance(atoms, int):
            return atoms,
        else:
            return tuple(atoms)

    def shell_ranges(self, atoms):
        """
        Retrieves shell ranges corresponding to the set of atoms.
        Args:
            atoms (list, tuple, set): a list of atoms shells belonging to;

        Returns:
            A list of tuples with ranges of shell slices.
        """
        atoms = self.__dressed_atoms__(atoms)
        result = numpy.zeros(self.__mol__._bas.shape[0] + 2, dtype=bool)
        for a in atoms:
            result[1:-1] = numpy.logical_or(result[1:-1], self.__mol__._bas[:, gto.ATOM_OF] == a)
        r1 = result[:-1]
        r2 = result[1:]
        fr = numpy.argwhere(numpy.logical_and(numpy.logical_not(r1), r2))[:, 0]
        to = numpy.argwhere(numpy.logical_and(r1, numpy.logical_not(r2)))[:, 0]
        return numpy.stack((fr, to), axis=1)

    def atomic_basis_size(self, atom):
        """
        Retrieves the full basis size of each atom.
        Args:
            atoms (int): atom ID;

        Returns:
            The total number of basis functions.
        """
        return sum(self.__mol__.bas_len_cart(i) for i in self.__mol__.atom_shell_ids(atom))

    def intor_atoms(self, name, *atoms, **kwargs):
        """
        A version of `pyscf.mole.Mole.intor` accepting lists of atoms instead of shells.
        Args:
            name (str): integral name;
            *atoms (nested list): atoms lists;
            **kwargs: keywords passed to `pyscf.mole.Mole.intor`;

        Returns:
            An array or tensor with integrals.
        """
        raise NotImplementedError

    def get_ovlp(self, atoms1, atoms2):
        """
        Retrieves an overlap matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (row index);

        Returns:
            A rectangular matrix with overlap integral values.
        """
        return self.intor_atoms('int1e_ovlp_sph', atoms1, atoms2)

    def get_kin(self, atoms1, atoms2):
        """
        Retrieves a kinetic energy matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (row index);

        Returns:
            A rectangular matrix with kinetic energy matrix values.
        """
        return self.intor_atoms('int1e_kin', atoms1, atoms2)

    def get_ext_pot(self, atoms1, atoms2):
        """
        Retrieves an external potential energy matrix.
        Args:
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (row index);

        Returns:
            A rectangular matrix with external potential matrix values.
        """
        return self.intor_atoms('int1e_nuc', atoms1, atoms2)

    def get_hcore(self, atoms1, atoms2):
        """
        Retrieves a core part of the Hamiltonian.
        Args:
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (row index);

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
        return self.intor_atoms("int2e_sph", atoms1, atoms2, atoms3, atoms4)


def intor(mol, name, *shells, **kwargs):
    """
    A version of `pyscf.mole.Mole.intor` accepting lists of shell ranges instead of single ranges.
    Args:
        mol (pyscf.mol.Mole): the Mole object;
        name (str): integral name;
        *shells (nested list): shell ranges lists;
        **kwargs: keywords passed to `pyscf.mole.Mole.intor`;

    Returns:
        An array with integrals.
    """
    shls_slice = tuple()
    basis_size = []
    for dim, shell_list in enumerate(shells):
        shell_list = numpy.array(shell_list)
        if len(shell_list.shape) == 1:
            shls_slice += tuple(shell_list)
        elif len(shell_list.shape) != 2:
            raise ValueError("Cannot recognize shell list: {}".format(repr(shell_list)))
        elif len(shell_list) == 1:
            shls_slice += tuple(shell_list[0])
        else:
            ints = []
            for sh in shell_list:
                ints.append(intor(mol, name, *list(shells[:dim]+(sh,)+shells[dim+1:]), **kwargs))
            return numpy.concatenate(ints, axis=dim)
        basis_size.append(sum(
            mol.bas_len_cart(i) for i in range(shls_slice[-2], shls_slice[-1])
        ))

    if "do_not_hack_pyscf" in kwargs:
        kwargs["shls_slice"] = shls_slice
        result = mol.intor(name, **kwargs).view()

    else:
        bas = mol._bas
        shells = sorted(set(sum((list(range(i, j)) for i, j in zip(shls_slice[::2], shls_slice[1::2])), [])))
        mol._bas = mol._bas[shells, :]
        shls_slice = tuple(numpy.searchsorted(shells, i) for i in shls_slice)

        kwargs["shls_slice"] = shls_slice
        result = mol.intor(name, **kwargs).view()

        mol._bas = bas
    result.shape = tuple(basis_size)
    return result


class IntegralProvider(AbstractIntegralProvider):
    def intor_atoms(self, name, *atoms, **kwargs):
        return intor(self.__mol__, name, *(self.shell_ranges(i) for i in atoms), **kwargs)

    intor_atoms.__doc__ = AbstractIntegralProvider.intor_atoms.__doc__


def as_dict(tensor, provider):
    """
    Converts a tensor into a dict form where keys are atomic indexes.
    Args:
        tensor (numpy.ndarray): a tensor to convert;
        provider (IntegralProvider): provider of integrals;

    Returns:
        A dict with tensor blocks.
    """
    result = {}
    r = range(len(provider.__mol__.natm))
    for a in itertools.product((r,)*len(tensor.shape)):
        result[a] = tensor[provider.get_block(*a)]
    return result


def pairs(n, s1=0, s2=0, symmetry=True):
    """
    Iterates over atom pairs.

    Args:
        n (int): total number of atoms;
        s1 (int): starting point - atom 1 index;
        s2 (int): starting point - atom 2 index;
        symmetry (bool): whether symmetries are taken into acocunt;

    Yields:
        Pairs of atomic indexes.
    """
    for i in range(s1*n+s2, n**2):
        a1 = i // n
        a2 = i % n
        if not symmetry or (a1 <= a2):
            yield a1, a2


def get_sparse_eri(provider, threshold=1e-12):
    """
    Retrieves a sparse representation of electron repulsion integrals.
    Args:
        provider (IntegralProvider): integral provider;
        threshold (float): a threshold for the Cauchy-Schwartz estimation speciifying integral blocks to discard;

    Returns:
        A dict where keys are four atomic indexes and values are four-center integral blocks.
    """

    diagonal = {}

    for a1, a2 in pairs(provider.__mol__.natm):
        integrals = provider.get_eri(a1, a2, a1, a2)
        diagonal[a1, a2] = integrals

    eri = {}
    for a1, a2 in pairs(provider.__mol__.natm):
        for a3, a4 in pairs(provider.__mol__.natm, a1, a2):
            if (diagonal[a1, a2] * diagonal[a3, a4]).max() > threshold ** 2:
                if a1 == a3 and a2 == a4:
                    integrals = diagonal[a1, a2]

                    eri[a1, a2, a3, a4] = integrals
                    eri[a2, a1, a3, a4] = integrals.transpose((1, 0, 2, 3))
                    eri[a1, a2, a4, a3] = integrals.transpose((0, 1, 3, 2))
                    eri[a2, a1, a4, a3] = integrals.transpose((1, 0, 3, 2))

                else:
                    integrals = provider.get_eri(a1, a2, a3, a4)

                    eri[a1, a2, a3, a4] = integrals
                    eri[a2, a1, a3, a4] = integrals.transpose((1, 0, 2, 3))
                    eri[a1, a2, a4, a3] = integrals.transpose((0, 1, 3, 2))
                    eri[a2, a1, a4, a3] = integrals.transpose((1, 0, 3, 2))

                    eri[a3, a4, a1, a2] = integrals.transpose((2, 3, 0, 1))
                    eri[a4, a3, a1, a2] = integrals.transpose((3, 2, 0, 1))
                    eri[a3, a4, a2, a1] = integrals.transpose((2, 3, 1, 0))
                    eri[a4, a3, a2, a1] = integrals.transpose((3, 2, 1, 0))

    return eri


class SimpleCachingIntegralProvider(IntegralProvider):
    def __init__(self, mol):
        IntegralProvider.__init__(self, mol)
        self.cache = {}
        self.__stat_1__ = 0
        self.__stat_2__ = 0

    __init__.__doc__ = IntegralProvider.__init__.__doc__

    def intor_atoms(self, name, *atoms, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("Cannot cache integral values with keyword arguments")
        atoms = tuple(self.__dressed_atoms__(i) for i in atoms)
        if name not in self.cache:
            self.cache[name] = {}
        isolated = tuple()
        for ax, a in enumerate(atoms):
            if len(a) == 1:
                isolated += a
            if len(a) != 1:
                return numpy.concatenate(
                    tuple(self.intor_atoms(name, *(atoms[:ax]+(i,)+atoms[ax+1:])) for i in a),
                    axis=ax,
                )
        cache = self.cache[name]
        self.__stat_1__ += 1
        if isolated not in cache:
            self.__stat_2__ += 1
            cache[isolated] = IntegralProvider.intor_atoms(self, name, *isolated)
        return cache[isolated]

    intor_atoms.__doc__ = IntegralProvider.intor_atoms.__doc__

    def cache_factor(self):
        """
        Calculates number of cache accesses relative to the number of integral evaluations.
        Returns:
            Cache factor.
        """
        return 1.0*self.__stat_1__ / self.__stat_2__


def fermi_distribution(chemical_potential, temperature, energies):
    """
    Fermi distribution function.
    Args:
        chemical_potential (float): the chemical potential;
        temperature (float): temperature in energy units;
        energies (numpy.ndarray): vavlues of states' energies;

    Returns:
        An array with occupation numbers.
    """
    return 2./(numpy.exp((energies - chemical_potential)/temperature) + 1)


def gaussian_distribution(chemical_potential, temperature, energies):
    """
    Gaussian distribution function.
    Args:
        chemical_potential (float): the chemical potential;
        temperature (float): temperature in energy units;
        energies (numpy.ndarray): vavlues of states' energies;

    Returns:
        An array with occupation numbers.
    """
    return 1 - special.erf((energies-chemical_potential)/temperature)


class NonSelfConsistentMeanField(object):
    def __init__(self, mf):
        """
        A non-self-consistent mean-field model.
        Args:
            mf: a reference mean-field solution;
        """
        self.mol = mf.mol.copy()
        self._eri = mf._eri

        self.__veff__ = mf.get_veff()
        self.__hcore__ = mf.get_hcore()
        self.__ovlp__ = mf.get_ovlp()

        self.e_tot = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = mf.mo_occ.copy()

    def get_ovlp(self):
        return self.__ovlp__

    def get_hcore(self):
        return self.__hcore__

    def get_veff(self):
        return self.__veff__

    def get_fock(self):
        return self.get_hcore() + self.get_veff()

    def make_rdm1(self):
        return scf.hf.make_rdm1(self.mo_coeff, self.mo_occ)

    def energy_elec(self):
        h1e = self.get_hcore()
        dm = self.make_rdm1()
        vhf = self.get_veff()
        e1 = numpy.einsum('ji,ji', h1e.conj(), dm).real
        e_coul = numpy.einsum('ji,ji', vhf.conj(), dm).real * .5
        return e1 + e_coul, e_coul

    def energy_nuc(self):
        return self.mol.energy_nuc()

    def energy_tot(self):
        return self.energy_elec()[0] + self.energy_nuc()

    def kernel(self):
        """
        Performs a non-self-consistent calculation.
        Returns:
            The total energy.
        """
        self.mo_energy, self.mo_coeff = linalg.eigh(
            self.get_fock(),
            self.get_ovlp(),
        )
        self.e_tot = self.energy_tot()
        logger.note(self.mol, 'non-SCF energy = %.15g', self.e_tot)
        return self.e_tot


def partial_etot(hcore, eri, rho1, rho2, s):
    """
    Calculates a partial total energy value using democratic partitioning.
    Args:
        hcore (numpy.ndarray): the one-particle Hamiltonian part;
        eri (numpy.ndarray): the two-particle Hamiltonian part;
        rho1 (numpy.ndarray): the one-particle density matrix;
        rho2 (numpy.ndarray): the two-particle density matrix;
        s (slice): a slice or an array defining system fragment;

    Returns:
        The partial total energy value.
    """
    eri = restore(1, eri, hcore.shape[0])
    term1 = hcore*rho1
    term2 = (eri - 0.5*eri.transpose(0, 3, 2, 1))*rho2

    def democratic_partitioning(x, significant_slice):
        empty_slice = slice(None)
        return 1.0/len(x.shape) * sum(x[(empty_slice,)*i + (significant_slice,)].sum() for i in range(len(x.shape)))

    return democratic_partitioning(term1, s) + democratic_partitioning(term2, s)


def partial_nelec(rho1, s):
    """
    Calculates the partial electronic occupation of the system.
    Args:
        rho1 (numpy.ndarray): the one-particle density matrix;
        s (slice): a slice or an array defining system fragment;

    Returns:
        The partial occupation number.
    """
    return numpy.diag(rho1)[s].sum()


class ModelRHF(RHF):
    def __init__(self, hcore, eri, e_vac=0, ovlp=None, **kwargs):
        """
        RHF solver of model Hamiltonians with DMET interface.
        Args:
            hcore (numpy.ndarray): the core part of the Hamiltonian;
            eri (numpy.ndarray): ERI 4-tensor;
            e_vac (float): the energy of the vacuum;
            ovlp (numpy.ndarray): the overlap matrix;
            **kwargs: keyword arguments to a dummy Mole object;
        """
        mol = gto.Mole()
        for k, v in kwargs.items():
            setattr(mol, k, v)
        mol.build()
        RHF.__init__(self, mol)
        self.__hcore__ = hcore
        if ovlp is None:
            self.__ovlp__ = numpy.eye(hcore.shape[0])
        else:
            self.__ovlp__ = ovlp
        self.__e_vac__ = e_vac
        self._eri = restore(8, eri, eri.shape[0])

    def get_hcore(self, *args, **kwargs):
        return self.__hcore__

    def get_ovlp(self, *args, **kwargs):
        return self.__ovlp__

    def energy_nuc(self):
        return self.__e_vac__

    def partial_etot(self, s):
        rho1 = self.make_rdm1()
        rho2 = numpy.einsum("ij,kl->ijkl", rho1, rho1) / 2
        return partial_etot(self.__hcore__, self._eri, rho1, rho2, s)

    def partial_nelec(self, s):
        return partial_nelec(self.make_rdm1(), s)


class ModelFCI(FCISolver):
    def __init__(self, hcore, eri, e_vac=0, ovlp=None, **kwargs):
        """
        FCI solver of model Hamiltonians with DMET interface.
        Args:
            hcore (numpy.ndarray): the core part of the Hamiltonian;
            eri (numpy.ndarray): ERI 4-tensor;
            e_vac (float): the energy of the vacuum;
            ovlp (numpy.ndarray): the overlap matrix;
            **kwargs: keyword arguments to a dummy Mole object;
        """
        mol = gto.Mole()
        for k, v in kwargs.items():
            setattr(mol, k, v)
        mol.build()
        FCISolver.__init__(self, mol)
        self.__hcore__ = hcore
        if ovlp is None:
            self.__ovlp__ = numpy.eye(hcore.shape[0])
        else:
            self.__ovlp__ = ovlp
        self.__e_vac__ = e_vac
        self.__eri__ = restore(8, eri, eri.shape[0])

        self.e_tot = None
        self.psi = None
        self._keys = set(self.__dict__.keys())

    def kernel(self, **kwargs):
        e, psi = FCISolver.kernel(
            self,
            self.__hcore__,
            self.__eri__,
            self.__hcore__.shape[0],
            self.mol.nelectron,
            **kwargs
        )
        e += self.__e_vac__
        self.e_tot = e
        self.psi = psi
        return e, psi

    def make_rdm1(self, **kwargs):
        return FCISolver.make_rdm1(
            self,
            self.psi,
            self.__hcore__.shape[0],
            self.mol.nelectron,
            **kwargs
        )

    def make_rdm12(self, **kwargs):
        return FCISolver.make_rdm12(
            self,
            self.psi,
            self.__hcore__.shape[0],
            self.mol.nelectron,
            **kwargs
        )

    def partial_etot(self, s):
        rho1, rho2 = self.make_rdm12()
        return partial_etot(self.__hcore__, self.__eri__, rho1, rho2, s)

    def partial_nelec(self, s):
        return partial_nelec(self.make_rdm1(), s)
