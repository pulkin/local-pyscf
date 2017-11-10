import numpy
from scipy import special
from pyscf import gto


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
        self.mol = mol

    def __ao_labels_at__(self):
        return numpy.array(tuple(i[0] for i in self.mol.ao_labels(fmt=False)))

    def get_atom_basis(self, atoms, domain=None):
        """
        Retrieves basis function indices corresponding to the list of atoms.
        Args:
            atoms (list, tuple): a subset of atoms where the basis functions reside;
            domain (list, tuple): the parent domain, if any;

        Returns:
            A list of basis functions' indices.
        """
        ao = self.__ao_labels_at__()
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
            return tuple(range(self.mol.natm))
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
        result = numpy.zeros(self.mol._bas.shape[0]+2, dtype=bool)
        for a in atoms:
            result[1:-1] = numpy.logical_or(result[1:-1], self.mol._bas[:, gto.ATOM_OF] == a)
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
        return sum(self.mol.bas_len_cart(i) for i in self.mol.atom_shell_ids(atom))


class IntegralProvider(AbstractIntegralProvider):
    def intor(self, name, *shells, **kwargs):
        """
        A version of `pyscf.mole.Mole.intor` accepting lists of shell ranges instead of single ranges.
        Args:
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
                    ints.append(self.intor(name, *list(shells[:dim]+(sh,)+shells[dim+1:]), **kwargs))
                return numpy.concatenate(ints, axis=dim)
            basis_size.append(sum(
                self.mol.bas_len_cart(i) for i in range(shls_slice[-2], shls_slice[-1])
            ))
        kwargs["shls_slice"] = shls_slice
        result = self.mol.intor(name, **kwargs).view()
        result.shape = tuple(basis_size)
        return result

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
        return self.intor(name, *(self.shell_ranges(i) for i in atoms), **kwargs)

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

