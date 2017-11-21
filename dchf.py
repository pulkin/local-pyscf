from pyscf import scf, mp, cc
from pyscf.lib import logger, diis

import numpy
import scipy
from scipy import cluster

import common


class HFLocalIntegralProvider(common.IntegralProvider):
    def get_j(self, dm, atoms1, atoms2, atoms3=None, atoms4=None):
        """
        Retrieves the J term in HF formalism (Coulomb repulsion).
        Args:
            dm (numpy.ndarray, dict): a dense or sparse density matrix;
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (row index);
            atoms3 (list, tuple): a subset of atoms where the basis functions reside (first internal summation index);
            atoms4 (list, tuple): a subset of atoms where the basis functions reside (second internal summation index);

        Returns:
            A matrix with Coulomb repulsion terms belonging to a given subset of atoms.
        """
        return numpy.einsum("ijkl,kl->ij", self.get_eri(atoms1, atoms2, atoms3, atoms4), dm)

    def get_k(self, dm, atoms1, atoms2, atoms3=None, atoms4=None):
        """
        Retrieves the K term in HF formalism (exchange interaction).
        Args:
            dm (numpy.ndarray, dict): a dense or sparse density matrix;
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (row index);
            atoms3 (list, tuple): a subset of atoms where the basis functions reside (first internal summation index);
            atoms4 (list, tuple): a subset of atoms where the basis functions reside (second internal summation index);

        Returns:
            A matrix with Coulomb repulsion terms belonging to a given subset of atoms.
        """
        return numpy.einsum("ijkl,jk->il", self.get_eri(atoms1, atoms3, atoms4, atoms2), dm)

    def get_v_eff(self, *args, **kwargs):
        """
        Retrieves the effective potential matrix terms in the HF formalism.
        Args:
            *args, **kwargs: see the description of `self.get_j` and `self.get_k`;

        Returns:
            An effective potential matrix terms belonging to a given subset of atoms.
        """
        return self.get_j(*args, **kwargs) - 0.5*self.get_k(*args, **kwargs)

    def get_fock(self, dm, atoms1, atoms2, **kwargs):
        """
        Retrieves the Fock matrix terms.
        Args:
            dm (numpy.ndarray, dict): a dense or sparse density matrix;
            atoms1 (list, tuple): a subset of atoms where the basis functions reside (column index);
            atoms2 (list, tuple): a subset of atoms where the basis functions reside (row index);
            **kwargs: see the description of `self.get_j` and `self.get_k`;

        Returns:
            Fock matrix terms belonging to a given subset of atoms.
        """
        return self.get_hcore(atoms1, atoms2) + self.get_v_eff(dm, atoms1, atoms2, **kwargs)

    def get_orbs(self, dm, atoms, **kwargs):
        """
        Retrieves local HF orbitals and energies.
        Args:
            dm (numpy.ndarray, dict): a dense or sparse density matrix;
            atoms (list, tuple): a subset of atoms where the basis functions of the iteration reside;
            **kwargs: see the description of `self.get_j` and `self.get_k`;

        Returns:
            Fock matrix terms belonging to a given subset of atoms.
        """
        fock = self.get_fock(dm, atoms, atoms, **kwargs)
        ovlp = self.get_ovlp(atoms, atoms)
        return scipy.linalg.eigh(fock, ovlp)


class Domain(object):
    def __init__(self, atoms, provider, partition_matrix=None, core=None):
        """
        Describes a domain composed of atoms.
        Args:
            atoms (list, tuple): a list of atoms of the domain;
            provider (common.IntegralProvider): provider of integral values;
            partition_matrix (numpy.ndarray): partition matrix for this domain;
            core (list, tuple): atoms included in the core of this domain;
        """
        self.atoms = tuple(atoms)
        self.ao = provider.get_atom_basis(atoms)
        self.shell_ranges = provider.shell_ranges(self.atoms)

        n = len(self.ao)

        if partition_matrix is None and core is None:
            self.partition_matrix = numpy.ones((n, n), dtype=float)

        elif partition_matrix is not None:
            self.partition_matrix = partition_matrix

        elif core is not None:
            core_idx = provider.get_atom_basis(core, domain=atoms)
            boundary_idx = provider.get_atom_basis(list(set(atoms) - set(core)), domain=atoms)
            self.partition_matrix = numpy.zeros((n, n), dtype=float)
            self.partition_matrix[numpy.ix_(core_idx, core_idx)] = 1.0
            self.partition_matrix[numpy.ix_(core_idx, boundary_idx)] = 0.5
            self.partition_matrix[numpy.ix_(boundary_idx, core_idx)] = 0.5

        self.d2i = provider.get_block(self.atoms, self.atoms)
        self.hcore = provider.get_hcore(self.atoms, self.atoms)
        self.ovlp = provider.get_ovlp(self.atoms, self.atoms)
        self.eri = provider.get_eri(self.atoms, self.atoms, self.atoms, self.atoms)

        self.mol = provider.__mol__.copy()
        self.mol._bas = numpy.concatenate(tuple(self.mol._bas[start:end] for start, end in self.shell_ranges), axis=0)
        self.mol.nelectron = self.mol.atom_charges()[self.atoms, ].sum()


class DCHF(HFLocalIntegralProvider):
    def __init__(self, mol, distribution_function=common.gaussian_distribution, temperature=30, eri_threshold=1e-12):
        """
        An implementation of divide-conquer Hartree-Fock calculations. The domains are added via `self.add_domain`
        method and stored inside `self.domains` list. Each list item contains all information on the domain including
        the local space description and all relevant integral values.
        Args:
            mol (pyscf.mole.Mole): a Mole object to perform calculations;
            distribution_function (func): a finite-temperature distribution function;
            temperature (float): temperature of the distribution in Kelvin;
            eri_threshold (float): threshold to discard electron repulsion integrals according to Cauchy-Schwartz upper
            boundary;
        """
        super(DCHF, self).__init__(mol)
        self.distribution_function = distribution_function
        self.temperature = temperature*8.621738e-5
        self.eri_threshold = eri_threshold

        self.domains = []

        self.dm = None
        self.hf_energy = None
        self.mu = None
        self.convergence_history = []
        self.eri_j = self.eri_k = None

    def domains_erase(self):
        """
        Erases all domain information.
        """
        self.domains = []

    def add_domain(self, domain, partition_matrix=None, core=None):
        """
        Adds a domain.
        Args:
            domain (list, tuple): a list of atoms included into this domain;
            partition_matrix (numpy.ndarray): partition matrix for this domain;
            core (list, tuple): atoms included in the core of this domain;
        """
        self.domains.append(Domain(domain, self, partition_matrix=partition_matrix, core=core))

    def domains_auto(self, n, **kwargs):
        """
        Splits into domains based on K-means clustering as implemented in scipy.
        Args:
            n (int): the number of domains;
            **kwargs: keyword arguments to scipy.cluster.vq.kmeans2
        """
        default = dict(
            minit="points",
        )
        default.update(kwargs)
        coordinates = self.__mol__.atom_coords()
        centroids, labels = cluster.vq.kmeans2(cluster.vq.whiten(coordinates), n, **default)
        for i in range(n):
            self.add_domain(numpy.argwhere(numpy.array(labels) == i)[:, 0])

    def domains_pattern(self, n):
        """
        Calculates a domain pattern.
        Args:
            n (int): the number of dimensions;

        Returns:
            An `n`-dimensional tensor masking the union of domains.
        """
        result = numpy.zeros((len(self.__ao_ownership__),)*n)
        for d in self.domains:
            result[numpy.ix_(*((d.ao,)*n))] = True
        return result

    def prepare_dm(self):
        """
        Prepares the density matrix.
        """
        if self.dm is None:
            self.dm = scf.hf.get_init_guess(self.__mol__)
        self.dm *= self.domains_pattern(2)
        self.dm *= self.__mol__.nelectron / (self.dm*self.get_ovlp(None, None)).sum()

    def build(self):
        """
        Calculates ERI.
        """
        self.prepare_dm()
        self.eri_j = {}
        self.eri_k = {}
        # Diagonal
        for i, d in enumerate(self.domains):
            self.eri_j[i, i] = d.eri
            self.eri_k[i, i] = d.eri
        # Off-diagonal
        for i, d1 in enumerate(self.domains):
            for j, d2 in enumerate(self.domains):
                if j > i:
                    self.eri_j[i, j] = self.get_eri(d1.atoms, d1.atoms, d2.atoms, d2.atoms)
                    self.eri_j[j, i] = self.eri_j[i, j].transpose((2, 3, 0, 1))
                    self.eri_k[i, j] = self.get_eri(d1.atoms, d2.atoms, d2.atoms, d1.atoms)
                    self.eri_k[j, i] = self.eri_k[i, j].transpose((1, 0, 3, 2))

    def domains_cover(self, r=True):
        """
        Checks whether every atom is present in, at least, one domain.
        Args:
            r (bool): raises an exception if the return value is False;

        Returns:
            True if domains cover all atoms.
        """
        all_atoms = set(range(self.__mol__.natm))
        covered_atoms = set(numpy.concatenate(tuple(i.atoms for i in self.domains), axis=0))
        result = all_atoms == covered_atoms
        if not result and r:
            raise ValueError("Atoms "+",".join(list(
                "{:d}".format(i) for i in (all_atoms - covered_atoms)
            ))+" are not covered by any domain")
        return result

    def update_domain_eigs(self):
        """
        Updates domains' eigenstates and eigenvalues.
        """
        for i, d in enumerate(self.domains):
            # d["h"] = d["hcore"] + numpy.einsum("ijkl,kl->ij", d["eri_j"], self.dm) - 0.5*numpy.einsum("ijkl,jk->il", d["eri_k"], self.dm)
            d.h = d.hcore.copy()
            for j, d2 in enumerate(self.domains):
                dm = self.dm[d2.d2i] * d2.partition_matrix
                d.h += numpy.einsum("ijkl,kl->ij", self.eri_j[i, j], dm) -\
                       0.5*numpy.einsum("ijkl,jk->il", self.eri_k[i, j], dm)
            # __h_test__ = d.hcore +\
            #     numpy.einsum("ijkl,kl->ij", self.get_eri(d.atoms, d.atoms, None, None), self.dm) -\
            #     0.5 * numpy.einsum("ijkl,jk->il", self.get_eri(d.atoms, None, None, d.atoms), self.dm)
            # from numpy import testing
            # testing.assert_allclose(d.h, __h_test__)
            d.e, d.psi = scipy.linalg.eigh(d.h, d.ovlp)
            # d["weights"] = numpy.einsum("ij,kj,ik,ik->j", d["psi"], d["psi"], d["ovlp"], d["partition_matrix"])
            d.weights = numpy.einsum("ij,kj,ik,ik->j", d.psi, d.psi, d.ovlp, d.partition_matrix)

    def update_chemical_potential(self):
        """
        Calculates the chemical potential.
        Returns:
            The chemical potential.
        """
        fock_energies = numpy.concatenate(list(i.e for i in self.domains), axis=0)
        fock_energy_weights = numpy.concatenate(list(i.weights for i in self.domains), axis=0)
        self.mu = scipy.optimize.minimize_scalar(
            lambda x: ((self.distribution_function(x, self.temperature, fock_energies)*fock_energy_weights).sum() - self.__mol__.nelectron) ** 2,
            bounds=(fock_energies.min(), fock_energies.max()),
        )["x"]
        return self.mu

    def update_domain_dm(self):
        """
        Updates density matrixes of domains.
        """
        for domain in self.domains:
            domain.occupations = self.distribution_function(self.mu, self.temperature, domain.e)
            domain.dm = numpy.einsum("ij,j,kj->ik", domain.psi, domain.occupations, domain.psi)

    def update_total_dm(self):
        """
        Updates the total density matrix.
        Returns:
            The maximal deviation from the previous density matrix.
        """
        old_dm = self.dm
        self.dm = numpy.zeros_like(self.dm)
        self.hf_energy = 0
        for domain in self.domains:
            masked_dm = domain.dm * domain.partition_matrix
            self.dm[domain.d2i] += masked_dm
            self.hf_energy += 0.5 * ((domain.h + domain.hcore) * masked_dm).sum()

        return abs(self.dm-old_dm).max()

    def kernel(self, tolerance=1e-6, maxiter=100, dm_hook="diis"):
        """
        Performs self-consistent iterations.
        Args:
            tolerance (float): density matrix convergence criterion;
            maxiter (int): maximal number of iterations;
            dm_hook (func): a hook called right after the density matrix was updated. It is called with a single
            argument, self, and return a new density matrix. It is allowed to not return anything. A special value
            "diis" stands for `pyscf.diis.DIIS.update` with an adjusted input.

        Returns:
            The converged energy value which is also stored as `self.hf_energy`.
        """
        if dm_hook == "diis":
            logger.info(self.__mol__, "Initializing DIIS ...")
            d = diis.DIIS()

            def dm_hook(dchf):
                return d.update(dchf.dm)

        logger.info(self.__mol__, "Checking domain coverage ...")
        self.domains_cover(r=True)
        logger.info(self.__mol__, "Calculating ERI blocks ...")
        self.build()
        self.convergence_history = []

        logger.info(self.__mol__, "Running self-consistent calculation ...")
        while True:
            self.update_domain_eigs()
            self.update_chemical_potential()
            self.update_domain_dm()
            delta = self.update_total_dm()
            if dm_hook is not None:
                result = dm_hook(self)
                if not result is None:
                    self.dm = result
            logger.info(self.__mol__, "  E = {:.10f} delta = {:.3e}".format(
                self.hf_energy,
                delta,
            ))
            self.convergence_history.append(delta)
            if delta < tolerance:
                return self.hf_energy

            if maxiter is not None and len(self.convergence_history) >= maxiter:
                raise RuntimeError("The maximal number of iterations {:d} reached. The error {:.3e} is still above the requested tolerance of {:.3e}".format(
                    maxiter,
                    delta,
                    tolerance,
                ))


def energy_2(domains, w_occ, amplitude_calculator=None, with_t2=True):
    """
    Calculates the second-order energy correction in domain setup.
    Args:
        domains (iterable): a list of domains;
        w_occ (float): a parameter splitting the second-order energy contributions between occupied and virtual
        molecular orbitals;
        amplitude_calculator (func): calculator of second-order amplitudes. If None, then MP2 amplitudes are calculated;
        with_t2 (bool): whether to save amplitudes;

    Returns:
        The energy correction.
    """
    result = 0
    if with_t2:
        result_t2 = []
    else:
        result_t2 = None
    for domain in domains:

        occupations = domain.occupations
        selection_occ = numpy.argwhere(occupations >= 1)[:, 0]
        selection_virt = numpy.argwhere(occupations < 1)[:, 0]

        psi = domain.psi
        psi_occ = psi[:, selection_occ]
        psi_virt = psi[:, selection_virt]

        core_mask = numpy.diag(domain.partition_matrix)[:, numpy.newaxis]
        psi_occ_core = psi_occ * core_mask
        psi_virt_core = psi_virt * core_mask

        __ov = common.transform(common.transform(domain.eri, psi_occ, axes=2), psi_virt, axes=3)
        xvov = common.transform(common.transform(__ov, psi_occ_core, axes=0), psi_virt, axes=1)
        oxov = common.transform(common.transform(__ov, psi_occ, axes=0), psi_virt_core, axes=1)

        if amplitude_calculator is None:
            e = domain.e
            e_occ = e[selection_occ]
            e_virt = e[selection_virt]

            ovov = common.transform(common.transform(__ov, psi_occ, axes=0), psi_virt, axes=1)
            t1 = None
            t2 = ovov / (
                e_occ[:, numpy.newaxis, numpy.newaxis, numpy.newaxis] -
                e_virt[numpy.newaxis, :, numpy.newaxis, numpy.newaxis] +
                e_occ[numpy.newaxis, numpy.newaxis, :, numpy.newaxis] -
                e_virt[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
            )
        else:
            t1, t2 = amplitude_calculator(domain)

        amplitudes = 0
        if t2 is not None:
            amplitudes = t2
        if t1 is not None:
            amplitudes += numpy.einsum("ia,jb->iajb", t1, t1)

        if amplitudes is not 0:
            result += ((xvov * w_occ + oxov * (1.0 - w_occ)) * (2 * amplitudes - numpy.swapaxes(amplitudes, 0, 2))).sum()
        if result_t2 is not None:
            result_t2.append(amplitudes)

    return result, result_t2


def pyscf_mp2_amplitude_calculator(domain):
    """
    Calculates MP2 amplitudes in the domain.
    Args:
        domain (Domain): a domain to calculate at;

    Returns:
        MP2 amplitudes.
    """
    mf = scf.RHF(domain.mol)
    mf.build(domain.mol)
    mf.mo_coeff = domain.psi
    mf.mo_energy = domain.e
    mf.mo_occ = domain.occupations
    domain_mp2 = mp.MP2(mf)
    domain_mp2.kernel()
    return None, domain_mp2.t2.swapaxes(1, 2)


def pyscf_ccsd_amplitude_calculator(domain):
    """
    Calculates CCSD amplitudes in the domain.
    Args:
        domain (Domain): a domain to calculate at;

    Returns:
        CCSD amplitudes.
    """
    mf = scf.RHF(domain.mol)
    mf.build(domain.mol)
    mf.mo_coeff = domain.psi
    mf.mo_energy = domain.e
    mf.mo_occ = domain.occupations
    domain_ccsd = cc.CCSD(mf)
    domain_ccsd.kernel()
    return domain_ccsd.t1, domain_ccsd.t2.swapaxes(1, 2)


class DCMP2(object):
    def __init__(self, dchf, w_occ=1):
        """
        An implementation of the divide-conquer MP2 on top of the divide-conquer Hartree-Fock.
        Args:
            dchf (DCHF): a completed divide-conquer Hartree-Fock calculation
            w_occ (float): a parameter splitting the second-order energy contributions between occupied and virtual
            molecular orbitals;
        """
        self.mf = dchf
        self.w_occ = w_occ

        self.e2 = self.t2 = None

    def kernel(self):
        """
        Calculates DC-MP2 energy and amplitudes.
        Returns:
            DC-MP2 energy correction.
        """
        self.e2, self.t2 = energy_2(self.mf.domains, self.w_occ)
        return self.e2, self.t2


class DCCCSD(DCMP2):
    def __init__(self, dchf, w_occ=1):
        """
        An implementation of the divide-conquer CCSD on top of the divide-conquer Hartree-Fock.
        Args:
            dchf (DCHF): a completed divide-conquer Hartree-Fock calculation
            w_occ (float): a parameter splitting the second-order energy contributions between occupied and virtual
            molecular orbitals;
        """
        DCMP2.__init__(self, dchf, w_occ=w_occ)
        self.e1 = self.t1 = None

    def kernel(self):
        """
        Calculates DC-CCSD energy and amplitudes.
        Returns:
            DC-CCSD energy correction.
        """
        self.e2, self.t2 = energy_2(self.mf.domains, self.w_occ, amplitude_calculator=pyscf_ccsd_amplitude_calculator)

