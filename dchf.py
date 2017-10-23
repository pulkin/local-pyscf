from pyscf import scf
from pyscf.lib import logger

import numpy
import scipy

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


class DCHF(HFLocalIntegralProvider):
    def __init__(self, mol):
        super(DCHF, self).__init__(mol)
        self.domains = []
        self.dm = scf.hf.get_init_guess(mol)
        self.hf_energy = None

    def add_domain(self, domain, domain_partition_matrix=None, domain_core=None):
        """
        Adds a domain.
        Args:
            domain (list, tuple): a list of atoms included into this domain;
            domain_partition_matrix (numpy.ndarray): partition matrix for this domain;
            domain_core (list, tuple): atoms included in the core of this domain;
        """
        domain_basis = self.get_atom_basis(domain)
        if domain_partition_matrix is None and domain_core is None:
            raise ValueError("Either 'domain_partition_matrix' or 'domain_core' has to be specified")
        elif domain_partition_matrix is not None:
            pass
        elif domain_core is not None:
            core_idx = self.get_atom_basis(domain_core, domain=domain)
            boundary_idx = self.get_atom_basis(list(set(domain) - set(domain_core)), domain=domain)
            domain_partition_matrix = numpy.zeros((len(domain_basis), len(domain_basis)), dtype=float)
            domain_partition_matrix[numpy.ix_(core_idx, core_idx)] = 1.0
            domain_partition_matrix[numpy.ix_(core_idx, boundary_idx)] = 0.5
            domain_partition_matrix[numpy.ix_(boundary_idx, core_idx)] = 0.5
        self.domains.append({
            "domain": domain,
            "basis": domain_basis,
            "partition_matrix": domain_partition_matrix,
            "eri_j": self.get_eri(domain, domain, None, None),
            "eri_k": self.get_eri(domain, None, None, domain),
            "hcore": self.get_hcore(domain, domain),
            "ovlp": self.get_ovlp(domain, domain),
        })

    def update_domain_eigs(self):
        """
        Updates domains' eigenstates and eigenvalues.
        """
        for d in self.domains:
            d["h"] = d["hcore"] + numpy.einsum("ijkl,kl->ij", d["eri_j"], self.dm) - 0.5*numpy.einsum("ijkl,jk->il", d["eri_k"], self.dm)
            d["e"], d["psi"] = scipy.linalg.eigh(d["h"], d["ovlp"])
            d["weights"] = numpy.einsum("ij,kj,ik,ik->j", d["psi"], d["psi"], d["ovlp"], d["partition_matrix"])

    def update_dm(self):
        """
        Updates the density matrix.
        """
        all_e = numpy.concatenate(list(i["e"] for i in self.domains), axis=0)
        all_did = numpy.concatenate(list((j,)*len(i["e"]) for j, i in enumerate(self.domains)), axis=0)
        all_sid = numpy.concatenate(list(numpy.arange(len(i["e"])) for i in self.domains), axis=0)

        order = numpy.argsort(all_e)

        occupied = 0
        old_dm = self.dm
        self.dm = numpy.zeros_like(self.dm)
        self.hf_energy = 0
        break_after = False
        for i in order:
            domain = self.domains[all_did[i]]
            j = all_sid[i]
            psi = domain["psi"][:, j]
            pm = domain["partition_matrix"]
            weight = 2*domain["weights"][j]
            space = domain["basis"]
            h = domain["h"]
            hcore = domain["hcore"]
            dm = 2*numpy.outer(psi, psi)*pm
            if occupied+weight > self.mol.nelectron:
                dm /= weight
                weight = self.mol.nelectron-occupied
                dm *= weight
                break_after = True
            self.dm[numpy.ix_(space, space)] += dm
            self.hf_energy += 0.5*((h+hcore)*dm).sum()
            occupied += weight
            if break_after:
                break
        return abs(self.dm-old_dm).max()

    def iter_hf(self, tolerance=1e-6):
        self.dm = scf.get_init_guess(self.mol)
        while True:
            self.update_domain_eigs()
            delta = self.update_dm()
            logger.info(self.mol, "  E = {:.10f} delta = {:.3e}".format(
                self.hf_energy,
                delta,
            ))
            if delta < tolerance:
                return self.hf_energy
