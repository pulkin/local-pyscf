from pyscf.lib import logger, diis
from pyscf.ao2mo import restore

import numpy
import scipy
from common import IntegralProvider, transform, NonSelfConsistentMeanField

import uuid
from collections import OrderedDict


def m2utri(a):
    """
    Retrieves a plain array with upper-triangular matrix values.
    Args:
        a (numpy.ndarray): a matrix to process;

    Returns:
        A plain array with values of the upper-triangular part of the matrix.
    """
    return a[numpy.triu_indices(a.shape[0])]


def utri2m(a):
    """
    Transforms a plain array with upper-triangular matrix values into a full array.
    Args:
        a (numpy.ndarray): a plain array to process;

    Returns:
        A square Hemritian matrix.
    """
    d = (1+8*len(a))**.5
    if d % 2 != 1:
        raise ValueError("Matrix has a wrong size {:d}".format(len(a)))
    d = (int(d)-1)/2
    result = numpy.empty((d, d), dtype=a.dtype)
    indices = numpy.triu_indices(d)
    result[indices] = a
    result[(indices[1], indices[0])] = a.conj()
    return result


class GenericDMETUmatSelfConsistency(object):
    def __init__(self, driver, umat_projector, reference_solution, dm_projector=None, log=None, solution_dm_slice=None):
        """
        Consistency between the u-matrix and the density matrix of the projected fragment.
        Args:
            driver: a mean-field object;
            umat_projector (numpy.ndarray): a projector for the u-matrix from the local domain to the entire model;
            reference_solution: an arbitrary driver capable of calculating one-particle density matrix or
            the one-particle density matrix itself;
            dm_projector (numpy.ndarray): a projector for the density matrix from the entire model onto the local domain;
            log: a pyscf object to log to;
            solution_dm_slice (slice): an optional slice to apply to the reference density matrix;
        """
        self.driver = driver
        self.__original_driver_hcore__ = driver.get_hcore
        self.umat_projector = umat_projector
        self.dm_projector = umat_projector if dm_projector is None else dm_projector

        self.reference_dm = None
        self.set_reference_dm(reference_solution, solution_dm_slice=solution_dm_slice)

        if log is not None:
            self.__log__ = log
        else:
            self.__log__ = driver.mol

        self.initial_guess = None
        self.final_parameters = None

        self.__previous_umat_value__ = None
        self.__previous_value__ = None
        self.__previous_jac_value__ = None

    def set_reference_dm(self, solver, solution_dm_slice=None):
        """
        Sets the reference density matrix from the solver.
        Args:
            solver: an arbitrary driver capable of calculating one-particle density matrix or
            the one-particle density matrix itself;
            solution_dm_slice (slice): an optional slice to apply to the reference density matrix;
        """
        n = self.dm_projector.shape[1]
        if isinstance(solver, numpy.ndarray):
            dm = solver
        else:
            dm = solver.make_rdm1()
        if solution_dm_slice is not None:
            dm = dm[solution_dm_slice, solution_dm_slice]
        if dm.shape != (n, n):
            raise ValueError("The shape of the reference density matrix {} is expected to be {:d}".format(
                dm.shape,
                n
            ))
        self.reference_dm = dm

    def get_default_guess(self):
        """
        Retrieves the default guess.
        Returns:
            The default guess.
        """
        n = self.dm_projector.shape[1]
        return numpy.zeros(n*n)

    def parametrize_umat(self, parameters):
        """
        Calculates the u-matrix in the projected basis.
        Args:
            parameters (numpy.ndarray): parameters of the u-matrix;

        Returns:
            The u-matrix value.
        """
        n = self.umat_projector.shape[1]
        return parameters.reshape(n, n)

    def parametrize_umat_full(self, parameters):
        """
        Calculates the u-matrix in the full basis.
        Args:
            parameters (numpy.ndarray): parameters of the u-matrix;

        Returns:
            The u-matrix value.
        """
        return transform(self.parametrize_umat(parameters), self.umat_projector.T)

    def assign_parameters(self, parameters):
        """
        An interface to set the correlation potential "umat" from the given parameters.
        Args:
            parameters (numpy.ndarray): the u-matrix to set;
        """
        shape = self.__original_driver_hcore__().shape
        umat = self.parametrize_umat_full(parameters)
        if not shape == umat.shape:
            raise ValueError("The shape of u-mat {} differs from that of hcore {}".format(umat.shape, shape))
        self.driver.get_hcore = lambda *args, **kwargs: \
            self.__original_driver_hcore__(*args, **kwargs) + umat

    def cleanup_parameters(self):
        """
        Removes parametrization of `self.driver`.
        """
        self.driver.get_hcore = self.__original_driver_hcore__

    def target_dm_function(self, dm):
        """
        Calculates the target function of the calculated density matrix.
        Args:
            dm (numpy.ndarray): the calculated density matrix value.

        Returns:
            The target function value (scalar).
        """
        return (abs(self.reference_dm - dm)**2).sum()

    def f(self, parameters):
        """
        Finds the norm of the density matrix difference as a function of parameters.
        Args:
            parameters (numpy.ndarray): parameters to set;

        Returns:
            The density matrix difference.
        """
        self.assign_parameters(parameters)
        self.driver.kernel()
        logger.debug1(self.__log__, "| total energy {:.10f}".format(self.driver.e_tot))
        self.cleanup_parameters()
        return self.target_dm_function(transform(self.driver.make_rdm1(), self.dm_projector))

    def f_cached(self, parameters):
        """
        A cached version of `self.f`
        """
        logger.debug1(self.__log__, "| requested {}".format(parameters))

        if numpy.all(self.__previous_umat_value__ == parameters):
            logger.debug1(self.__log__, "| result from cache {}".format(self.__previous_value__))
            return self.__previous_value__

        else:
            result = self.f(parameters)
            logger.debug1(self.__log__, "| result {}".format(result))
            if self.__previous_umat_value__ is not None and self.__previous_jac_value__ is not None:
                step = result - self.__previous_value__
                proposed_step = self.__previous_jac_value__.dot(parameters - self.__previous_umat_value__)
                logger.debug1(self.__log__, "| non-linearity {:.3e}".format(
                    (step - proposed_step) / step,
                ))
            self.__previous_umat_value__ = parameters.copy()
            self.__previous_value__ = result
            self.__previous_jac_value__ = None
            return result

    def raw_gradients(self):
        """
        Calculates gradients using the following expression for the derivative:

        .. math::
            \frac{\partial D}{\partial \delta} = C_{occ} Z^\dagger C_{virt}^\dagger + C_{virt} Z C_{occ}^\dagger,

        .. math::
            Z = - \frac{C_{vir}^\dagger H^{(1)} C_{occ}}{E_{vir} - E_{occ}}

        Returns:
            Gradients.
        """
        # Occupied and virtual MOs
        occ = self.driver.mo_coeff[:, self.driver.mo_occ > 1]
        occ_e = self.driver.mo_energy[self.driver.mo_occ > 1]
        virt = self.driver.mo_coeff[:, self.driver.mo_occ <= 1]
        virt_e = self.driver.mo_energy[self.driver.mo_occ <= 1]

        gap = virt_e.min() - occ_e.max()
        logger.debug1(self.__log__, "| energy gap {:.3e}".format(
            gap,
        ))

        if virt_e.min() - occ_e.max() < 1e-4:
            logger.warn(self.__log__, "The energy gap is too small: {:.3e}".format(gap))

        # Project MOs
        occ_umat = transform(occ, self.umat_projector, axes=0)
        virt_umat = transform(virt, self.umat_projector, axes=0)
        occ_dm = transform(occ, self.dm_projector, axes=0)
        virt_dm = transform(virt, self.dm_projector, axes=0)

        denominator = occ_e[numpy.newaxis, :] - virt_e[:, numpy.newaxis]
        numenator = virt_umat[:, numpy.newaxis, :, numpy.newaxis] * occ_umat[numpy.newaxis, :, numpy.newaxis, :]
        z = numenator / denominator[numpy.newaxis, numpy.newaxis, :, :]

        return numpy.einsum("ij,abkj,lk->abil", occ_dm, z, virt_dm) + numpy.einsum("ij,abjk,lk->abil", virt_dm, z, occ_dm)

    def gradients(self):
        """
        Calculates gradients for the target function.
        Returns:
            Gradients.
        """
        return 2 * numpy.einsum(
            "ijkl,kl->ij",
            self.raw_gradients(),
            transform(self.driver.make_rdm1(), self.dm_projector) - self.reference_dm
        ).reshape(-1)

    def gradients_cached(self, parameters):
        """
        Cached version of `self.gradients`.
        """
        logger.debug1(self.__log__, "| requested gradients {}".format(parameters))

        # Check if the value of the norm has been computed
        if not numpy.all(self.__previous_umat_value__ == parameters):
            self.f_cached(parameters)

        # Check if the value of the gradient has been computed
        if self.__previous_jac_value__ is not None:
            logger.debug1(self.__log__, "| gradients from cache {}".format(self.__previous_jac_value__))
            return self.__previous_jac_value__

        result = self.gradients()
        self.__previous_jac_value__ = result
        logger.debug1(self.__log__, "| gradients {}".format(result))
        return result

    def kernel(self, x0=None, **kwargs):
        """
        Uses a scipy minimize the local density matrix difference.
        Args:
            x0 (numpy.ndarray): initial guess for the upper-triangular part of u-mat;
            **kwargs: additional keyword arguments to `scipy.optimize.minimize`;

        Returns:
            The optimal correlation potential matrix u-mat.
        """
        self.initial_guess = x0 if x0 is not None else self.get_default_guess()
        logger.debug1(self.__log__, "Minimizing the density matrix difference ...")
        logger.debug1(self.__log__, "Reference density matrix {}".format(self.reference_dm))
        kwargs["jac"] = self.gradients_cached
        result = scipy.optimize.minimize(self.f_cached, self.initial_guess, **kwargs)
        # kwargs["Dfun"] = self.jacobian
        # self.final_parameters = scipy.optimize.leastsq(self, self.initial_guess, **kwargs)
        if not result["success"]:
            raise ValueError("No minimum found")
        logger.debug1(self.__log__, "Reached minimum in {:d} iterations".format(
            result["nit"],
        ))
        self.final_parameters = result['x']
        return self.final_parameters


class GenericUtriDMETUmatSelfConsistency(GenericDMETUmatSelfConsistency):
    def get_default_guess(self):
        n = self.umat_projector.shape[1]
        return numpy.zeros(n * (n+1) // 2)
    get_default_guess.__doc__ = GenericDMETUmatSelfConsistency.get_default_guess.__doc__

    def parametrize_umat(self, parameters):
        return GenericDMETUmatSelfConsistency.parametrize_umat(self, utri2m(parameters))
    parametrize_umat.__doc__ = GenericDMETUmatSelfConsistency.parametrize_umat.__doc__

    def gradients(self):
        n = self.umat_projector.shape[1]
        gradients = GenericDMETUmatSelfConsistency.gradients(self).reshape(n, n)
        gradients = gradients + gradients.T - numpy.diag(numpy.diag(gradients))
        return m2utri(gradients)
    gradients.__doc__ = GenericDMETUmatSelfConsistency.gradients.__doc__


class FragmentFragmentDMETUSC(GenericUtriDMETUmatSelfConsistency):
    def __init__(self, driver, projector, reference_solution, log=None):
        """
        Consistency between the fragment u-matrix and the fragment density matrix.
        Args:
            driver: a mean-field object;
            projector (numpy.ndarray): a projector for the local fragment;
            reference_solution: an arbitrary driver capable of calculating one-particle density matrix or
            the one-particle density matrix itself;
            log: a pyscf object to log to;
        """
        n = projector.shape[1]
        GenericUtriDMETUmatSelfConsistency.__init__(
            self,
            driver,
            projector[:, :n // 2],
            reference_solution,
            log=log,
            solution_dm_slice=slice(None, n // 2)
        )


class FragmentFullDMETUSC(GenericUtriDMETUmatSelfConsistency):
    def __init__(self, driver, projector, reference_solution, log=None):
        """
        Consistency between the fragment u-matrix and the full density matrix.
        Args:
            driver: a mean-field object;
            projector (numpy.ndarray): a projector for the local fragment;
            reference_solution: an arbitrary driver capable of calculating one-particle density matrix or
            the one-particle density matrix itself;
            log: a pyscf object to log to;
        """
        n = projector.shape[1]
        GenericUtriDMETUmatSelfConsistency.__init__(
            self,
            driver,
            projector[:, :n // 2],
            reference_solution,
            dm_projector=projector,
            log=log,
        )


class MuFragmentDMETUSC(GenericDMETUmatSelfConsistency):
    def __init__(self, driver, projector, reference_solution, log=None):
        """
        Consistency between the chemical potential of the fragment and the fragment density matrix.
        Args:
            driver: a mean-field object;
            projector (numpy.ndarray): a projector for the local fragment;
            reference_solution: an arbitrary driver capable of calculating one-particle density matrix or
            the one-particle density matrix itself;
            log: a pyscf object to log to;
        """
        n = projector.shape[1]
        GenericDMETUmatSelfConsistency.__init__(
            self,
            driver,
            projector[:, :n // 2],
            reference_solution,
            log=log,
            solution_dm_slice=slice(None, n // 2)
        )

    def get_default_guess(self):
        return 0
    get_default_guess.__doc__ = GenericDMETUmatSelfConsistency.get_default_guess.__doc__

    def parametrize_umat(self, chemical_potential):
        """
        Calculates the u-matrix in the projected basis.
        Args:
            chemical_potential (float): the chemical potential;

        Returns:
            The u-matrix value.
        """
        return GenericDMETUmatSelfConsistency.parametrize_umat(
            self,
            -numpy.eye(self.umat_projector.shape[1]) * chemical_potential,
        )

    def gradients(self):
        n = self.umat_projector.shape[1]
        gradients = GenericDMETUmatSelfConsistency.gradients(self).reshape(n, n)
        return numpy.array([-numpy.diag(gradients).sum()])
    gradients.__doc__ = GenericDMETUmatSelfConsistency.gradients.__doc__


def get_sd_schmidt_basis(basis, domain, threshold=1e-14):
    """
    Retrieves the Schmidt basis for Slater determinants.
    Args:
        basis (numpy.ndarray): occupied single-particle orbitals forming the determinant;
        domain (iterable): (smaller) domain basis function indexes;
        threshold (float): a threshold to determine frozen orbitals;

    Returns:
        Four arrays of basis functions including frozen occupied orbitals residing inside and outside domain and
        active partially occupied orbitals residing inside and outside of the domain. All functions are padded
        with zeros to maintain the original basis size.
    """
    # Prepare indexes and test othonormality of the domain
    domain = list(set(domain))
    domain_conj = list(set(range(basis.shape[0])) - set(domain))

    # Diagonalize the overlap matrix inside the domain
    basis_domain = basis[domain, :]
    ovlp = basis_domain.T.dot(basis_domain)
    zeta, rotation = numpy.linalg.eigh(ovlp)

    # Stick boundary eigenvalues
    zeta[zeta < threshold] = 0  # These states are already localized outside the domain
    zeta[zeta > 1.0-threshold] = 1  # These states are already localized inside the domain

    # Rotate
    rotated_basis = numpy.dot(basis, rotation)

    # Active
    selection_active = numpy.logical_and(0 < zeta, zeta < 1)
    basis_active = rotated_basis[:, selection_active]
    nontrivial_zeta = zeta[selection_active]
    # Form 2N domain-specific active orbitals out of N active orbitals
    active_domain = basis_active[domain, :] / nontrivial_zeta[numpy.newaxis, :]**.5
    active_conj = basis_active[domain_conj, :] / (1.-nontrivial_zeta)[numpy.newaxis, :]**.5

    # Restore the original shape of active orbitals
    active_domain_full = numpy.zeros((basis.shape[0], active_domain.shape[1]), dtype=active_domain.dtype)
    active_domain_full[domain, :] = active_domain
    active_conj_full = numpy.zeros((basis.shape[0], active_conj.shape[1]), dtype=active_conj.dtype)
    active_conj_full[domain_conj, :] = active_conj

    # Frozen
    frozen_domain = rotated_basis[:, zeta == 1]
    frozen_conj = rotated_basis[:, zeta == 0]

    return frozen_domain, frozen_conj, active_domain_full, active_conj_full


def freeze(hcore, eri, n, e_vac=0):
    """
    Recalculates model parameters under the frozen-orbital approximation.
    Args:
        hcore (numpy.ndarray):
        eri (numpy.ndarray): electron repulsion integrals;
        n (int): number of of orbitals to freeze. First `n` orbitals from the basis are frozen;
        e_vac (float): the vacuum level;

    Returns:
        The new (smaller) `hcore`, `eri`, `e_vac` representing the frozen-core approximation.
    """
    new_e_vac = e_vac + 2 * numpy.diag(hcore)[:n].sum() + 2 * numpy.einsum("mmnn", eri[:n, :n, :n, :n]) - numpy.einsum(
        "mnnm", eri[:n, :n, :n, :n])

    new_hcore = hcore[n:, n:] + 2 * numpy.einsum("mpnn->mp", eri[n:, n:, :n, :n]) - numpy.einsum(
        "mnnp->mp", eri[n:, :n, :n, n:])

    new_eri = eri[n:, n:, n:, n:]
    return new_hcore, new_eri, new_e_vac


class GlobalChemicalPotentialFit(object):
    def __init__(self, solvers, factors, target, log=None):
        self.solvers = solvers
        self.factors = factors
        self.target = target

        if log is not None:
            self.__log__ = log
        else:
            self.__log__ = solvers[0].mol

    def nelec(self, mu):

        logger.debug1(self.__log__, "| requested mu={:.3e}".format(
            mu,
        ))

        result = []
        total = 0
        for ind, (i, c) in enumerate(zip(self.solvers, self.factors)):
            n = i.__hcore__.shape[0] // 2
            i.__hcore__[:n, :n] -= numpy.eye(n) * mu
            i.kernel()
            i.__hcore__[:n, :n] += numpy.eye(n) * mu
            dm = i.make_rdm1()
            result.append(numpy.diag(dm)[:n].sum())
            total += c*result[-1]
            logger.debug2(self.__log__, "| solver {:d} E={:.8f}".format(
                ind,
                i.e_tot,
            ))

        logger.debug1(self.__log__, "| occupations {} = {:.3e}".format(
            "+".join("{:.1f}*{:.3f}".format(i, j) for i, j in zip(self.factors, result)),
            total,
        ))

        return total-self.target

    def kernel(self, x0=None, **kwargs):

        logger.debug(self.__log__, "Target: {:.10f} electrons".format(
            self.target,
        ))

        chemical_potential = scipy.optimize.newton(
            self.nelec,
            0 if x0 is None else x0,
            **kwargs
        )

        logger.debug(self.__log__, "Resulting chemical potential is {:.3e}".format(
            chemical_potential,
        ))

        return chemical_potential


class DMET(object):
    def __init__(
            self,
            cheap,
            expensive,
            self_consistency,
            domains,
            nested_verbosity=0,
            freeze=False,
            associate=None,
            style="interacting-bath",
            conv_tol=1e-5,
            schmidt_threshold=1e-14,
    ):
        """
        A DMET driver.
        Args:
            cheap: an initialized cheap mean-field solver of the entire system;
            expensive: a class performing an expensive calculation of the embedded system;
            self_consistency: a self-consistency driver;
            domains (list): a nested list with basis functions to be included into local domains;
            nested_verbosity (int): the verbosity level of the nested output;
            freeze (bool): freeze the mean-field effective field;
            associate (list): a list defining domain IDs. Domains with the same ID are assumed to
            be the same and to be embedded in the same environment;
            style (str): either 'ineracting-bath' or 'non-interacting-bath';
            conv_tol (float): the convergence criterion;
            schmidt_threshold (float): a threshold to assign frozen orbitals in Schmidt decomposition
            of a mean-field solution;
        """
        self.__mol__ = cheap.mol.copy()

        if freeze:
            self.__mf_solver__ = NonSelfConsistentMeanField(cheap)
        else:
            self.__mf_solver__ = cheap
            cheap.verbose = nested_verbosity
        self.__mf_solver__.mol.verbose = nested_verbosity
        self.__nested_verbosity__ = nested_verbosity
        self.__correlated_solver__ = expensive
        self.__self_consistency__ = self_consistency

        style_options = ("interacting-bath", "non-interacting-bath")
        if style not in style_options:
            raise ValueError("The 'style' keyword argument must be either of {}".format(style_options))
        self.__style__ = style
        self.conv_tol = conv_tol
        self.__schmidt_threshold__ = schmidt_threshold

        if associate == "all":
            associate = ("default",) * len(domains)

        if associate is None:
            self.__domains__ = OrderedDict((uuid.uuid4(), [i]) for i in domains)
        else:
            self.__domains__ = OrderedDict()
            for i, (domain, domain_id) in enumerate(zip(domains, associate)):
                if domain_id in self.__domains__:
                    prev = self.__domains__[domain_id][-1]
                    if len(prev) != len(domain):
                        raise ValueError("The size of the domain {:d} is different from one of the previously associated with ID {}: {:d}".format(
                            len(domain),
                            repr(domain_id),
                            len(prev),
                        ))
                    self.__domains__[domain_id].append(domain)
                else:
                    self.__domains__[domain_id] = [domain]
        self.check_domains()

        self.__orthogonal_basis__ = self.get_orthogonal_basis()
        self.__orthogonal_basis_inv__ = scipy.linalg.inv(self.__orthogonal_basis__)

        self.convergence_history = []
        self.e_tot = None

        self.umat = None

    def check_domains(self):
        """
        Performs a check whether domains cover the whole space and do not overlap. Raises an exception otherwise.
        """
        n = len(self.__mf_solver__.mo_energy)
        space = numpy.zeros(n, dtype=int)
        for domain_id, domain_list in self.__domains__.items():
            for d in domain_list:
                space[d] += 1
        if not numpy.all(space == 1):
            if (space == 0).sum() > 0:
                raise ValueError(
                    "Basis functions {} do not belong to either of the domains".format(numpy.argwhere(space == 0)[:, 0])
                )
            elif (space > 1).sum() > 0:
                raise ValueError(
                    "Basis functions {} belong to two or more domains".format(numpy.argwhere(space > 1)[:, 0])
                )
            else:
                raise ValueError("Internal error")

    def get_orthogonal_basis(self):
        """
        Calculates an orthogonal basis for this model.
        Returns:
            Orthogonal wavefunctions expressed in atomic orbitals.
        """
        return scipy.linalg.sqrtm(self.__mf_solver__.get_ovlp())

    def run_mf_kernel(self):
        """
        Runs a mean-field kernel.
        Returns:
            The driver.
        """
        driver_orig_hcore = self.__mf_solver__.get_hcore
        if self.umat is not None:
            self.__mf_solver__.get_hcore = lambda *args, **kwargs: driver_orig_hcore(*args, **kwargs) + self.get_umat()
        self.__mf_solver__.kernel()
        self.__mf_solver__.get_hcore = driver_orig_hcore
        return self.__mf_solver__

    def iter_schmidt_basis(self):
        """
        Retrieves Schmidt single-particle basis sets for each of the unique domain.

        Returns:
            Domain ID, domain basis functions and Schmidt single-particle basis sets. Schmidt basis includes
            domain frozen orbitals, embedding frozen orbitals, domain active orbitals and embedding active orbitals.
        """
        occ = self.__mf_solver__.mo_coeff[:, self.__mf_solver__.mo_occ > 1]
        occ_orth = transform(occ, self.__orthogonal_basis__, axes=0)
        for domain_id, domain_list in self.__domains__.items():
            d = domain_list[0]
            ffaa = get_sd_schmidt_basis(occ_orth, d, threshold=self.__schmidt_threshold__)
            n = len(domain_list[0])
            if ffaa[2].shape[1] == n:
                ffaa[2][:n, :] = numpy.eye(n)
            yield domain_id, d, ffaa

    def get_umat(self, exclude=None):
        """
        Sums all u-matrices.
        Args:
            exclude (str): exclude the primary domain matrix with the specified ID;

        Returns:
            The value of the u-matrix.
        """
        result = 0
        for domain_id, domain_list in self.__domains__.items():
            umat = self.umat[domain_id]
            if exclude != domain_id:
                result += umat
            primary = domain_list[0]
            for secondary in domain_list[1:]:
                shift = secondary - primary
                if not numpy.all(shift == shift[0]):
                    raise ValueError("Only translational invariance is supported so far")
                shift = shift[0]
                result += numpy.roll(numpy.roll(umat, shift, axis=0), shift, axis=1)
        return result

    def convergence_measure(self, new_umat):
        """
        Retrieves the convergence measure.
        Args:
            new_umat (dict): updated values of the u-matrix;

        Returns:
            A float representing the cumulative change in u-matrices.
        """
        result = 0
        for k in self.__domains__:
            diff = new_umat[k] - self.umat[k] if self.umat is not None else new_umat[k]
            result += numpy.linalg.norm(diff)
        return result

    def get_embedded_solver(self, schmidt_basis, kind="interacting"):
        """
        Prepares an embedded solver for the given Schmidt basis.
        Args:
            schmidt_basis (tuple, list): a list of frozen and active orbitals;
            kind (int): local fragment ID for the non-interacting bath formulation or "interacting" otherwise;

        Returns:
            The embedded solver.
        """
        f1, f2, a1, a2 = schmidt_basis
        n_frozen = f1.shape[1] + f2.shape[1]
        n_active = a1.shape[1] + a2.shape[1]
        logger.debug1(self.__mol__, "Active orbitals (domain+embedding): {:d}+{:d}".format(
            a1.shape[1], a2.shape[1],
        ))
        logger.debug1(self.__mol__, "Frozen orbitals (domain+embedding): {:d}+{:d}".format(
            f1.shape[1], f2.shape[1],
        ))
        hcore_ao = self.__mf_solver__.get_hcore()

        if kind in self.__domains__:
            if self.umat is not None:
                hcore_ao = hcore_ao + self.get_umat(exclude=kind)
            schmidt_projection = self.__orthogonal_basis_inv__.T.dot(numpy.concatenate((a1, a2), axis=1))
            logger.debug(self.__mol__, "Transforming active orbitals ...")
            hcore = transform(hcore_ao, schmidt_projection)
            n = a1.shape[1]
            partial_eri = transform(restore(1, self.__mf_solver__._eri, hcore_ao.shape[0]), schmidt_projection[:, :n])
            eri = numpy.zeros((n_active,) * 4, dtype=partial_eri.dtype)
            eri[:n, :n, :n, :n] = partial_eri
            e_vac = self.__mf_solver__.energy_nuc()

        elif kind == "interacting":
            schmidt_projection = self.__orthogonal_basis_inv__.T.dot(numpy.concatenate(schmidt_basis, axis=1))
            logger.debug(self.__mol__, "Transforming orbitals ...")
            hcore = transform(hcore_ao, schmidt_projection)
            eri = transform(restore(1, self.__mf_solver__._eri, hcore_ao.shape[0]), schmidt_projection)
            e_vac = self.__mf_solver__.energy_nuc()
            logger.debug1(self.__mol__, "Freezing external orbitals ...")
            hcore, eri, e_vac = freeze(hcore, eri, n_frozen, e_vac=e_vac)

        else:
            raise ValueError("Unknown kind: {}".format(kind))

        return self.__correlated_solver__(
            hcore,
            eri,
            nelectron=n_active,
            e_vac=e_vac,
            verbose=self.__nested_verbosity__,
        )

    def kernel(self, tolerance="default", maxiter=30):
        """
        Performs a self-consistent DMET calculation.
        Args:
            tolerance (float): convergence criterion;
            maxiter (int): maximal number of iterations;
        """
        if tolerance == "default":
            tolerance = self.conv_tol

        self.convergence_history = []

        # mixers = dict((k, diis.DIIS()) for k in self.__domains__.keys())

        while True:

            logger.info(self.__mol__, "DMET step {:d}".format(
                len(self.convergence_history),
            ))

            mf = self.run_mf_kernel()

            umat = {}

            logger.debug1(self.__mol__, "Mean-field solver total energy E = {:.10f}".format(
                mf.e_tot,
            ))

            self.e_tot = 0
            total_occupation = 0

            domain_ids = []
            embedded_solvers = []
            replica_numbers = []
            schmidt_bases = []

            # Build embedded solvers
            logger.info(self.__mol__, "Building embedded solvers ...")
            for domain_id, domain_basis, schmidt_basis in self.iter_schmidt_basis():

                domain_ids.append(domain_id)
                if self.__style__ == "interacting-bath":
                    embedded_solvers.append(self.get_embedded_solver(schmidt_basis))
                elif self.__style__ == "non-interacting-bath":
                    embedded_solvers.append(self.get_embedded_solver(schmidt_basis, kind=domain_id))
                else:
                    raise ValueError("Internal error: unknown style '{}'".format(self.__style__))
                replica_numbers.append(len(self.__domains__[domain_id]))
                schmidt_bases.append(schmidt_basis[2:])

            # Fit chemical potential
            logger.info(self.__mol__, "Fitting chemical potential ...")
            GlobalChemicalPotentialFit(
                embedded_solvers,
                replica_numbers,
                self.__mol__.nelectron,
                log=self.__mol__,
            ).kernel()

            # Fit the u-matrix
            logger.info(self.__mol__, "Fitting the u-matrix ...")
            for domain_id, embedded_solver, schmidt_basis, nreplica in zip(domain_ids, embedded_solvers, schmidt_bases, replica_numbers):

                logger.debug(self.__mol__, "Domain {}".format(domain_id))
                logger.debug1(self.__mol__, "Primary basis: {}".format(self.__domains__[domain_id][0]))
                if len(self.__domains__[domain_id]) > 1:
                    for i, b in enumerate(self.__domains__[domain_id][1:]):
                        logger.debug1(self.__mol__, "Secondary basis {:d}: {}".format(i, b))

                logger.debug1(self.__mol__, "Correlated solver total energy E = {:.10f}".format(
                    embedded_solver.e_tot,
                ))

                n_active_domain = schmidt_basis[0].shape[1]
                # TODO: fix this; no need to recalculate hcore
                partial_energy = embedded_solver.partial_etot(
                    slice(n_active_domain),
                    transform(
                        self.__mf_solver__.get_hcore(),
                        self.__orthogonal_basis_inv__.T.dot(numpy.concatenate(schmidt_basis, axis=1)),
                    ),
                )
                self.e_tot += nreplica * partial_energy

                logger.debug1(self.__mol__, "Correlated solver partial energy E = {:.10f}".format(
                    partial_energy,
                ))

                partial_occupation = embedded_solver.partial_nelec(slice(n_active_domain))
                total_occupation += nreplica * partial_occupation

                logger.debug1(self.__mol__, "Correlated solver partial occupation N = {:.10f}".format(
                    partial_occupation,
                ))

                logger.debug2(self.__mol__, "Correlated solver density matrix: {}".format(embedded_solver.make_rdm1()))

                if tolerance is not None:
                    # Continue with self-consistency
                    nscf_mf = NonSelfConsistentMeanField(mf)
                    nscf_mf.kernel()

                    sc = self.__self_consistency__(
                        nscf_mf,
                        self.__orthogonal_basis__.dot(numpy.concatenate(schmidt_basis, axis=1)),
                        embedded_solver,
                        log=self.__mol__,
                    )
                    sc.kernel(x0=None)

                    local_umat = sc.parametrize_umat_full(sc.final_parameters)
                    umat[domain_id] = local_umat

                    logger.debug(self.__mol__, "Parameters: {}".format(
                        sc.final_parameters,
                    ))

            self.e_tot += mf.energy_nuc()

            if tolerance is not None:
                self.convergence_history.append(self.convergence_measure(umat))
                self.umat = umat
                # self.umat = dict((k, mixers[k].update(umat[k])) for k in self.__domains__.keys())

                logger.info(self.__mol__, "E = {:.10f} delta = {:.3e} q = {:.3e} max(umat) = {:.3e}".format(
                    self.e_tot,
                    self.convergence_history[-1],
                    self.__mol__.nelectron - total_occupation,
                    max(v.max() for v in self.umat.values()),
                ))

            else:

                logger.info(self.__mol__, "E = {:.10f} q = {:.3e}".format(
                    self.e_tot,
                    self.__mol__.nelectron - total_occupation,
                ))

            if tolerance is None or self.convergence_history[-1] < tolerance:
                return self.e_tot

            if maxiter is not None and len(self.convergence_history) >= maxiter:
                raise RuntimeError("The maximal number of iterations {:d} reached. The error {:.3e} is still above the requested tolerance of {:.3e}".format(
                    maxiter,
                    self.convergence_history[-1],
                    tolerance,
                ))


class AbInitioDMET(DMET):
    def __init__(self, cheap, expensive, self_consistency, domains, **kwargs):
        """
        Ab-initio DMET driver where domains are atoms rather than basis functions.
        Args:
            cheap: an initialized cheap mean-field solver of the entire system;
            expensive: a class performing an expensive calculation of the embedded system;
            self_consistency: a self-consistency driver;
            domains (list): a nested list with basis functions to be included into local domains;
            **kwargs: keyword arguments to `DMET`;
        """
        ip = IntegralProvider(cheap.mol)
        domains = list(ip.get_atom_basis(i) for i in domains)
        DMET.__init__(self, cheap, expensive, self_consistency, domains, **kwargs)
