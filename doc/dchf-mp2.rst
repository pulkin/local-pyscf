*****************************************
Divide-conquer MP2 by Kobayashi and Nakai
*****************************************

Errors in MP2 energies
======================

Hartree-Fock (HF) and second-order correction energies (MP2) obtained with DC approach compared to conventional calculations.

.. plot:: plots/30-dcmp2-errors_cached.py

Domain size dependence
======================

Errors in HF+MP2 energies as a function of the domain size and run time.
The benchmark model is a dimerized hydrogen chain of size 24.

.. plot:: plots/31-dcmp2-errors-domain_cached.py

Weight w_occ dependence
=======================

Errors in HF+MP2 and sole MP2 energies as a function of the parameter w_occ splitting the MP2 energy between occupied
and valence molecular orbitals.

.. plot:: plots/32-dcmp2-errors-wocc_cached.py
