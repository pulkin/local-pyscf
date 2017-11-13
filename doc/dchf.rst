****************************************
Divide-conquer HF by Kobayashi and Nakai
****************************************

Convergence
===========

The study of how the amplitude error decreases with iterations.

.. plot:: plots/20-dchf-convergence_cached.py

Complexity
==========

The study of how the calculation time scales with the model increase.
The expected problem complexity in conventional HF is :math:`O(N^4)` while this implementation of DC-HF should scale
as :math:`O(N^3)` where :math:`N` is the number of atoms in the system.

.. plot:: plots/21-dchf-complexity_cached.py

Errors in HF energies
======================

Total energies obtained with DC-HF compared to conventional HF.

.. plot:: plots/22-dchf-errors_cached.py

Domain size dependence
======================

Errors in HF energies as a function of the domain size and run time.
The benchmark model is a dimerized hydrogen chain of size 24.

 .. plot:: plots/23-dchf-errors-domain_cached.py
