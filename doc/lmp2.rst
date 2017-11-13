****************************
Local MP2 by Pulay and Saebo
****************************

Convergence
===========

The study of how the amplitude error decreases with iterations.

.. plot:: plots/10-lmp2-convergence_cached.py

Complexity
==========

The study of how the calculation time scales with the model increase.
The expected problem complexity in conventional MP2 is :math:`O(N^5)` while this implementation of local MP2 should scale
as :math:`O(N^3)` where :math:`N` is the number of atoms in the system.

.. plot:: plots/11-lmp2-complexity_cached.py

Errors in MP2 energies
======================

Energies obtained with LMP2 compared to conventional MP2.

.. plot:: plots/12-lmp2-errors_cached.py
