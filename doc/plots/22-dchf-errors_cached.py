#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.figure(figsize=(16, 4.8))
pyplot.subplot(121)
pyplot.semilogy([0, 1, 2, 3, 4, 5, 6], [1.0835776720341528e-13, 2.126016444374083e-08, 0.001685599609623889, 1.4123599610874985e-05, 0.011493718776240947, 3.4306672231032564e-05, 1.4123599610578927e-05], marker='_', mew=5, markersize=30, ls='None')
pyplot.xticks([0, 1, 2, 3, 4, 5, 6], ['He chain N=6', 'H chain N=12 d=6.0', 'H dim chain N=12', '... 3(0) config', '... 2(1) config', '... 2(2) config', '... 4(2) config'], rotation=15)
pyplot.ylabel('Energy difference per atom (Ry)')
pyplot.grid(axis='y')
pyplot.subplot(122)
pyplot.semilogy([0, 1, 2, 3, 4, 5, 6], [1.7481654457363049e-08, 0.00027446799566078865, 0.023748974905231514, 0.0027834788210318526, 0.23184446461122366, 0.005309024815691385, 0.0027834788210312073], marker='_', mew=5, markersize=30, ls='None')
pyplot.xticks([0, 1, 2, 3, 4, 5, 6], ['He chain N=6', 'H chain N=12 d=6.0', 'H dim chain N=12', '... 3(0) config', '... 2(1) config', '... 2(2) config', '... 4(2) config'], rotation=15)
pyplot.ylabel('Maximal density matrix deviation')
pyplot.grid(axis='y')
pyplot.show()

