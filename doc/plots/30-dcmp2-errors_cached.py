#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.figure(figsize=(8, 4.8))
pyplot.semilogy([0, 1, 2, 3, 4], [3.9671969413272258e-14, 2.1260165183889512e-08, 0.0016855996096228527, 3.430667223206877e-05, 1.5154790137028584e-05], marker='_', mew=5, markersize=30, ls='None', label='HF')
pyplot.semilogy([0, 1, 2, 3, 4], [7.5690145345808091e-08, 4.9615592308700314e-06, 0.00045329587334150728, 8.0409401501229296e-06, 4.1353468078139666e-06], marker='_', mew=5, markersize=30, ls='None', label='MP2')
pyplot.xticks([0, 1, 2, 3, 4], ['He chain N=6', 'H chain N=12 d=6.0', 'H dim chain N=12', '+ buffer', '+ larger domain size'])
pyplot.ylabel('Energy difference per atom (Ry)')
pyplot.grid(axis='y')
pyplot.legend()
pyplot.show()

