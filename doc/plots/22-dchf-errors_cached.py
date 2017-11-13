#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.figure(figsize=(8, 4.8))
pyplot.semilogy([0, 1, 2, 3, 4], [4.2040445199139263e-14, 2.1260164739800302e-08, 0.0016855996096232968, 3.4306672232660894e-05, 1.5154790137176613e-05], marker='_', mew=5, markersize=30, ls='None')
pyplot.xticks([0, 1, 2, 3, 4], ['He chain N=6', 'H chain N=12 d=6.0', 'H dim chain N=12', '+ buffer', '+ larger domain size'])
pyplot.ylabel('Energy difference per atom (Ry)')
pyplot.grid(axis='y')
pyplot.show()

