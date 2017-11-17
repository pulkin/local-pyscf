#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.figure(figsize=(8, 4.8))
pyplot.semilogy([0, 1, 2, 3, 4], [1.0954200509634877e-13, 2.1260164295711093e-08, 0.0016855996096243331, 3.4306672231476654e-05, 1.4123599610726956e-05], marker='_', mew=5, markersize=30, ls='None', label='HF')
pyplot.semilogy([0, 1, 2, 3, 4], [7.5690145216282076e-08, 4.9615592382946478e-06, 0.00045329587334149341, 8.0409401501437463e-06, 4.1220559728651534e-06], marker='_', mew=5, markersize=30, ls='None', label='MP2')
pyplot.semilogy([0, 1, 2, 3, 4], [1.2044411410588873e-06, 6.3763168945985923e-05, 0.00043706511522751773, 0.00015052856107538684, 7.5628387378072939e-05], marker='_', mew=5, markersize=30, ls='None', label='CCSD')
pyplot.xticks([0, 1, 2, 3, 4], ['He chain N=6', 'H chain N=12 d=6.0', 'H dim chain N=12', '... 2(2) config', '... 4(2) config'], rotation=15)
pyplot.ylabel('Energy difference per atom (Ry)')
pyplot.grid(axis='y')
pyplot.legend()
pyplot.show()

