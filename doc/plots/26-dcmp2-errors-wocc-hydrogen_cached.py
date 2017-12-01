#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.subplot(121)
pyplot.semilogy([0, 1, 2], [0.011919421037239586, 0.00012816627140188608, 5.2245612049972578e-06], marker="x", label="w_occ=1.0")
pyplot.semilogy([0, 1, 2], [0.011919421037239586, 0.00013941769967856787, 2.6872015732593013e-06], marker="x", label="w_occ=0.5")
pyplot.semilogy([0, 1, 2], [0.011919421037239586, 0.00015066912795536069, 1.4984194163236708e-07], marker="x", label="w_occ=0.0")
pyplot.xticks([0, 1, 2], ["2(0)", "4(2)", "8(4)"])
pyplot.ylabel("Absolute error (Ry)")
pyplot.xlabel("Domain size (buffer size)")
pyplot.legend()
pyplot.title("DC-MP2")
pyplot.subplot(122)
pyplot.semilogy([0, 1, 2], [0.016085699630404071, 0.0024415554799844807, 0.00035903972955775298], marker="x", label="w_occ=1.0")
pyplot.semilogy([0, 1, 2], [0.016085699740406745, 0.0048230753028825069, 0.00079594401443028318], marker="x", label="w_occ=0.5")
pyplot.semilogy([0, 1, 2], [0.016085699733860648, 0.0072045951258222773, 0.0012328149770984842], marker="x", label="w_occ=0.0")
pyplot.xticks([0, 1, 2], ["2(0)", "4(2)", "8(4)"])
pyplot.ylabel("Absolute error (Ry)")
pyplot.xlabel("Domain size (buffer size)")
pyplot.legend()
pyplot.title("DC-CCSD")
pyplot.show()

