#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.semilogy([0, 1, 2, 3], [0.032041692342920758, 0.00033357599357064505, 8.1660527988247633e-05, 5.2568559527621517e-07], marker='o', label='w_occ=1.0')
pyplot.semilogy([0, 1, 2, 3], [0.011919421037234756, 0.00012808777945583216, 7.3353934356146233e-06, 4.1279459306320376e-07], marker='x', ls='--', label='w_occ=1.0 (e2 error only)')
pyplot.semilogy([0, 1, 2, 3], [0.032041692342920758, 0.0003220229347245529, 7.7954622693499065e-05, 4.2372575659310741e-07], marker='o', label='w_occ=0.5')
pyplot.semilogy([0, 1, 2, 3], [0.011919421037234756, 0.00013964083830192431, 3.6294881408660551e-06, 3.10834754380096e-07], marker='x', ls='--', label='w_occ=0.5 (e2 error only)')
pyplot.semilogy([0, 1, 2, 3], [0.032041692342920758, 0.00031046987587846075, 7.424871739891703e-05, 3.2176591768795504e-07], marker='o', label='w_occ=0.0')
pyplot.semilogy([0, 1, 2, 3], [0.011919421037234756, 0.00015119389714801645, 7.641715371597968e-08, 2.0887491547494363e-07], marker='x', ls='--', label='w_occ=0.0 (e2 error only)')
pyplot.xticks([0, 1, 2, 3], ['2(0)', '4(2)', '6(4)', '12(6)'])
pyplot.ylabel('Absolute error (Ry)')
pyplot.xlabel('Domain size (buffer size)')
pyplot.legend()
pyplot.show()

