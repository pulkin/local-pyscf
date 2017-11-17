#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.semilogy([0, 1, 2, 3], [0.032041692342924089, 0.00030204478277512381, 8.1485804052428001e-05, 5.2935975003443048e-07], marker='o', label='w_occ=1.0')
pyplot.semilogy([0, 1, 2, 3], [0.011919421037235645, 0.00012779413589841804, 7.3518206629152871e-06, 4.1259344485400362e-07], marker='x', ls='--', label='w_occ=1.0 (e2 error only)')
pyplot.semilogy([0, 1, 2, 3], [0.032041692342924089, 0.00029079577432877812, 7.7788439763859518e-05, 4.2738727862312587e-07], marker='o', label='w_occ=0.5')
pyplot.semilogy([0, 1, 2, 3], [0.011919421037235645, 0.00013904314434476372, 3.6544563743468039e-06, 3.1062097344269901e-07], marker='x', ls='--', label='w_occ=0.5 (e2 error only)')
pyplot.semilogy([0, 1, 2, 3], [0.032041692342924089, 0.00027954676588237692, 7.4091075475235524e-05, 3.2541480721182126e-07], marker='o', label='w_occ=0.0')
pyplot.semilogy([0, 1, 2, 3], [0.011919421037235645, 0.00015029215279116492, 4.2907914277190429e-08, 2.086485020313944e-07], marker='x', ls='--', label='w_occ=0.0 (e2 error only)')
pyplot.xticks([0, 1, 2, 3], ['2(0)', '4(2)', '6(4)', '12(6)'])
pyplot.ylabel('Absolute error (Ry)')
pyplot.xlabel('Domain size (buffer size)')
pyplot.legend()
pyplot.show()

