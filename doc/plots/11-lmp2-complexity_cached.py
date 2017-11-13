#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.loglog([8, 12, 16, 24, 32, 48, 64], [0.008569002151489258, 0.03431415557861328, 0.13283991813659668, 0.9201431274414062, 3.735200881958008, 27.873284816741943, 1083.2681589126587], marker='x', label='He chains (conventional MP2) s=nan')
pyplot.loglog([8, 12, 16, 24, 32, 48], [0.6403460502624512, 1.8115911483764648, 4.302359104156494, 15.49633002281189, 41.71060109138489, 184.52525806427002], marker='o', label='He chains (LMP2) s=3.6')
pyplot.loglog([8, 12, 16, 24, 32, 48, 64], [0.005057096481323242, 0.019192934036254883, 0.07035684585571289, 0.48464083671569824, 1.9332208633422852, 14.653261184692383, 447.15065002441406], marker='x', label='H chains (conventional MP2) s=nan')
pyplot.loglog([8, 12, 16, 24], [4.286357879638672, 17.896310806274414, 48.510149002075195, 197.94836497306824], marker='o', label='H chains (LMP2) s=3.5')
pyplot.xlabel('Model size')
pyplot.ylabel('Run time (s)')
pyplot.grid()
pyplot.legend()
pyplot.show()

