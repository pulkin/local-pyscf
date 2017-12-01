#/usr/bin/env python
"""
This file was generated automatically.
"""
from matplotlib import pyplot
pyplot.loglog([8, 12, 16, 24, 32, 48], [0.2869398593902588, 0.5708320140838623, 1.0386121273040771, 2.4340879917144775, 4.406055927276611, 10.139120101928711], marker='o', label='DCHF pow=2.1')
pyplot.loglog([8, 12, 16, 24, 32, 48, 64], [0.26192593574523926, 0.3420219421386719, 0.706855058670044, 1.3042271137237549, 2.5092599391937256, 9.706380844116211, 30.801491022109985], marker='x', label='pyscf HF pow=3.9')
pyplot.loglog([8, 12, 16, 24, 32], [0.2854800224304199, 0.5460660457611084, 1.255007028579712, 4.652163028717041, 13.339195013046265], marker='+', label='custom HF pow=3.6')
pyplot.xlabel('Model size')
pyplot.ylabel('Run time (s)')
pyplot.legend()
pyplot.show()

