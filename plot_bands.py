#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:01:58 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from diagonalization import get_Energies_in_polars

c = 3e18 # nm/s  #3e9 # m/s
m_e =  5.1e8 / c**2 # meV s²/m²
m = 0.403 * m_e # meV s²/m²
hbar = 6.58e-13 # meV s
gamma_S = hbar**2 / (2*m) # meV (nm)²
gamma_s = 5 * gamma_S

E_F = 50.6 # meV
k_F_s = np.sqrt(E_F / gamma_s ) # 1/nm
k_F_S = np.sqrt(E_F / gamma_S ) # 1/nm
Delta = 0.2   #  meV
mu = 50.6   # 623 Delta #50.6  #  meV
Lambda = 8 * Delta # 8 * Delta  #0.644 meV 
B = 1 * Delta

k_x_values = np.linspace(-np.pi, np.pi)
k_y = 0

