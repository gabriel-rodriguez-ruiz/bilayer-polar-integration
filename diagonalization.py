#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 09:29:41 2025

@author: gabriel
"""

import numpy as np
import scipy
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y, alpha_0, alpha_z

def get_Hamiltonian_in_polars(k, theta, mu_s, mu_S, B_x_s, B_y_s,
                              B_x_S, B_y_S, Delta, phi_x, phi_y, q_x, q_y,
                              q_B_x, q_B_y,
                              gamma_s, gamma_S, Lambda):
    """Return the Hamiltonian for a given k."""
    k_x = k * np.cos(theta)
    k_y = k * np.sin(theta)
    chi_k_s = lambda k_x, k_y: gamma_s * (k_x**2 + k_y**2) - mu_s
    chi_k_S = lambda k_x, k_y: gamma_S * (k_x**2 + k_y**2) - mu_S
    H_S = ( chi_k_S(k_x + phi_x + q_x, k_y + phi_y + q_y ) * np.kron(( tau_0 + tau_z )/2, sigma_0)
            - chi_k_S(-k_x + phi_x + q_x, -k_y + phi_y + q_y) * np.kron( ( tau_0 - tau_z )/2, sigma_0)
            - B_y_S * np.kron(tau_0, sigma_y)
            - B_x_S * np.kron(tau_0, sigma_x)
            - Delta * np.kron(tau_x, sigma_0) 
            )
    H_s = ( chi_k_s(k_x + phi_x + q_x + q_B_x, k_y + phi_y + q_y + q_B_y) * np.kron(( tau_z + tau_0 )/2, sigma_0)
            - chi_k_s(-k_x + phi_x + q_x + q_B_x, -k_y + phi_y + q_y + q_B_y) * np.kron( (tau_z - tau_0 )/2, sigma_0)
            - B_y_s * np.kron(tau_0, sigma_y)
            - B_x_s * np.kron(tau_0, sigma_x)
            + Lambda * (k_x + phi_x + q_x + q_B_x) * np.kron( ( tau_z + tau_0 )/2, sigma_y )  #to be checked
            - Lambda * (-k_x + phi_x + q_x + q_B_x) * np.kron( (tau_z - tau_0 )/2, sigma_y )
            - Lambda * (k_y + phi_y + q_y + q_B_y) * np.kron( ( tau_z + tau_0 )/2, sigma_x )
            + Lambda * (-k_y + phi_y + q_y + q_B_y) * np.kron( ( tau_z - tau_0 )/2, sigma_x )
           )
    return 1/2 * ( np.kron((alpha_0 + alpha_z)/2, H_S)
                   + np.kron((alpha_0 - alpha_z)/2, H_s)
                   )

def get_Energies_in_polars(k_values, theta_values, mu_s, mu_S, B_x_s,
                           B_y_s, B_x_S, B_y_S, Delta,
                           phi_x, phi_y, q_x, q_y,
                           q_B_x, q_B_y,
                           gamma_s, gamma_S, Lambda):
    """Return the energies of the Hamiltonian at a given k."""
    E = np.zeros((len(k_values), len(theta_values), 4))
    for i, k in enumerate(k_values):
        for j, theta in enumerate(theta_values):
            H = get_Hamiltonian_in_polars(k, theta, mu_s, mu_S, B_x_s, B_y_s,
                                          B_x_S, B_y_S, Delta, phi_x, phi_y, q_x, q_y,
                                          q_B_x, q_B_y,
                                          gamma_s, gamma_S, Lambda)
            E[i, j, :] = np.linalg.eigvalsh(H)
    return E
