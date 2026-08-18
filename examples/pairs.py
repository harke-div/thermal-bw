"""example for differential secondary pair production in an isotropic blackbody field"""
import numpy as np
from scipy.integrate import simpson

from thermal_bw import (
    BlackbodySpectrum,
    alpha_isotropic_gauss,
    pair_distribution,
    pair_injection,
    pair_spectrum,
)


target = BlackbodySpectrum(50.0)
gamma_energy = 10.0  # MeV

electron_energy = np.linspace(0.511, 13.5, 400)  # total energy, MeV
dalpha_dE = pair_spectrum(gamma_energy, electron_energy, target)
probability = pair_distribution(gamma_energy, electron_energy, target)

alpha = alpha_isotropic_gauss(gamma_energy, target)
print("opacity:", alpha, "cm^-1")
print("integrated pair spectrum / opacity:", simpson(dalpha_dE, x=electron_energy) / alpha)
print("integrated conditional distribution:", simpson(probability, x=electron_energy))

# Local gamma-ray number density in cm^-3 MeV^-1.
E_gamma = np.logspace(0.0, 2.0, 40)
n_gamma = 1e6 * (E_gamma / 10.0) ** -2.0
E_pair = np.logspace(np.log10(0.511), np.log10(50.0), 80)
Q_e = pair_injection(E_gamma, n_gamma, E_pair, target)
print("pair source shape:", Q_e.shape, "cm^-3 s^-1 MeV^-1")
