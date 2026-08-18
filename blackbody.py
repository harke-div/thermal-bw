"""example for blackbody opacity"""

import numpy as np

from thermal_bw import alpha_blackbody_gauss, alpha_exact, alpha_fit

temperature = 50.0
eta = np.geomspace(0.01, 50.0, 100)
energy = 261.121 * eta / temperature

fast = alpha_fit(energy, temperature)
gauss = alpha_blackbody_gauss(energy, temperature)
reference = alpha_exact(2.0, temperature, epsrel=1e-8)

print(f"alpha_fit(2 MeV, 50 keV) = {alpha_fit(2.0, temperature):.12e} cm^-1")
print(f"alpha_exact(2 MeV, 50 keV) = {reference:.12e} cm^-1")
relative = np.abs(fast - gauss) / gauss
print(f"array shape = {fast.shape}; max Gauss difference = {np.max(relative):.3e}")
