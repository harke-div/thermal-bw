"""example for arbitrary isotropic spectrum"""

import numpy as np

from thermal_bw import PowerLawSpectrum, alpha_isotropic_cached, alpha_isotropic_gauss

target = PowerLawSpectrum(
    normalization=1e12,
    index=1.5,
    reference_keV=1.0,
    cutoff_keV=500.0,
    energy_bounds_keV=(0.1, 5000.0),
)

energy = np.logspace(-1, 2, 100)
fast = alpha_isotropic_cached(energy, target, preset="balanced")
check = alpha_isotropic_gauss(energy[::20], target, n_energy=160, n_angle=128)

print(f"computed {fast.size} opacity values")
print(check)
