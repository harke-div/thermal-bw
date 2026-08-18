"""Here we define some small literature expressions used by the validation scripts. 
Currently, we only look at Celli et al (2017); this may be subject to change."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from thermal_bw import thermal_eta
from thermal_bw.constants import C_CGS, H_CGS, KEV_TO_ERG, PI, SIGMA_T


def celli2017_alpha(E_MeV, kT_keV):
    """Celli, Palladino & Vissani (2017), Eq. (8)."""
    eta = np.asarray(thermal_eta(E_MeV, kT_keV), dtype=float)
    x = 1.0 / eta
    y = 1.52 * np.power(x, 0.89)
    f_value = 3.68 * x * (-np.log(-np.expm1(-y)))

    classical_radius_sq = 3.0 * SIGMA_T / (8.0 * PI)
    hbar_c = H_CGS * C_CGS / (2.0 * PI)
    thermal_wavenumber = np.asarray(kT_keV, dtype=float) * KEV_TO_ERG / hbar_c
    return classical_radius_sq / PI * thermal_wavenumber**3 * f_value


def _psi(z):
    if z < 1.0e-6:
        return -math.log(-math.expm1(-z))
    if z < 35.0:
        return -math.log1p(-math.exp(-z))
    return math.exp(-z)


def celli2017_exact_f(x):
    """Celli, Palladino & Vissani (2017), Eq. (A.1)."""
    x = float(x)

    def integrand(beta):
        if beta <= 0.0:
            return 0.0
        one_minus_beta2 = 1.0 - beta * beta
        if one_minus_beta2 <= 0.0:
            return 0.0
        log_ratio = math.log1p(beta) - math.log1p(-beta)
        response = (2.0 * beta / one_minus_beta2**2) * (
            (3.0 - beta**4) * log_ratio
            - 2.0 * beta * (2.0 - beta * beta)
        )
        return x * x * response * _psi(x / one_minus_beta2)

    points = [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]
    if x < 1.0:
        transition = math.sqrt(max(0.0, 1.0 - x))
        points.extend(
            [max(0.0, transition - 0.05), transition, min(1.0, transition + 0.05)]
        )
    points = sorted(set(points))

    total = 0.0
    for lower, upper in zip(points[:-1], points[1:]):
        if upper <= lower:
            continue
        value, _ = quad(
            integrand,
            lower,
            upper,
            epsabs=1.0e-13,
            epsrel=3.0e-10,
            limit=500,
        )
        total += value
    return total
