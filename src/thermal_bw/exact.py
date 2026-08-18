"""Here I code adaptive reference quadrature for isotropic thermal Breit-Wheeler opacity."""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from .constants import C_CGS, H_CGS, KEV_TO_ERG, ME_C2_KEV, PI
from .cross_sections import sigma_breit_wheeler_beta, sigma_breit_wheeler_s
from .exceptions import InputValidationError
from .units import as_value, positive_values


def sigma_breit_wheeler(beta: float) -> float:
    """Backward-compatible scalar wrapper for the Breit--Wheeler cross section."""
    return float(sigma_breit_wheeler_beta(beta))


def blackbody_photon_density_per_energy(eps_erg: float, kT_erg: float) -> float:
    """Blackbody photon number density per unit energy in cm^-3 erg^-1."""
    if eps_erg <= 0.0 or kT_erg <= 0.0:
        return 0.0
    x = eps_erg / kT_erg
    if x < 50.0:
        occupation = 1.0 / math.expm1(x)
    else:
        exp_minus_x = math.exp(-x)
        occupation = exp_minus_x / (1.0 - exp_minus_x)
    prefactor = (8.0 * PI) / (H_CGS**3 * C_CGS**3)
    return prefactor * eps_erg**2 * occupation


def _planck_factor_scalar(u: float) -> float:
    """Stable scalar u^2/(exp(u)-1)."""
    if u < 50.0:
        return u * u / math.expm1(u)
    exp_minus_u = math.exp(-u)
    return u * u * exp_minus_u / (1.0 - exp_minus_u)


def alpha_exact_keV(
    E_keV: float,
    kT_keV: float,
    *,
    u_base_max: float = 80.0,
    tail_width: float = 80.0,
    epsabs: float = 0.0,
    epsrel: float = 3e-4,
    angle_epsrel: float | None = None,
) -> float:
    """Adaptive blackbody reference with a threshold-aware Planck tail."""
    E_keV = float(positive_values(E_keV, "keV", "E_keV must be finite and positive"))
    kT_keV = float(positive_values(kT_keV, "keV", "kT_keV must be finite and positive"))
    if u_base_max <= 0.0 or tail_width <= 0.0:
        raise InputValidationError("ERROR: u_base_max and tail_width must be positive")
    if epsrel <= 0.0 or (angle_epsrel is not None and angle_epsrel <= 0.0):
        raise InputValidationError("ERROR: relative tolerances must be positive")

    eta = (E_keV / ME_C2_KEV) * (kT_keV / ME_C2_KEV)
    u_threshold = 1.0 / eta
    u_upper = max(float(u_base_max), u_threshold + float(tail_width))
    angle_tol = epsrel if angle_epsrel is None else float(angle_epsrel)
    prefactor = (8.0 * PI) / (H_CGS**3 * C_CGS**3) * (kT_keV * KEV_TO_ERG) ** 3

    def integrand_u(u: float) -> float:
        z = eta * u
        if z <= 1.0:
            return 0.0
        t_min = 2.0 / z

        def integrand_t(t: float) -> float:
            return 0.5 * t * float(sigma_breit_wheeler_s(z * t))

        angular, _ = integrate.quad(
            integrand_t,
            t_min,
            2.0,
            epsabs=epsabs,
            epsrel=angle_tol,
            limit=200,
            points=[t_min],
        )
        return _planck_factor_scalar(u) * angular

    # Resolve the threshold and Wien tail.
    candidates = [
        u_threshold,
        u_threshold + 0.25,
        u_threshold + 1.0,
        u_threshold + 4.0,
        u_threshold + 12.0,
        u_threshold + 32.0,
        u_upper,
    ]
    breakpoints = sorted({min(max(v, u_threshold), u_upper) for v in candidates})
    total = 0.0
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:]):
        if hi <= lo:
            continue
        piece, _ = integrate.quad(
            integrand_u,
            lo,
            hi,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=300,
            points=[lo],
        )
        total += piece
    return float(prefactor * total)


def alpha_exact(E_MeV: float, kT_keV: float, **kwargs) -> float:
    """Tail-safe adaptive reference for ``E`` in MeV and ``kT`` in keV."""
    return alpha_exact_keV(
        float(as_value(E_MeV, "MeV")) * 1.0e3,
        float(as_value(kT_keV, "keV")),
        **kwargs,
    )


def alpha_grid(E_MeV, kT_keV, **kwargs) -> np.ndarray:
    """Compute adaptive reference values on a grid of shape ``(len(kT), len(E))``."""
    E_MeV = as_value(E_MeV, "MeV")
    kT_keV = as_value(kT_keV, "keV")
    out = np.zeros((len(kT_keV), len(E_MeV)), dtype=float)
    for i, temp in enumerate(kT_keV):
        for j, energy in enumerate(E_MeV):
            out[i, j] = alpha_exact(float(energy), float(temp), **kwargs)
    return out
