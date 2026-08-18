"""Here I present a reference calculation for blackbody via Gauss-Legendre quadrature."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .constants import C_CGS, H_CGS, KEV_TO_ERG, ME_C2_KEV, PI
from .cross_sections import sigma_breit_wheeler_s
from .exceptions import InputValidationError
from .units import positive_values


def _planck_factor(u):
    """Return u^2 / (exp(u) - 1) without overflow in the Wien tail."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    moderate = u < 50.0
    out[moderate] = u[moderate] ** 2 / np.expm1(u[moderate])
    tail = ~moderate
    if np.any(tail):
        exp_minus_u = np.exp(-u[tail])
        out[tail] = u[tail] ** 2 * exp_minus_u / (1.0 - exp_minus_u)
    return out


def _angle_kernel_threshold_aware(z, nodes, weights):
    """Return 0.5 integral t sigma(z t) dt over the allowed interval."""
    z = np.asarray(z, dtype=float)
    result = np.zeros_like(z)
    active = z > 1.0
    if np.any(active):
        za = z[active]
        t_min = 2.0 / za
        half_width = 0.5 * (2.0 - t_min)
        midpoint = 0.5 * (2.0 + t_min)
        t = half_width[:, None] * nodes[None, :] + midpoint[:, None]
        s = za[:, None] * t
        result[active] = 0.5 * half_width * np.sum(
            weights[None, :] * t * sigma_breit_wheeler_s(s), axis=1
        )
    return result


def alpha_blackbody_gauss(
    E_MeV,
    kT_keV,
    *,
    n_angle: int = 64,
    n_planck: int = 128,
    u_base_max: float = 80.0,
    tail_width: float = 80.0,
):
    """Fixed Gauss--Legendre blackbody reference with threshold-aware limits."""
    E = positive_values(E_MeV, "MeV", "E_MeV must be finite and positive")
    T = positive_values(kT_keV, "keV", "kT_keV must be finite and positive")
    E, T = np.broadcast_arrays(E, T)
    if int(n_angle) != n_angle or n_angle < 4:
        raise InputValidationError("ERROR: n_angle must be an integer >= 4")
    if int(n_planck) != n_planck or n_planck < 4:
        raise InputValidationError("ERROR: n_planck must be an integer >= 4")
    if u_base_max <= 0.0 or tail_width <= 0.0:
        raise InputValidationError("ERROR: u_base_max and tail_width must be positive")

    angle_nodes, angle_weights = leggauss(int(n_angle))
    u_nodes, u_weights = leggauss(int(n_planck))
    prefactor_density = (8.0 * PI) / (H_CGS**3 * C_CGS**3)

    out = np.zeros(E.size, dtype=float)
    for idx, (energy, temp) in enumerate(zip(E.reshape(-1), T.reshape(-1))):
        eta = (energy * 1.0e3 / ME_C2_KEV) * (temp / ME_C2_KEV)
        u_threshold = 1.0 / eta
        u_upper = max(float(u_base_max), u_threshold + float(tail_width))
        half_width = 0.5 * (u_upper - u_threshold)
        midpoint = 0.5 * (u_upper + u_threshold)
        u = half_width * u_nodes + midpoint
        w_u = half_width * u_weights
        kernel = _angle_kernel_threshold_aware(eta * u, angle_nodes, angle_weights)
        kernel_integral = np.sum(w_u * _planck_factor(u) * kernel)
        kT_erg = temp * KEV_TO_ERG
        out[idx] = prefactor_density * kT_erg**3 * kernel_integral

    out = out.reshape(E.shape)
    if out.ndim == 0:
        return float(out)
    return out
