"""Breit-Wheeler cross sections.

The dimensionless invariant used here is

    s = (E / m_e c^2) (epsilon / m_e c^2) (1 - cos psi),

thus the vacuum pair-production threshold is ``s >= 2``.
"""
from __future__ import annotations

import numpy as np

from .constants import SIGMA_T


def sigma_breit_wheeler_s(s):
    """Return the unpolarized Breit--Wheeler cross section in cm^2.

    Parameters
    ----------
    s : float or array-like
        Dimensionless invariant in the thermal-bw convention. The cross section
        dissapears for ``s <= 2``.
    """
    arr = np.asarray(s, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    mask = np.isfinite(arr) & (arr > 2.0)
    if np.any(mask):
        beta = np.sqrt(1.0 - 2.0 / arr[mask])
        beta = np.clip(beta, 0.0, 1.0 - 1e-15)
        beta2 = beta * beta
        logterm = np.log((1.0 + beta) / (1.0 - beta))
        bracket = (3.0 - beta**4) * logterm - 2.0 * beta * (2.0 - beta2)
        out[mask] = (3.0 / 16.0) * SIGMA_T * (1.0 - beta2) * bracket
    if out.ndim == 0:
        return float(out)
    return out


def sigma_breit_wheeler_beta(beta):
    """Return the unpolarized cross section from the CM-frame speed ``beta``."""
    arr = np.asarray(beta, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    mask = np.isfinite(arr) & (arr > 0.0) & (arr < 1.0)
    if np.any(mask):
        b = arr[mask]
        b2 = b * b
        logterm = np.log((1.0 + b) / (1.0 - b))
        bracket = (3.0 - b**4) * logterm - 2.0 * b * (2.0 - b2)
        out[mask] = (3.0 / 16.0) * SIGMA_T * (1.0 - b2) * bracket
    if out.ndim == 0:
        return float(out)
    return out
