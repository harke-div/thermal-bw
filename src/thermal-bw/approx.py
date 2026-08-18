"""Here I present an analytic surrogate for lightweight calculations of isotropic blackbody opacity."""
from __future__ import annotations

from typing import Iterable, Literal
import warnings

import numpy as np

from .constants import ME_C2_KEV
from .exceptions import InputValidationError, OutOfDomainError, OutOfDomainWarning
from .units import as_value, positive_values


# Fit vector: log(A), log(c), log(d), p, a, q.  The final exponent is fixed at -5.
DEFAULT_PARAMS = np.array(
    [
        -4.629432082373661,
        -1.5054575455548949,
        -7.476232398309175,
        0.6801779422917057,
        0.9870557618730673,
        1.002079526450841,
    ],
    dtype=float,
)
SURROGATE_ETA_DOMAIN = (1.0e-2, 50.0)
_DEFAULT_COEFFS = (
    float(np.exp(DEFAULT_PARAMS[0])),
    float(np.exp(DEFAULT_PARAMS[1])),
    float(np.exp(DEFAULT_PARAMS[2])),
    float(DEFAULT_PARAMS[3]),
    float(DEFAULT_PARAMS[4]),
    float(DEFAULT_PARAMS[5]),
)


def _parameters(params: Iterable[float]):
    if params is DEFAULT_PARAMS:
        return _DEFAULT_COEFFS
    values = np.asarray(list(params), dtype=float)
    if values.shape != (6,) or np.any(~np.isfinite(values)):
        raise InputValidationError("ERROR: params must contain six finite values")
    logA, logc, logd, p, a, q = values
    return np.exp(logA), np.exp(logc), np.exp(logd), p, a, q


def _eta(E_MeV, kT_keV):
    return (E_MeV * 1.0e3 / ME_C2_KEV) * (kT_keV / ME_C2_KEV)


def thermal_eta(E_MeV, kT_keV):
    """Return eta = E_gamma kT / (m_e c^2)^2."""
    E = positive_values(E_MeV, "MeV", "E_MeV must contain finite positive values")
    T = positive_values(kT_keV, "keV", "kT_keV must contain finite positive values")
    E, T = np.broadcast_arrays(E, T)
    value = _eta(E, T)
    return float(value) if value.ndim == 0 else value


def _reduced_surrogate(eta, params):
    A, c, d, p, a, q = _parameters(params)
    eta_p = eta**p
    one_plus = 1.0 + d * eta
    one_plus_5 = one_plus * one_plus * one_plus * one_plus * one_plus
    suppression = np.exp(-((a / eta) ** q))
    return A * suppression * (eta_p / eta) / (1.0 + c * eta_p / one_plus_5)


def _evaluate(E_keV, T_keV, params):
    eta = np.maximum((E_keV / ME_C2_KEV) * (T_keV / ME_C2_KEV), 1e-300)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = T_keV**3 * _reduced_surrogate(eta, params)
    return np.where(np.isfinite(value) & (value > 0.0), value, 0.0)


def alpha_model(E_keV, kT_keV, params: Iterable[float] = DEFAULT_PARAMS):
    """Evaluate the fitted expression with both energies in keV."""
    E = positive_values(E_keV, "keV", "E_keV must contain finite positive values")
    T = positive_values(kT_keV, "keV", "kT_keV must contain finite positive values")
    E, T = np.broadcast_arrays(E, T)
    value = _evaluate(E, T, params)
    return float(value) if value.ndim == 0 else value


def alpha_fit(
    E_MeV,
    kT_keV,
    params: Iterable[float] = DEFAULT_PARAMS,
    *,
    bounds: Literal["warn", "raise", "ignore"] = "warn",
):
    """Evaluate the blackbody opacity surrogate in cm^-1."""
    E = positive_values(E_MeV, "MeV", "E_MeV must contain finite positive values")
    T = positive_values(kT_keV, "keV", "kT_keV must contain finite positive values")
    E, T = np.broadcast_arrays(E, T)
    if bounds not in {"warn", "raise", "ignore"}:
        raise InputValidationError("ERROR: bounds must be 'warn', 'raise', or 'ignore'")

    eta = _eta(E, T)
    if bounds != "ignore":
        lower, upper = SURROGATE_ETA_DOMAIN
        inside = (eta >= lower * (1.0 - 1e-12)) & (eta <= upper * (1.0 + 1e-12))
        if not np.all(inside):
            message = (
                "blackbody surrogate evaluated outside its validated dimensionless range "
                f"{lower:g} <= eta <= {upper:g}; use alpha_blackbody_gauss or "
                "alpha_exact when controlled accuracy is required"
            )
            if bounds == "raise":
                raise OutOfDomainError("ERROR: " + message)
            warnings.warn("WARNING: " + message, OutOfDomainWarning, stacklevel=2)

    value = _evaluate(E * 1.0e3, T, params)
    return float(value) if value.ndim == 0 else value


def within_validated_domain(E_MeV, kT_keV):
    """Return whether eta lies inside the validated surrogate interval."""
    eta = np.asarray(thermal_eta(E_MeV, kT_keV), dtype=float)
    lower, upper = SURROGATE_ETA_DOMAIN
    return (eta >= lower * (1.0 - 1e-12)) & (eta <= upper * (1.0 + 1e-12))
