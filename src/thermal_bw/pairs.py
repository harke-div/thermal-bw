"""Differential electron--positron production in isotropic photon fields.

Notes on implementation and references
--------------------------------------- 
 Please do note that the analytical differential kernel below implements the published result of
 Böttcher & Schlickeiser (1997), especially Eqs. (24)--(29). The same Böttcher--Schlickeiser formalism is used in SOPRANO (Gasparyan, Bégué & Sahakyan 2022, 
 Appendix A4). I have coded the direct centre-of-momentum calculation to be used at poorly conditioned points and to evaluate the same Breit--Wheeler interaction 
 through the CM cross section and lab/CM kinematics in Böttcher & Schlickeiser, Eqs. (2), (5)--(7), and (11)."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from .constants import C_CGS, ME_C2_KEV, SIGMA_T
from .exceptions import InputValidationError
from .isotropic import (
    _boundaries,
    _integration_bounds_for_energy,
    _has_log_span,
    _legendre_nodes,
    _orders_for_segments,
    alpha_isotropic_gauss,
)
from .targets import (
    BlackbodySpectrum,
    CompositeSpectrum,
    GreybodySpectrum,
    PhotonSpectrum,
    discrete_lines,
    integration_breakpoints,
)
from .units import as_value, positive_values

_ME_C2_MEV = ME_C2_KEV / 1.0e3


def _i_term(alpha: float, product: float, c_value: float) -> float:
    """
Notes on implementation and references
--------------------------------------    
    The published logarithmic primitive for c > 0 differs from this form by
     a constant that is independent of alpha, which should cancel between integration limits.
     In Böttcher & Schlickeiser (1997), Eq. (28), the c > 0 branch is
    
       log(alpha*sqrt(c) + sqrt(P + c*alpha**2)) / sqrt(c),  P = x1*x2.
    
    Since asinh(y) = log(y + sqrt(1+y**2)), the expression used below is the
    same primitive minus log(P)/(2*sqrt(c)). That term has no alpha
    dependence, so it cancels exactly between the two limits in Eq. (26)."""
    if c_value > 0.0:
        root_c = math.sqrt(c_value)
        return math.asinh(alpha * math.sqrt(c_value / product)) / root_c
    """
    Notes on implementation and references
    --------------------------------------
    The c < 0 arcsin branch is the form given directly in Eq. (28). The
    clipping only protects the endpoint against roundoff beyond |arg| = 1."""
    root_minus_c = math.sqrt(-c_value)
    arg = alpha * math.sqrt((-c_value) / product)
    return math.asin(min(1.0, max(-1.0, arg))) / root_minus_c


def _h_term(
    alpha: float,
    x1: float,
    x2: float,
    gamma_e: float,
    branch: int,
) -> float:
    product = x1 * x2
    if branch > 0:
        c_value = (x1 - gamma_e) ** 2 - 1.0
        d_value = x1 * x1 + product + gamma_e * (x2 - x1)
    else:
        c_value = (x2 - gamma_e) ** 2 - 1.0
        d_value = x2 * x2 + product - gamma_e * (x2 - x1)

    if abs(c_value) < 1.0e-8:
        """ 
        Notes on implementation and references
        --------------------------------------
        Finite c -> 0 limit from Böttcher & Schlickeiser (1997), Eq. (29).
        Also, using the stated limit avoids evaluating the cancelling 1/c
        terms of Eq. (27) directly near c = 0."""
        return (
            (alpha**3 / 12.0 - alpha * d_value / 8.0) / product**1.5
            + (alpha**3 / 6.0 + alpha / 2.0 + 1.0 / (4.0 * alpha))
            / math.sqrt(product)
        )

    radicand = product + c_value * alpha * alpha
    scale = max(product, abs(c_value * alpha * alpha), 1.0)
    if radicand < 0.0 and abs(radicand) <= 5.0e-14 * scale:
        radicand = 0.0
    if radicand <= 0.0:
        return math.nan

    root = math.sqrt(radicand)
    i_value = _i_term(alpha, product, c_value)
    return (
        -alpha / (8.0 * root) * (d_value / product + 2.0 / c_value)
        + 0.25 * (2.0 - (product - 1.0) / c_value) * i_value
        + root / 4.0 * (alpha / c_value + 1.0 / (alpha * product))
    )


def _pair_alpha_bounds(x1: float, x2: float, gamma_e: float) -> tuple[float, float]:
    """Return the allowed CM lepton-energy interval for a fixed lab energy."""
    product = x1 * x2
    energy = x1 + x2
    y_value = gamma_e * (energy - gamma_e) + 1.0
    discriminant = (y_value - energy) * (y_value + energy)
    if discriminant <= 0.0:
        return 0.0, 0.0

    root_disc = math.sqrt(discriminant)
    alpha_a = math.sqrt(0.5 * (y_value + root_disc))
    # alpha_a * alpha_b = (x1 + x2) / 2; this avoids subtracting
    # nearly equal quantities in the lower root.
    # This identity follows directly from the two roots in Böttcher &
    # Schlickeiser (1997), Eq. (25), so it changes only how the smaller root
    # is evaluated numerically.
    alpha_b = energy / (2.0 * alpha_a)
    return max(1.0, alpha_b), min(math.sqrt(product), alpha_a)


def _pair_dsigma_analytic_scalar(
    x1: float,
    x2: float,
    gamma_e: float,
) -> tuple[float, float]:
    """Evaluate the reduced Böttcher--Schlickeiser kernel and its conditioning."""
    product = x1 * x2
    energy = x1 + x2
    if x1 <= 0.0 or x2 <= 0.0 or gamma_e < 1.0 or product <= 1.0:
        return 0.0, 1.0

    gamma_lo, gamma_hi = pair_gamma_bounds(x1, x2)
    if gamma_e <= gamma_lo or gamma_e >= gamma_hi:
        return 0.0, 1.0

    alpha_lo, alpha_hi = _pair_alpha_bounds(x1, x2, gamma_e)
    if alpha_hi <= alpha_lo:
        return 0.0, 1.0

    def primitive(alpha: float) -> float:
        return (
            math.sqrt(max(0.0, energy * energy - 4.0 * alpha * alpha)) / 4.0
            + _h_term(alpha, x1, x2, gamma_e, 1)
            + _h_term(alpha, x1, x2, gamma_e, -1)
        )

    upper = primitive(alpha_hi)
    lower = primitive(alpha_lo)
    if not (math.isfinite(upper) and math.isfinite(lower)):
        return math.nan, 0.0

    difference = upper - lower
    scale = abs(upper) + abs(lower)
    conditioning = abs(difference) / scale if scale > 0.0 else 1.0

    # Böttcher & Schlickeiser write the symmetric photon-population source with
    # a 1/2 collision-counting factor. Here x1 is a distinguished test photon,
    # so the one-lepton marginal is twice that population kernel.
    return 1.5 * SIGMA_T / (product * product) * difference, conditioning


def _cm_dsigma_domega(cos_theta, alpha):
    """Unpolarized CM differential Breit--Wheeler cross section."""
    # This is the standard CM differential cross section used in Böttcher &
    # Schlickeiser (1997), Eq. (11), rather than a separate pair-production model.
    cos_theta = np.asarray(cos_theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta2 = np.maximum(0.0, 1.0 - 1.0 / (alpha * alpha))
    beta = np.sqrt(beta2)
    sin2 = 1.0 - cos_theta * cos_theta
    numerator = 1.0 + 2.0 * beta2 * sin2 - beta2 * beta2 * (1.0 + sin2 * sin2)
    denominator = (1.0 - beta2 * cos_theta * cos_theta) ** 2
    r0_sq = 3.0 * SIGMA_T / (8.0 * math.pi)
    return r0_sq * beta / (4.0 * alpha * alpha) * numerator / denominator


def _pair_dsigma_direct_scalar(
    x1: float,
    x2: float,
    gamma_e: float,
    *,
    n_alpha: int = 24,
    n_phi: int = 48,
) -> float:
    """Evaluate the pair kernel directly in CM phase space."""
    # The calculation follows the same published Breit--Wheeler physics as
    # the analytical route above: the CM differential cross section and the
    # lab/CM Lorentz kinematics of Böttcher & Schlickeiser (1997), Sec. 2 and
    # Eqs. (2), (5)--(7), and (11). It is used only as a numerically separate
    # evaluation of the same kernel.
    product = x1 * x2
    energy = x1 + x2
    if x1 <= 0.0 or x2 <= 0.0 or gamma_e < 1.0 or product <= 1.0:
        return 0.0

    gamma_lo, gamma_hi = pair_gamma_bounds(x1, x2)
    if gamma_e <= gamma_lo or gamma_e >= gamma_hi:
        return 0.0

    alpha_lo, alpha_hi = _pair_alpha_bounds(x1, x2, gamma_e)
    if alpha_hi <= alpha_lo:
        return 0.0

    nodes, weights = _legendre_nodes(n_alpha)
    alpha = 0.5 * (alpha_hi - alpha_lo) * nodes + 0.5 * (alpha_hi + alpha_lo)
    alpha_weights = 0.5 * (alpha_hi - alpha_lo) * weights

    boost_beta2 = np.maximum(0.0, 1.0 - 4.0 * alpha * alpha / (energy * energy))
    lepton_beta2 = np.maximum(0.0, 1.0 - 1.0 / (alpha * alpha))
    boost_beta = np.sqrt(boost_beta2)
    lepton_beta = np.sqrt(lepton_beta2)
    denominator = energy * boost_beta * lepton_beta
    valid = denominator > 0.0
    if not np.any(valid):
        return 0.0

    cos_to_boost = np.zeros_like(alpha)
    cos_to_boost[valid] = (2.0 * gamma_e - energy) / denominator[valid]
    valid &= np.abs(cos_to_boost) <= 1.0 + 2.0e-11
    if not np.any(valid):
        return 0.0
    cos_to_boost = np.clip(cos_to_boost, -1.0, 1.0)

    cos_incident_boost = np.zeros_like(alpha)
    cos_incident_boost[valid] = (x1 - x2) / (energy * boost_beta[valid])
    cos_incident_boost = np.clip(cos_incident_boost, -1.0, 1.0)

    phi = (np.arange(n_phi, dtype=float) + 0.5) * (2.0 * math.pi / n_phi)
    cos_phi = np.cos(phi)
    u = cos_to_boost[:, None]
    z = cos_incident_boost[:, None]
    cos_theta = (
        u * z
        + np.sqrt(np.maximum(0.0, 1.0 - u * u))
        * np.sqrt(np.maximum(0.0, 1.0 - z * z))
        * cos_phi[None, :]
    )
    dsigma = _cm_dsigma_domega(cos_theta, alpha[:, None])
    phi_integral = (2.0 * math.pi / n_phi) * np.sum(dsigma, axis=1)

    integrand = np.zeros_like(alpha)
    integrand[valid] = (
        8.0
        * alpha[valid] ** 3
        / (
            product**2
            * energy
            * boost_beta[valid]
            * lepton_beta[valid]
        )
        * phi_integral[valid]
    )
    return float(np.sum(alpha_weights * integrand))


def _pair_dsigma_scalar(x1: float, x2: float, gamma_e: float) -> float:
    """Use the analytical kernel except where subtraction becomes ill-conditioned."""
    product = x1 * x2
    if x1 <= 0.0 or x2 <= 0.0 or gamma_e < 1.0 or product <= 1.0:
        return 0.0

    energy = x1 + x2
    midpoint = 0.5 * energy
    # The closed form has a removable midpoint singularity for equal photons.
    if (
        abs(x1 - x2) <= 1.0e-13 * max(x1, x2, 1.0)
        and abs(gamma_e - midpoint) <= 1.0e-13 * max(energy, 1.0)
    ):
        delta = 2.0e-7 * max(midpoint - 1.0, 1.0)
        left, _ = _pair_dsigma_analytic_scalar(x1, x2, midpoint - delta)
        right, _ = _pair_dsigma_analytic_scalar(x1, x2, midpoint + delta)
        return 0.5 * (left + right)

    value, conditioning = _pair_dsigma_analytic_scalar(x1, x2, gamma_e)
    ratio = max(x1, x2) / min(x1, x2)
    if ratio >= 100.0 and (
        not math.isfinite(value)
        or value <= 0.0
        or conditioning < 1.0e-10
    ):
        # Numerical sensitivity of the full Böttcher--Schlickeiser production
        # rate has also been noted in modelling work. Weidinger & Spanier
        # (2015, A&A 573, A7) used an approximate injection expression for
        # better numerical stability; here the physical kernel is retained and
        # only the numerical route is changed at poorly conditioned points.
        return _pair_dsigma_direct_scalar(x1, x2, gamma_e)
    return value


def pair_gamma_bounds(x1: float, x2: float) -> tuple[float, float]:
    """Return the kinematic Lorentz-factor range of one produced lepton."""
    x1 = float(x1)
    x2 = float(x2)
    if not math.isfinite(x1) or not math.isfinite(x2) or x1 < 0.0 or x2 < 0.0:
        raise InputValidationError("ERROR: photon energies must be finite and non-negative")
    product = x1 * x2
    if product <= 1.0:
        return 0.0, 0.0
    energy = x1 + x2
    if energy <= 2.0 * product:
        return 1.0, energy - 1.0

    difference = abs(x1 - x2)
    midpoint = 0.5 * energy
    half_width = 0.5 * difference * math.sqrt(1.0 - 1.0 / product)
    gamma_hi = midpoint + half_width
    # Use the product of the two roots to avoid cancellation in gamma_lo.
    root_product = product + difference * difference / (4.0 * product)
    gamma_lo = root_product / gamma_hi
    return gamma_lo, gamma_hi


def pair_dsigma_dgamma(x1, x2, gamma_e):
    """
Notes on implementation and references
--------------------------------------    
    Angle-averaged differential cross section for one produced lepton.

    The analytical expression I use implements Böttcher & Schlickeiser (1997),
    Eqs. (24)--(29).

    Parameters are dimensionless photon energies ``x1`` and ``x2`` in units of
    ``m_e c^2`` and the produced-lepton Lorentz factor ``gamma_e``. The result
    is in cm^2 per unit Lorentz factor. Its integral over ``gamma_e`` equals
    the ordinary isotropic angle-averaged gamma-gamma cross section for a
    distinguished test photon. This convention is twice the kernel appearing
    in the symmetric-population source rate of Böttcher & Schlickeiser (1997),
    where a factor 1/2 avoids the double-counting of identical photon pairs.
    """
    a, b, g = np.broadcast_arrays(
        np.asarray(x1, dtype=float),
        np.asarray(x2, dtype=float),
        np.asarray(gamma_e, dtype=float),
    )
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(g)):
        raise InputValidationError("ERROR: pair-kernel inputs must be finite")
    if np.any(a < 0.0) or np.any(b < 0.0) or np.any(g < 0.0):
        raise InputValidationError("ERROR: pair-kernel inputs must be non-negative")

    out = np.empty(a.size, dtype=float)
    for i, (v1, v2, vg) in enumerate(zip(a.ravel(), b.ravel(), g.ravel())):
        value = _pair_dsigma_scalar(float(v1), float(v2), float(vg))
        out[i] = max(0.0, value) if math.isfinite(value) else value
    out = out.reshape(a.shape)
    return float(out) if out.ndim == 0 else out


def _pair_target_bounds(
    target: PhotonSpectrum,
    E_MeV: float,
    gamma_e: float,
    explicit_bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    lo, hi = _integration_bounds_for_energy(target, E_MeV, explicit_bounds)
    if explicit_bounds is not None:
        return lo, hi

    x1 = E_MeV / _ME_C2_MEV
    required_keV = max(0.0, gamma_e + 1.0 - x1) * ME_C2_KEV
    if isinstance(target, (BlackbodySpectrum, GreybodySpectrum)):
        hi = max(hi, required_keV + 80.0 * float(target.kT_keV))
    return lo, hi


def _pair_line_spectrum(E_MeV: float, gamma_e, target: PhotonSpectrum):
    lines = discrete_lines(target)
    gamma = np.asarray(gamma_e, dtype=float)
    if lines is None:
        return np.zeros_like(gamma, dtype=float)
    energies_keV, densities_cm3 = lines
    x1 = E_MeV / _ME_C2_MEV
    result = np.zeros_like(gamma, dtype=float)
    for eps_keV, density_cm3 in zip(energies_keV, densities_cm3):
        x2 = float(eps_keV) / ME_C2_KEV
        result += float(density_cm3) * pair_dsigma_dgamma(x1, x2, gamma)
    return result


def pair_spectrum(
    E_MeV,
    electron_energy_MeV,
    target: PhotonSpectrum,
    *,
    n_energy: int = 160,
    energy_bounds_keV: tuple[float, float] | None = None,
):
    """Return the differential electron or positron production coefficient.

    The returned quantity is ``d alpha / d E_e`` in cm^-1 MeV^-1 for one
    charge species. Electron and positron marginals are identical for the
    unpolarized isotropic calculation used here.
    """
    E = float(positive_values(E_MeV, "MeV", "E_MeV must be finite and positive"))
    electron_energy = as_value(electron_energy_MeV, "MeV")
    electron_energy = np.asarray(electron_energy, dtype=float)
    if np.any(~np.isfinite(electron_energy)) or np.any(electron_energy < _ME_C2_MEV):
        raise InputValidationError(
            "ERROR: electron_energy_MeV must contain finite total energies >= 0.511 MeV"
        )
    if not isinstance(target, PhotonSpectrum):
        raise InputValidationError("ERROR: target must implement PhotonSpectrum")
    if int(n_energy) != n_energy or n_energy < 4:
        raise InputValidationError("ERROR: n_energy must be an integer >= 4")

    if isinstance(target, CompositeSpectrum) and energy_bounds_keV is None:
        values = [
            pair_spectrum(E, electron_energy, component, n_energy=n_energy)
            for component in target.components
        ]
        result = np.sum(np.asarray(values), axis=0)
        return float(result) if np.asarray(result).ndim == 0 else result

    gamma = electron_energy / _ME_C2_MEV
    result = _pair_line_spectrum(E, gamma, target)
    if discrete_lines(target) is not None:
        result = result / _ME_C2_MEV
        return float(result) if result.ndim == 0 else result

    x1 = E / _ME_C2_MEV
    flat_gamma = gamma.reshape(-1)
    flat_result = result.reshape(-1)
    threshold_keV = ME_C2_KEV / x1

    for index, gamma_e in enumerate(flat_gamma):
        if gamma_e < 1.0:
            continue
        lo_base, hi = _pair_target_bounds(target, E, float(gamma_e), energy_bounds_keV)
        required_keV = max(0.0, float(gamma_e) + 1.0 - x1) * ME_C2_KEV
        lo = max(lo_base, threshold_keV, required_keV)
        if not _has_log_span(lo, hi):
            continue

        boundaries = _boundaries(target, lo, hi)
        orders = _orders_for_segments(boundaries, int(n_energy))
        subtotal = 0.0
        for seg_lo, seg_hi, order in zip(boundaries[:-1], boundaries[1:], orders):
            nodes, weights = _legendre_nodes(int(order))
            log_lo, log_hi = math.log(seg_lo), math.log(seg_hi)
            log_eps = 0.5 * (log_hi - log_lo) * nodes + 0.5 * (log_hi + log_lo)
            eps_keV = np.exp(log_eps)
            quadrature = 0.5 * (log_hi - log_lo) * weights * eps_keV
            density = np.asarray(target.number_density(eps_keV), dtype=float)
            if (
                density.shape != eps_keV.shape
                or np.any(~np.isfinite(density))
                or np.any(density < 0.0)
            ):
                raise InputValidationError("ERROR: target returned an invalid number-density array")
            x2 = eps_keV / ME_C2_KEV
            kernel = pair_dsigma_dgamma(x1, x2, float(gamma_e))
            subtotal += float(np.sum(quadrature * density * kernel))
        flat_result[index] += subtotal

    # d gamma_e / d E_e(MeV) = 1 / (m_e c^2 in MeV).
    result = flat_result.reshape(gamma.shape) / _ME_C2_MEV
    return float(result) if result.ndim == 0 else result


def pair_spectrum_adaptive(
    E_MeV,
    electron_energy_MeV,
    target: PhotonSpectrum,
    *,
    epsrel: float = 1.0e-7,
    energy_bounds_keV: tuple[float, float] | None = None,
):
    """Evaluate the differential pair spectrum with adaptive target integration."""
    E = float(as_value(E_MeV, "MeV"))
    electron_energy = np.asarray(as_value(electron_energy_MeV, "MeV"), dtype=float)
    if not np.isfinite(E) or E <= 0.0:
        raise InputValidationError("ERROR: E_MeV must be finite and positive")
    if np.any(~np.isfinite(electron_energy)) or np.any(electron_energy < _ME_C2_MEV):
        raise InputValidationError(
            "ERROR: electron_energy_MeV must contain finite total energies >= 0.511 MeV"
        )
    if not isinstance(target, PhotonSpectrum):
        raise InputValidationError("ERROR: target must implement PhotonSpectrum")
    if not np.isfinite(epsrel) or epsrel <= 0.0:
        raise InputValidationError("ERROR: epsrel must be finite and positive")

    if isinstance(target, CompositeSpectrum) and energy_bounds_keV is None:
        values = [
            pair_spectrum_adaptive(E, electron_energy, component, epsrel=epsrel)
            for component in target.components
        ]
        result = np.sum(np.asarray(values), axis=0)
        return float(result) if np.asarray(result).ndim == 0 else result

    gamma = electron_energy / _ME_C2_MEV
    result = _pair_line_spectrum(E, gamma, target)
    if discrete_lines(target) is not None:
        result = result / _ME_C2_MEV
        return float(result) if result.ndim == 0 else result

    x1 = E / _ME_C2_MEV
    threshold_keV = ME_C2_KEV / x1
    flat_gamma = gamma.reshape(-1)
    flat_result = result.reshape(-1)

    for index, gamma_e in enumerate(flat_gamma):
        if gamma_e < 1.0:
            continue
        lo_base, hi = _pair_target_bounds(target, E, float(gamma_e), energy_bounds_keV)
        required_keV = max(0.0, float(gamma_e) + 1.0 - x1) * ME_C2_KEV
        lo = max(lo_base, threshold_keV, required_keV)
        if not _has_log_span(lo, hi):
            continue

        points = [math.log(x) for x in integration_breakpoints(target) if lo < x < hi]

        def integrand(log_epsilon: float) -> float:
            epsilon = math.exp(log_epsilon)
            density = float(target.number_density(epsilon))
            if not math.isfinite(density) or density < 0.0:
                raise InputValidationError("ERROR: target returned an invalid number density")
            x2 = epsilon / ME_C2_KEV
            return epsilon * density * pair_dsigma_dgamma(x1, x2, float(gamma_e))

        value, _ = integrate.quad(
            integrand,
            math.log(lo),
            math.log(hi),
            epsabs=0.0,
            epsrel=epsrel,
            limit=400,
            points=points or None,
        )
        flat_result[index] += value

    result = flat_result.reshape(gamma.shape) / _ME_C2_MEV
    return float(result) if result.ndim == 0 else result


def pair_distribution(
    E_MeV,
    electron_energy_MeV,
    target: PhotonSpectrum,
    *,
    n_energy: int = 160,
    n_angle: int = 128,
):
    """Return ``P(E_e | E_gamma)`` in MeV^-1 for one produced lepton."""
    spectrum = pair_spectrum(E_MeV, electron_energy_MeV, target, n_energy=n_energy)
    alpha = alpha_isotropic_gauss(E_MeV, target, n_energy=n_energy, n_angle=n_angle)
    if alpha <= 0.0:
        return np.zeros_like(np.asarray(spectrum, dtype=float))
    result = np.asarray(spectrum, dtype=float) / float(alpha)
    return float(result) if result.ndim == 0 else result


def pair_injection(
    gamma_energy_MeV,
    gamma_density_cm3_MeV,
    electron_energy_MeV,
    target: PhotonSpectrum,
    *,
    n_energy: int = 128,
    combined: bool = False,
    same_photon_population: bool = False,
):
    """Convolve a gamma-ray population into an electron/positron source term.

    ``gamma_density_cm3_MeV`` is the differential incident gamma-ray number
    density in cm^-3 MeV^-1. The result is in cm^-3 s^-1 MeV^-1. By default it
    is the source for one charge species; ``combined=True`` returns e- plus e+.

    Set ``same_photon_population=True`` only when the incident gamma-ray array
    and ``target`` represent the same physical photon population. The resulting
    factor 1/2 prevents counting each unordered photon pair twice. Leave it
    false when the incident and target fields are different populations.
    """
    gamma_energy = np.asarray(as_value(gamma_energy_MeV, "MeV"), dtype=float)
    gamma_density = np.asarray(
        as_value(gamma_density_cm3_MeV, "1 / (cm3 MeV)"), dtype=float
    )
    electron_energy = np.asarray(as_value(electron_energy_MeV, "MeV"), dtype=float)
    if (
        gamma_energy.ndim != 1
        or gamma_density.ndim != 1
        or gamma_energy.size != gamma_density.size
    ):
        raise InputValidationError(
            "ERROR: gamma energy and density must be one-dimensional arrays of equal length"
        )
    if (
        gamma_energy.size < 2
        or np.any(~np.isfinite(gamma_energy))
        or np.any(gamma_energy <= 0.0)
    ):
        raise InputValidationError(
            "ERROR: gamma_energy_MeV must contain at least two finite positive values"
        )
    if np.any(np.diff(gamma_energy) <= 0.0):
        raise InputValidationError("ERROR: gamma_energy_MeV must be strictly increasing")
    if np.any(~np.isfinite(gamma_density)) or np.any(gamma_density < 0.0):
        raise InputValidationError("ERROR: gamma_density_cm3_MeV must be finite and non-negative")
    if (
        electron_energy.ndim != 1
        or np.any(~np.isfinite(electron_energy))
        or np.any(electron_energy < _ME_C2_MEV)
    ):
        raise InputValidationError(
            "ERROR: electron_energy_MeV must be a one-dimensional array of finite "
            "total energies >= 0.511 MeV"
        )

    if not isinstance(combined, (bool, np.bool_)):
        raise InputValidationError("ERROR: combined must be boolean")
    if not isinstance(same_photon_population, (bool, np.bool_)):
        raise InputValidationError("ERROR: same_photon_population must be boolean")

    matrix = np.empty((gamma_energy.size, electron_energy.size), dtype=float)
    for i, E in enumerate(gamma_energy):
        matrix[i] = pair_spectrum(float(E), electron_energy, target, n_energy=n_energy)
    integrand = gamma_density[:, None] * matrix
    result = C_CGS * integrate.trapezoid(integrand, x=gamma_energy, axis=0)
    if same_photon_population:
        result = 0.5 * result
    if combined:
        result = 2.0 * result
    return result
