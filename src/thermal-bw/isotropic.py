"""Here I code opacity calculations in arbitrary isotropic target photon fields."""
from __future__ import annotations

from functools import lru_cache
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

from .constants import ME_C2_KEV
from .cross_sections import sigma_breit_wheeler_s
from .exceptions import InputValidationError
from .targets import PhotonSpectrum, discrete_lines, integration_breakpoints
from .units import positive_values


class ResolutionWarning(UserWarning):
    """Warning that fixed quadrature was replaced by adaptive integration."""


@lru_cache(maxsize=32)
def _legendre_nodes(order: int):
    if int(order) != order or order < 4:
        raise InputValidationError("ERROR: quadrature order must be an integer >= 4")
    return leggauss(int(order))


def _validate_energy(E_MeV):
    return positive_values(
        E_MeV, "MeV", "E_MeV must contain only finite positive values"
    )


def _integration_bounds_for_energy(
    target: PhotonSpectrum,
    energy_MeV: float,
    explicit_bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Target-energy bounds for one test-photon energy."""
    if explicit_bounds is not None:
        lo, hi = map(float, explicit_bounds)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
            raise InputValidationError("ERROR: energy bounds must satisfy 0 < lower < upper")
        return lo, hi

    lo, hi = map(float, target.energy_bounds_keV)
    from .targets import BlackbodySpectrum, GreybodySpectrum

    if isinstance(target, (BlackbodySpectrum, GreybodySpectrum)):
        threshold = ME_C2_KEV**2 / (float(energy_MeV) * 1.0e3)
        hi = max(hi, threshold + 80.0 * float(target.kT_keV))
    return lo, hi


def _global_bounds_for_range(
    target: PhotonSpectrum,
    E_min_MeV: float,
    explicit_bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    return _integration_bounds_for_energy(target, E_min_MeV, explicit_bounds)


def _has_log_span(lo: float, hi: float) -> bool:
    return hi > lo and np.log(hi) > np.log(lo)


def _boundaries(target: PhotonSpectrum, lo: float, hi: float) -> np.ndarray:
    values = [lo]
    values.extend(x for x in integration_breakpoints(target) if lo < x < hi)
    values.append(hi)
    values = np.asarray(sorted(set(map(float, values))))
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise InputValidationError("ERROR: invalid target integration boundaries")
    return values


def _orders_for_segments(boundaries: np.ndarray, total_order: int) -> np.ndarray:
    if int(total_order) != total_order or total_order < 4:
        raise InputValidationError("ERROR: quadrature order must be an integer >= 4")
    spans = np.diff(np.log(boundaries))
    if np.any(spans <= 0.0):
        raise InputValidationError("ERROR: invalid logarithmic integration span")
    if spans.size == 1:
        return np.asarray([int(total_order)])
    return np.maximum(8, np.rint(total_order * spans / spans.sum()).astype(int))


def angle_averaged_cross_section(product, *, n_angle: int = 96):
    """Return the isotropic angle-averaged cross section in cm^2."""
    z = np.asarray(product, dtype=float)
    if np.any(~np.isfinite(z)) or np.any(z < 0.0):
        raise InputValidationError("ERROR: product must be finite and non-negative")

    nodes, weights = _legendre_nodes(n_angle)
    flat = z.reshape(-1)
    out = np.zeros_like(flat)
    active = flat > 1.0
    if np.any(active):
        z_active = flat[active]
        t_min = 2.0 / z_active
        half = 0.5 * (2.0 - t_min)
        mid = 0.5 * (2.0 + t_min)
        t = half[:, None] * nodes[None, :] + mid[:, None]
        s = z_active[:, None] * t
        out[active] = 0.5 * half * np.sum(
            weights[None, :] * t * sigma_breit_wheeler_s(s), axis=1
        )
    out = out.reshape(z.shape)
    return float(out) if out.ndim == 0 else out


def _line_contribution(E, target: PhotonSpectrum, kernel) -> np.ndarray:
    lines = discrete_lines(target)
    if lines is None:
        return np.zeros_like(E, dtype=float)
    energies, densities = lines
    z = E.reshape(-1, 1) * 1.0e3 * energies.reshape(1, -1) / ME_C2_KEV**2
    values = np.sum(densities.reshape(1, -1) * np.asarray(kernel(z)), axis=1)
    return values.reshape(E.shape)


def _density(target: PhotonSpectrum, energy_keV):
    value = np.asarray(target.number_density(energy_keV), dtype=float)
    if value.shape != np.shape(energy_keV) or np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise InputValidationError("ERROR: target returned an invalid number-density array")
    return value


def alpha_isotropic_gauss(
    E_MeV,
    target: PhotonSpectrum,
    *,
    n_energy: int = 128,
    n_angle: int = 96,
    energy_bounds_keV: tuple[float, float] | None = None,
):
    """Integrate an isotropic target with fixed Gauss-Legendre quadrature."""
    from .targets import CompositeSpectrum

    E = _validate_energy(E_MeV)
    if not isinstance(target, PhotonSpectrum):
        raise InputValidationError("ERROR: target must implement PhotonSpectrum")

    if isinstance(target, CompositeSpectrum) and energy_bounds_keV is None:
        pieces = [
            alpha_isotropic_gauss(E, item, n_energy=n_energy, n_angle=n_angle)
            for item in target.components
        ]
        result = np.sum(np.asarray(pieces), axis=0)
        return float(result) if np.asarray(result).ndim == 0 else result

    kernel = lambda z: angle_averaged_cross_section(z, n_angle=n_angle)
    result = _line_contribution(E, target, kernel).reshape(-1)
    if discrete_lines(target) is not None:
        result = result.reshape(E.shape)
        return float(result) if result.ndim == 0 else result

    for i, energy in enumerate(E.reshape(-1)):
        lo, hi = _integration_bounds_for_energy(target, float(energy), energy_bounds_keV)
        lo = max(lo, ME_C2_KEV**2 / (float(energy) * 1.0e3))
        if not _has_log_span(lo, hi):
            continue

        boundaries = _boundaries(target, lo, hi)
        orders = _orders_for_segments(boundaries, n_energy)
        for left, right, order in zip(boundaries[:-1], boundaries[1:], orders):
            nodes, weights = _legendre_nodes(int(order))
            log_left, log_right = np.log(left), np.log(right)
            log_eps = 0.5 * (log_right - log_left) * nodes + 0.5 * (log_right + log_left)
            eps = np.exp(log_eps)
            jac = 0.5 * (log_right - log_left) * weights * eps
            z = float(energy) * 1.0e3 * eps / ME_C2_KEV**2
            result[i] += np.sum(jac * _density(target, eps) * kernel(z))

    result = result.reshape(E.shape)
    return float(result) if result.ndim == 0 else result


def _cached_interval(E, target, kernel, n_energy, bounds, chunk_size):
    """vectorized fixed quadrature over one target-energy interval"""
    nodes, weights = _legendre_nodes(int(n_energy))
    flat = E.reshape(-1)
    result = np.zeros_like(flat)
    lo_base, hi = map(float, bounds)

    for start in range(0, flat.size, int(chunk_size)):
        stop = min(start + int(chunk_size), flat.size)
        energies = flat[start:stop]
        threshold = ME_C2_KEV**2 / (energies * 1.0e3)
        lo = np.maximum(lo_base, threshold)
        active = lo < hi
        if not np.any(active):
            continue

        log_lo = np.log(lo[active])
        log_hi = np.log(hi)
        log_eps = 0.5 * (log_hi - log_lo)[:, None] * nodes + 0.5 * (log_hi + log_lo)[:, None]
        eps = np.exp(log_eps)
        jac = 0.5 * (log_hi - log_lo)[:, None] * weights * eps
        z = energies[active, None] * 1.0e3 * eps / ME_C2_KEV**2
        value = np.sum(jac * _density(target, eps) * np.asarray(kernel(z)), axis=1)
        result[start + np.flatnonzero(active)] = value
    return result.reshape(E.shape)


def alpha_isotropic_cached(
    E_MeV,
    target: PhotonSpectrum,
    *,
    preset: str = "accurate",
    kernel=None,
    n_energy: int | None = None,
    energy_bounds_keV: tuple[float, float] | None = None,
    chunk_size: int = 4096,
):
    """Integrate an isotropic target using a tabulated angular kernel"""
    from .kernel import cached_kernel_preset
    from .targets import CompositeSpectrum

    E = _validate_energy(E_MeV)
    if not isinstance(target, PhotonSpectrum):
        raise InputValidationError("ERROR: target must implement PhotonSpectrum")
    if preset not in {"fast", "balanced", "accurate"}:
        raise InputValidationError("ERROR: preset must be 'fast', 'balanced', or 'accurate'")
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise InputValidationError("ERROR: chunk_size must be a positive integer")

    if kernel is None:
        kernel = cached_kernel_preset(preset)
    if not callable(kernel):
        raise InputValidationError("ERROR: kernel must be callable")
    if n_energy is None:
        n_energy = {"fast": 24, "balanced": 32, "accurate": 64}[preset]

    if isinstance(target, CompositeSpectrum) and energy_bounds_keV is None:
        pieces = [
            alpha_isotropic_cached(
                E,
                item,
                preset=preset,
                kernel=kernel,
                n_energy=n_energy,
                chunk_size=chunk_size,
            )
            for item in target.components
        ]
        result = np.sum(np.asarray(pieces), axis=0)
        return float(result) if np.asarray(result).ndim == 0 else result

    result = _line_contribution(E, target, kernel)
    if discrete_lines(target) is not None:
        return float(result) if result.ndim == 0 else result

    lo, hi = _global_bounds_for_range(target, float(np.min(E)), energy_bounds_keV)
    boundaries = _boundaries(target, lo, hi)
    orders = _orders_for_segments(boundaries, n_energy)
    for left, right, order in zip(boundaries[:-1], boundaries[1:], orders):
        result = result + _cached_interval(
            E,
            target,
            kernel,
            int(order),
            (float(left), float(right)),
            int(chunk_size),
        )
    return float(result) if result.ndim == 0 else result


def alpha_isotropic_adaptive(
    E_MeV: float,
    target: PhotonSpectrum,
    *,
    epsrel: float = 1e-8,
    epsabs: float = 0.0,
    angle_epsrel: float | None = None,
):
    """nested adaptive reference calculation for one test-photon energy"""
    from .targets import CompositeSpectrum

    E = float(_validate_energy(E_MeV))
    if not isinstance(target, PhotonSpectrum):
        raise InputValidationError("ERROR: target must implement PhotonSpectrum")
    if epsrel <= 0.0 or epsabs < 0.0:
        raise InputValidationError("ERROR: require epsrel > 0 and epsabs >= 0")

    if isinstance(target, CompositeSpectrum):
        return float(sum(
            alpha_isotropic_adaptive(
                E, item, epsrel=epsrel, epsabs=epsabs, angle_epsrel=angle_epsrel
            )
            for item in target.components
        ))

    line_value = float(_line_contribution(
        np.asarray(E),
        target,
        lambda z: angle_averaged_cross_section(z, n_angle=256),
    ))
    if discrete_lines(target) is not None:
        return line_value

    lo, hi = _integration_bounds_for_energy(target, E)
    lo = max(lo, ME_C2_KEV**2 / (E * 1.0e3))
    if not _has_log_span(lo, hi):
        return line_value
    angle_tol = epsrel if angle_epsrel is None else angle_epsrel

    def target_integrand(logeps: float) -> float:
        eps = float(np.exp(logeps))
        z = E * 1.0e3 * eps / ME_C2_KEV**2
        if z <= 1.0:
            return 0.0

        def angular_integrand(t: float) -> float:
            return 0.5 * t * sigma_breit_wheeler_s(z * t)

        kernel, _ = integrate.quad(
            angular_integrand,
            2.0 / z,
            2.0,
            epsabs=epsabs,
            epsrel=angle_tol,
            limit=200,
        )
        density = float(target.number_density(eps))
        if not np.isfinite(density) or density < 0.0:
            raise InputValidationError("ERROR: target returned an invalid number density")
        return eps * density * kernel

    points = [np.log(x) for x in integration_breakpoints(target) if lo < x < hi]
    value, _ = integrate.quad(
        target_integrand,
        np.log(lo),
        np.log(hi),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
        points=points or None,
    )
    return float(line_value + value)


def alpha_isotropic_auto(
    E_MeV,
    target: PhotonSpectrum,
    *,
    rtol: float = 1e-3,
    initial_order: int = 24,
    max_order: int = 384,
    preset: str = "accurate",
    chunk_size: int = 4096,
):
    """Increase fixed quadrature order and use adaptive integration if needed.

    Known sharp features should still be supplied as target breakpoints.  For a
    callable without breakpoints, the converged fixed result is checked against
    the independent adaptive calculation.
    """
    from .targets import CallableSpectrum

    if rtol <= 0.0 or int(initial_order) != initial_order or initial_order < 4:
        raise InputValidationError("ERROR: require rtol > 0 and initial_order >= 4")
    if int(max_order) != max_order or max_order < 2 * initial_order:
        raise InputValidationError("ERROR: max_order must be an integer at least twice initial_order")

    E = _validate_energy(E_MeV)
    order = int(initial_order)
    previous = alpha_isotropic_cached(
        E, target, preset=preset, n_energy=order, chunk_size=chunk_size
    )

    while 2 * order <= int(max_order):
        order *= 2
        current = alpha_isotropic_cached(
            E, target, preset=preset, n_energy=order, chunk_size=chunk_size
        )
        scale = np.maximum(np.abs(current), np.finfo(float).tiny)
        error = np.abs(np.asarray(current) - np.asarray(previous)) / scale
        if np.all(error <= rtol):
            if isinstance(target, CallableSpectrum) and not target.integration_breakpoints_keV:
                flat = np.asarray(E).reshape(-1)
                adaptive = np.asarray([
                    alpha_isotropic_adaptive(float(value), target, epsrel=max(rtol / 5.0, 1e-8))
                    for value in flat
                ]).reshape(np.shape(E))
                scale = np.maximum(np.abs(adaptive), np.finfo(float).tiny)
                mismatch = np.abs(np.asarray(current) - adaptive) / scale
                if np.any(mismatch > rtol):
                    warnings.warn(
                        "WARNING: fixed quadrature did not meet the requested tolerance; returning adaptive integration",
                        ResolutionWarning,
                        stacklevel=2,
                    )
                    return float(adaptive) if adaptive.ndim == 0 else adaptive
            return current
        previous = current

    raise RuntimeError(
        f"ERROR: target-energy quadrature did not converge to rtol={rtol:g} by order {max_order}; "
        "use adaptive integration or provide feature breakpoints"
    )
