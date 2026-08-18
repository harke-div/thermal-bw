"""Target spectra for isotropic gamma-gamma opacity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator

from .constants import C_CGS, H_CGS, KEV_TO_ERG, PI
from .exceptions import InputValidationError
from .units import as_value


@runtime_checkable
class PhotonSpectrum(Protocol):
    """Protocol implemented by isotropic target photon spectra."""

    @property
    def energy_bounds_keV(self) -> tuple[float, float]:
        """Finite numerical integration bounds in keV."""

    def number_density(self, epsilon_keV):
        """Return differential number density in cm^-3 keV^-1."""


def _validate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lo = float(as_value(bounds[0], "keV"))
    hi = float(as_value(bounds[1], "keV"))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
        raise InputValidationError("ERROR: energy bounds must satisfy 0 < lower < upper and be finite")
    return lo, hi


def _validate_breakpoints(
    breakpoints: Sequence[float], bounds: tuple[float, float]
) -> tuple[float, ...]:
    lo, hi = _validate_bounds(bounds)
    values = np.asarray(tuple(breakpoints), dtype=float)
    if values.size == 0:
        return ()
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise InputValidationError(
            "ERROR: integration breakpoints must be a finite one-dimensional sequence"
        )
    if np.any(values <= lo) or np.any(values >= hi):
        raise InputValidationError("ERROR: integration breakpoints must lie strictly inside energy bounds")
    return tuple(float(x) for x in np.unique(values))


def integration_breakpoints(target: PhotonSpectrum) -> tuple[float, ...]:
    """Return validated target-energy breakpoints advertised by ``target``.

    Unknown user-defined targets are allowed to omit this optional property.
    """
    raw = getattr(target, "integration_breakpoints_keV", ())
    if callable(raw):
        raw = raw()
    return _validate_breakpoints(raw, target.energy_bounds_keV) if raw else ()


def discrete_lines(target: PhotonSpectrum) -> tuple[np.ndarray, np.ndarray] | None:
    """Return discrete line energies and integrated number densities, if any."""
    raw = getattr(target, "discrete_lines", None)
    if raw is None:
        return None
    if callable(raw):
        raw = raw()
    energies, densities = raw
    energies = np.asarray(energies, dtype=float)
    densities = np.asarray(densities, dtype=float)
    if (
        energies.ndim != 1
        or densities.ndim != 1
        or energies.size != densities.size
        or energies.size < 1
        or np.any(~np.isfinite(energies))
        or np.any(~np.isfinite(densities))
        or np.any(energies <= 0.0)
        or np.any(densities < 0.0)
    ):
        raise InputValidationError(
            "ERROR: discrete lines require positive energies and non-negative densities"
        )
    return energies, densities


@dataclass(frozen=True)
class BlackbodySpectrum:
    """Undiluted isotropic blackbody photon field."""

    kT_keV: float
    u_max: float = 80.0
    u_min: float = 1e-8

    def __post_init__(self):
        value = float(as_value(self.kT_keV, "keV"))
        object.__setattr__(self, "kT_keV", value)
        if not np.isfinite(self.kT_keV) or self.kT_keV <= 0.0:
            raise InputValidationError("ERROR: kT_keV must be finite and positive")
        if self.u_max <= self.u_min or self.u_min <= 0.0:
            raise InputValidationError("ERROR: require 0 < u_min < u_max")

    @property
    def energy_bounds_keV(self) -> tuple[float, float]:
        return self.u_min * self.kT_keV, self.u_max * self.kT_keV

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        return ()

    @property
    def has_infinite_high_energy_tail(self) -> bool:
        return True

    def number_density(self, epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        eps_erg = eps * KEV_TO_ERG
        kT_erg = self.kT_keV * KEV_TO_ERG
        x = eps_erg / kT_erg
        prefactor = (8.0 * PI) / (H_CGS**3 * C_CGS**3)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            denom = np.expm1(x)
            values = prefactor * eps_erg**2 / denom * KEV_TO_ERG
        values = np.where((eps > 0.0) & np.isfinite(values), values, 0.0)
        if values.ndim == 0:
            return float(values)
        return values


@dataclass(frozen=True)
class GreybodySpectrum:
    """Diluted blackbody with number density scaled by ``dilution``."""

    kT_keV: float
    dilution: float
    u_max: float = 80.0
    u_min: float = 1e-8

    def __post_init__(self):
        value = float(as_value(self.kT_keV, "keV"))
        object.__setattr__(self, "kT_keV", value)
        if not np.isfinite(self.dilution) or self.dilution < 0.0:
            raise InputValidationError("ERROR: dilution must be finite and non-negative")
        BlackbodySpectrum(self.kT_keV, self.u_max, self.u_min)

    @property
    def energy_bounds_keV(self) -> tuple[float, float]:
        return self.u_min * self.kT_keV, self.u_max * self.kT_keV

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        return ()

    @property
    def has_infinite_high_energy_tail(self) -> bool:
        return True

    def number_density(self, epsilon_keV):
        return self.dilution * BlackbodySpectrum(
            self.kT_keV, self.u_max, self.u_min
        ).number_density(epsilon_keV)


@dataclass(frozen=True)
class PowerLawSpectrum:
    """Finite isotropic power-law target spectrum.

    ``normalization`` is the differential number density at ``reference_keV``.
    An optional exponential cutoff may be supplied.
    """

    normalization: float
    index: float
    energy_bounds_keV: tuple[float, float]
    reference_keV: float = 1.0
    cutoff_keV: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "energy_bounds_keV", _validate_bounds(self.energy_bounds_keV))
        object.__setattr__(self, "reference_keV", float(as_value(self.reference_keV, "keV")))
        if self.cutoff_keV is not None:
            object.__setattr__(self, "cutoff_keV", float(as_value(self.cutoff_keV, "keV")))
        if not np.isfinite(self.normalization) or self.normalization < 0.0:
            raise InputValidationError("ERROR: normalization must be finite and non-negative")
        if not np.isfinite(self.index):
            raise InputValidationError("ERROR: index must be finite")
        if not np.isfinite(self.reference_keV) or self.reference_keV <= 0.0:
            raise InputValidationError("ERROR: reference_keV must be finite and positive")
        if self.cutoff_keV is not None and (
            not np.isfinite(self.cutoff_keV) or self.cutoff_keV <= 0.0
        ):
            raise InputValidationError("ERROR: cutoff_keV must be finite and positive")

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        return ()

    def number_density(self, epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        lo, hi = self.energy_bounds_keV
        with np.errstate(over="ignore", invalid="ignore"):
            values = self.normalization * (eps / self.reference_keV) ** (-self.index)
            if self.cutoff_keV is not None:
                values *= np.exp(-eps / self.cutoff_keV)
        values = np.where((eps >= lo) & (eps <= hi) & np.isfinite(values), values, 0.0)
        if values.ndim == 0:
            return float(values)
        return values


@dataclass(frozen=True)
class BrokenPowerLawSpectrum:
    """Finite continuous broken power-law target spectrum."""

    normalization_at_break: float
    index_low: float
    index_high: float
    break_keV: float
    energy_bounds_keV: tuple[float, float]
    cutoff_keV: float | None = None

    def __post_init__(self):
        bounds = _validate_bounds(self.energy_bounds_keV)
        object.__setattr__(self, "energy_bounds_keV", bounds)
        object.__setattr__(self, "break_keV", float(as_value(self.break_keV, "keV")))
        if self.cutoff_keV is not None:
            object.__setattr__(self, "cutoff_keV", float(as_value(self.cutoff_keV, "keV")))
        lo, hi = bounds
        if not np.isfinite(self.normalization_at_break) or self.normalization_at_break < 0.0:
            raise InputValidationError("ERROR: normalization_at_break must be non-negative")
        if not np.isfinite(self.break_keV) or not (lo < self.break_keV < hi):
            raise InputValidationError("ERROR: break_keV must lie inside energy_bounds_keV")
        if not np.isfinite(self.index_low) or not np.isfinite(self.index_high):
            raise InputValidationError("ERROR: indices must be finite")
        if self.cutoff_keV is not None and (
            not np.isfinite(self.cutoff_keV) or self.cutoff_keV <= 0.0
        ):
            raise InputValidationError("ERROR: cutoff_keV must be finite and positive")

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        return (float(self.break_keV),)

    def number_density(self, epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        lo, hi = self.energy_bounds_keV
        ratio = eps / self.break_keV
        values = np.where(
            eps < self.break_keV,
            self.normalization_at_break * ratio ** (-self.index_low),
            self.normalization_at_break * ratio ** (-self.index_high),
        )
        if self.cutoff_keV is not None:
            values *= np.exp(-eps / self.cutoff_keV)
        values = np.where((eps >= lo) & (eps <= hi) & np.isfinite(values), values, 0.0)
        if values.ndim == 0:
            return float(values)
        return values


class TabulatedSpectrum:
    """Positive spectrum with monotone log-log interpolation."""

    def __init__(self, epsilon_keV, number_density_per_keV):
        eps = as_value(epsilon_keV, "keV")
        dens = np.asarray(number_density_per_keV, dtype=float)
        if eps.ndim != 1 or dens.ndim != 1 or eps.size != dens.size or eps.size < 2:
            raise InputValidationError("ERROR: tabulated arrays must be one-dimensional and equally sized")
        if np.any(~np.isfinite(eps)) or np.any(~np.isfinite(dens)):
            raise InputValidationError("ERROR: tabulated arrays must be finite")
        if np.any(eps <= 0.0) or np.any(dens <= 0.0) or np.any(np.diff(eps) <= 0.0):
            raise InputValidationError(
                "ERROR: energies and densities must be positive; energies must increase"
            )
        self._epsilon = eps
        self._density = dens
        self._interp = PchipInterpolator(np.log(eps), np.log(dens), extrapolate=False)

    @property
    def energy_bounds_keV(self) -> tuple[float, float]:
        return float(self._epsilon[0]), float(self._epsilon[-1])

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        return tuple(float(x) for x in self._epsilon[1:-1])

    def number_density(self, epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        lo, hi = self.energy_bounds_keV
        out = np.zeros_like(eps, dtype=float)
        mask = np.isfinite(eps) & (eps >= lo) & (eps <= hi)
        if np.any(mask):
            out[mask] = np.exp(self._interp(np.log(eps[mask])))
        if out.ndim == 0:
            return float(out)
        return out


@dataclass(frozen=True)
class CallableSpectrum:
    """Adapter for a callable photon spectrum in cm^-3 keV^-1."""

    function: Callable
    energy_bounds_keV: tuple[float, float]
    breakpoints_keV: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "energy_bounds_keV", _validate_bounds(self.energy_bounds_keV))
        if not callable(self.function):
            raise InputValidationError("ERROR: function must be callable")
        object.__setattr__(
            self,
            "breakpoints_keV",
            _validate_breakpoints(self.breakpoints_keV, self.energy_bounds_keV),
        )

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        return self.breakpoints_keV

    def number_density(self, epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        out = np.asarray(self.function(eps), dtype=float)
        try:
            out = np.broadcast_to(out, eps.shape).astype(float, copy=False)
        except ValueError as exc:
            raise InputValidationError(
                "ERROR: callable output is not broadcastable to input shape"
            ) from exc
        if np.any(~np.isfinite(out)) or np.any(out < 0.0):
            raise InputValidationError("ERROR: callable spectrum returned non-finite or negative density")
        lo, hi = self.energy_bounds_keV
        out = np.where((eps >= lo) & (eps <= hi), out, 0.0)
        if out.ndim == 0:
            return float(out)
        return out


@dataclass(frozen=True)
class DiscreteLineSpectrum:
    """Monochromatic isotropic lines with integrated photon densities."""

    energies_keV: Sequence[float]
    number_densities_cm3: Sequence[float]

    def __post_init__(self):
        energies = as_value(self.energies_keV, "keV")
        densities = np.asarray(self.number_densities_cm3, dtype=float)
        if (
            energies.ndim != 1
            or densities.ndim != 1
            or energies.size != densities.size
            or energies.size < 1
            or np.any(~np.isfinite(energies))
            or np.any(~np.isfinite(densities))
            or np.any(energies <= 0.0)
            or np.any(densities < 0.0)
        ):
            raise InputValidationError(
                "ERROR: line energies must be positive; line densities must be non-negative"
            )
        order = np.argsort(energies)
        object.__setattr__(self, "energies_keV", tuple(float(x) for x in energies[order]))
        object.__setattr__(
            self,
            "number_densities_cm3",
            tuple(float(x) for x in densities[order]),
        )

    @property
    def energy_bounds_keV(self) -> tuple[float, float]:
        lo = float(self.energies_keV[0])
        hi = float(self.energies_keV[-1])
        if hi == lo:
            # Numerical methods handle the line term separately.
            return lo * (1.0 - 1e-12), hi * (1.0 + 1e-12)
        return lo, hi

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        return tuple(float(x) for x in self.energies_keV)

    @property
    def discrete_lines(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.energies_keV, dtype=float),
            np.asarray(self.number_densities_cm3, dtype=float),
        )

    def number_density(self, epsilon_keV):
        # Line opacity is added through ``discrete_lines``.
        eps = np.asarray(epsilon_keV, dtype=float)
        out = np.zeros_like(eps, dtype=float)
        return float(out) if out.ndim == 0 else out


class CompositeSpectrum:
    """Sum of multiple isotropic target spectra."""

    def __init__(self, components: Sequence[PhotonSpectrum]):
        if not components:
            raise InputValidationError("ERROR: CompositeSpectrum requires at least one component")
        self.components = tuple(components)
        for component in self.components:
            if not isinstance(component, PhotonSpectrum):
                raise InputValidationError("ERROR: all components must implement PhotonSpectrum")

    @property
    def energy_bounds_keV(self) -> tuple[float, float]:
        lows, highs = zip(*(c.energy_bounds_keV for c in self.components))
        return float(min(lows)), float(max(highs))

    @property
    def integration_breakpoints_keV(self) -> tuple[float, ...]:
        lo, hi = self.energy_bounds_keV
        values: list[float] = []
        for component in self.components:
            c_lo, c_hi = component.energy_bounds_keV
            values.extend([c_lo, c_hi])
            values.extend(integration_breakpoints(component))
        return tuple(sorted({float(x) for x in values if lo < x < hi}))

    def number_density(self, epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        out = np.zeros_like(eps, dtype=float)
        for component in self.components:
            out = out + np.asarray(component.number_density(eps), dtype=float)
        if out.ndim == 0:
            return float(out)
        return out
