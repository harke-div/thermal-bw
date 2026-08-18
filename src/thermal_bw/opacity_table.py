"""Here I present code for an interpolation table for repeated opacity calculations with one target."""
from __future__ import annotations

import warnings

import numpy as np

from .constants import ME_C2_KEV
from .exceptions import InputValidationError, OutOfDomainError, OutOfDomainWarning
from .isotropic import _validate_energy, alpha_isotropic_cached
from .targets import PhotonSpectrum


def _threshold_energy(target: PhotonSpectrum) -> float:
    if bool(getattr(target, "has_infinite_high_energy_tail", False)):
        return 0.0
    return ME_C2_KEV**2 / (float(target.energy_bounds_keV[1]) * 1.0e3)


def _coordinate(E, threshold):
    E = np.asarray(E, dtype=float)
    if threshold == 0.0:
        return np.log10(E)
    return np.log10(E / threshold - 1.0)


def _energy_from_coordinate(x, threshold):
    if threshold == 0.0:
        return 10.0**np.asarray(x)
    return threshold * (1.0 + 10.0**np.asarray(x))


class TargetOpacityTable:
    """Logarithmic interpolation table for one fixed target spectrum."""

    def __init__(
        self,
        E_min_MeV,
        E_max_MeV,
        threshold_MeV,
        x_nodes,
        log_alpha_nodes,
        requested_rtol,
        validation_max_relative_error,
    ):
        self.E_min_MeV = float(E_min_MeV)
        self.E_max_MeV = float(E_max_MeV)
        self.threshold_MeV = float(threshold_MeV)
        self.x_nodes = np.asarray(x_nodes, dtype=float)
        self.log_alpha_nodes = np.asarray(log_alpha_nodes, dtype=float)
        self.requested_rtol = float(requested_rtol)
        self.validation_max_relative_error = float(validation_max_relative_error)

    @property
    def n_nodes(self) -> int:
        return int(self.x_nodes.size)

    @property
    def representation_bytes(self) -> int:
        return int(self.x_nodes.nbytes + self.log_alpha_nodes.nbytes)

    @classmethod
    def build(
        cls,
        target: PhotonSpectrum,
        E_bounds_MeV: tuple[float, float],
        *,
        rtol: float = 1e-3,
        initial_nodes: int = 48,
        max_nodes: int = 4096,
        reference_preset: str = "accurate",
        reference_n_energy: int = 128,
    ):
        """Build a table, adding midpoint nodes until the requested tolerance is met."""
        if not isinstance(target, PhotonSpectrum):
            raise InputValidationError("ERROR: target must implement PhotonSpectrum")
        E_min, E_max = map(float, E_bounds_MeV)
        if not np.isfinite(E_min) or not np.isfinite(E_max) or E_min <= 0.0 or E_max <= E_min:
            raise InputValidationError("ERROR: E_bounds_MeV must satisfy 0 < lower < upper")
        if not 0.0 < rtol < 1.0:
            raise InputValidationError("ERROR: rtol must lie between zero and one")
        if int(initial_nodes) != initial_nodes or initial_nodes < 8:
            raise InputValidationError("ERROR: initial_nodes must be an integer >= 8")
        if int(max_nodes) != max_nodes or max_nodes < initial_nodes:
            raise InputValidationError("ERROR: max_nodes must be an integer >= initial_nodes")

        threshold = float(_threshold_energy(target))
        if E_max <= threshold:
            raise InputValidationError("ERROR: requested energy range lies below pair threshold")
        positive_min = max(E_min, threshold * (1.0 + 1e-8)) if threshold else E_min

        x_nodes = np.linspace(
            float(_coordinate(positive_min, threshold)),
            float(_coordinate(E_max, threshold)),
            int(initial_nodes),
        )

        def reference(x):
            E = _energy_from_coordinate(x, threshold)
            return np.asarray(
                alpha_isotropic_cached(
                    E,
                    target,
                    preset=reference_preset,
                    n_energy=int(reference_n_energy),
                ),
                dtype=float,
            )

        while True:
            alpha_nodes = reference(x_nodes)
            if np.any(alpha_nodes <= 0.0) or np.any(~np.isfinite(alpha_nodes)):
                raise RuntimeError("ERROR: reference opacity is invalid inside the table range")
            log_nodes = np.log(alpha_nodes)

            mid = 0.5 * (x_nodes[:-1] + x_nodes[1:])
            true = reference(mid)
            pred = np.exp(np.interp(mid, x_nodes, log_nodes))
            relative = np.abs(pred - true) / true
            failing = relative > rtol
            if not np.any(failing):
                max_error = float(np.max(relative)) if relative.size else 0.0
                break
            additions = mid[failing]
            if x_nodes.size + additions.size > int(max_nodes):
                raise RuntimeError("ERROR: opacity table did not reach the requested tolerance")
            x_nodes = np.sort(np.concatenate([x_nodes, additions]))

        return cls(
            E_min,
            E_max,
            threshold,
            x_nodes,
            log_nodes,
            rtol,
            max_error,
        )

    def __call__(self, E_MeV, *, bounds: str = "warn"):
        E = _validate_energy(E_MeV)
        if bounds not in {"warn", "raise", "ignore"}:
            raise InputValidationError("ERROR: bounds must be 'warn', 'raise', or 'ignore'")

        outside = (E < self.E_min_MeV) | (E > self.E_max_MeV)
        if np.any(outside):
            message = (
                f"opacity table is valid only for E_MeV in "
                f"[{self.E_min_MeV:g}, {self.E_max_MeV:g}]"
            )
            if bounds == "raise":
                raise OutOfDomainError("ERROR: " + message)
            if bounds == "warn":
                warnings.warn("WARNING: " + message, OutOfDomainWarning, stacklevel=2)

        out = np.zeros_like(E, dtype=float)
        positive = E > self.threshold_MeV
        if np.any(positive):
            x = _coordinate(E[positive], self.threshold_MeV)
            out[positive] = np.exp(
                np.interp(
                    x,
                    self.x_nodes,
                    self.log_alpha_nodes,
                    left=self.log_alpha_nodes[0],
                    right=self.log_alpha_nodes[-1],
                )
            )
        return float(out) if out.ndim == 0 else out
