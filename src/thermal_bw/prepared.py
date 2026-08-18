"""Here I present code for precomputed target quadrature for repeated opacity calculations."""
from __future__ import annotations

import numpy as np

from .constants import ME_C2_KEV
from .exceptions import InputValidationError
from .isotropic import (
    _boundaries,
    _global_bounds_for_range,
    _legendre_nodes,
    _orders_for_segments,
    _validate_energy,
)
from .targets import CompositeSpectrum, PhotonSpectrum, discrete_lines


class PreparedTarget:
    """Store quadrature nodes for one fixed target photon field."""

    def __init__(
        self,
        target: PhotonSpectrum,
        E_min_MeV: float,
        *,
        preset: str = "balanced",
        n_energy: int | None = None,
        kernel=None,
    ):
        from .kernel import cached_kernel_preset

        if not isinstance(target, PhotonSpectrum):
            raise InputValidationError("ERROR: target must implement PhotonSpectrum")
        if preset not in {"fast", "balanced", "accurate"}:
            raise InputValidationError("ERROR: preset must be 'fast', 'balanced', or 'accurate'")

        self.target = target
        self.E_min_MeV = float(_validate_energy(E_min_MeV))
        self.preset = preset
        self.kernel = cached_kernel_preset(preset) if kernel is None else kernel
        if not callable(self.kernel):
            raise InputValidationError("ERROR: kernel must be callable")

        if isinstance(target, CompositeSpectrum):
            self.components = tuple(
                PreparedTarget(
                    item,
                    self.E_min_MeV,
                    preset=preset,
                    n_energy=n_energy,
                    kernel=self.kernel,
                )
                for item in target.components
            )
            self._eps = np.empty(0)
            self._weighted_density = np.empty(0)
            self._lines = None
            self._bounds = target.energy_bounds_keV
            return

        self.components = ()
        self._lines = discrete_lines(target)
        if self._lines is not None:
            self._eps = np.empty(0)
            self._weighted_density = np.empty(0)
            self._bounds = target.energy_bounds_keV
            return

        if n_energy is None:
            if bool(getattr(target, "has_infinite_high_energy_tail", False)):
                n_energy = {"fast": 512, "balanced": 1024, "accurate": 2048}[preset]
            else:
                n_energy = {"fast": 24, "balanced": 32, "accurate": 64}[preset]

        lo, hi = _global_bounds_for_range(target, self.E_min_MeV)
        boundaries = _boundaries(target, lo, hi)
        orders = _orders_for_segments(boundaries, int(n_energy))
        eps_list = []
        weight_list = []
        for left, right, order in zip(boundaries[:-1], boundaries[1:], orders):
            nodes, weights = _legendre_nodes(int(order))
            log_left, log_right = np.log(left), np.log(right)
            log_eps = 0.5 * (log_right - log_left) * nodes + 0.5 * (log_right + log_left)
            eps = np.exp(log_eps)
            jac = 0.5 * (log_right - log_left) * weights * eps
            eps_list.append(eps)
            weight_list.append(jac)

        self._eps = np.concatenate(eps_list)
        weights = np.concatenate(weight_list)
        density = np.asarray(target.number_density(self._eps), dtype=float)
        if density.shape != self._eps.shape or np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise InputValidationError("ERROR: target returned an invalid number-density array")
        self._weighted_density = weights * density
        self._bounds = (lo, hi)

    @property
    def representation_bytes(self) -> int:
        if self.components:
            return int(sum(item.representation_bytes for item in self.components))
        line_bytes = 0
        if self._lines is not None:
            line_bytes = int(self._lines[0].nbytes + self._lines[1].nbytes)
        return int(self._eps.nbytes + self._weighted_density.nbytes + line_bytes)

    @property
    def energy_bounds_keV(self) -> tuple[float, float]:
        return float(self._bounds[0]), float(self._bounds[1])

    def opacity(self, E_MeV, *, chunk_size: int = 4096):
        E = _validate_energy(E_MeV)
        if np.any(E < self.E_min_MeV * (1.0 - 1e-14)):
            raise InputValidationError(
                f"ERROR: prepared target is valid only for E_MeV >= {self.E_min_MeV:.12g}"
            )
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise InputValidationError("ERROR: chunk_size must be a positive integer")

        if self.components:
            result = np.sum(
                np.asarray([item.opacity(E, chunk_size=chunk_size) for item in self.components]),
                axis=0,
            )
            return float(result) if np.asarray(result).ndim == 0 else result

        flat = E.reshape(-1)
        out = np.zeros_like(flat)
        if self._lines is not None:
            energies, densities = self._lines
            z = flat[:, None] * 1.0e3 * energies[None, :] / ME_C2_KEV**2
            out += np.sum(densities[None, :] * np.asarray(self.kernel(z)), axis=1)

        if self._eps.size:
            for start in range(0, flat.size, int(chunk_size)):
                stop = min(start + int(chunk_size), flat.size)
                z = flat[start:stop, None] * 1.0e3 * self._eps[None, :] / ME_C2_KEV**2
                out[start:stop] += np.sum(
                    self._weighted_density[None, :] * np.asarray(self.kernel(z)), axis=1
                )

        out = out.reshape(E.shape)
        return float(out) if out.ndim == 0 else out


def prepare_target(
    target: PhotonSpectrum,
    E_min_MeV: float,
    *,
    preset: str = "balanced",
    n_energy: int | None = None,
    kernel=None,
) -> PreparedTarget:
    """Prepare a fixed target for repeated opacity evaluations."""
    return PreparedTarget(
        target,
        E_min_MeV,
        preset=preset,
        n_energy=n_energy,
        kernel=kernel,
    )
