"""Here I present code for a cached isotropic Breit--Wheeler angular kernel."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .exceptions import InputValidationError


# The presets change only the number of tabulated kernel values and the
# angular quadrature used to make them.  Interpolation is linear in log K.
_PRESETS = {
    "fast": (256, 160),
    "balanced": (512, 192),
    "accurate": (2048, 240),
}


class CachedAngleAveragedKernel:
    """Tabulated version of the isotropic angle-averaged cross section."""

    def __init__(
        self,
        *,
        n_nodes: int = 512,
        n_angle: int = 192,
        q_min: float = -10.0,
        q_max: float = 8.0,
        outside: str = "direct",
    ):
        if int(n_nodes) != n_nodes or n_nodes < 32:
            raise InputValidationError("ERROR: n_nodes must be an integer >= 32")
        if int(n_angle) != n_angle or n_angle < 16:
            raise InputValidationError("ERROR: n_angle must be an integer >= 16")
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
            raise InputValidationError("ERROR: require finite q_min < q_max")
        if outside not in {"direct", "raise"}:
            raise InputValidationError("ERROR: outside must be 'direct' or 'raise'")

        from .isotropic import angle_averaged_cross_section

        self.n_nodes = int(n_nodes)
        self.n_angle = int(n_angle)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.outside = outside

        self.q = np.linspace(self.q_min, self.q_max, self.n_nodes)
        z = 1.0 + 10.0**self.q
        kernel = np.asarray(
            angle_averaged_cross_section(z, n_angle=self.n_angle), dtype=float
        )
        if np.any(~np.isfinite(kernel)) or np.any(kernel <= 0.0):
            raise RuntimeError("ERROR: failed to construct cached angular kernel")
        self.log_kernel = np.log(kernel)

    @classmethod
    def from_preset(cls, preset: str = "balanced"):
        """Construct the fast, balanced, or accurate kernel preset."""
        if preset not in _PRESETS:
            raise InputValidationError("ERROR: preset must be 'fast', 'balanced', or 'accurate'")
        n_nodes, n_angle = _PRESETS[preset]
        return cls(n_nodes=n_nodes, n_angle=n_angle)

    @property
    def z_bounds(self) -> tuple[float, float]:
        return 1.0 + 10.0**self.q_min, 1.0 + 10.0**self.q_max

    @property
    def representation_bytes(self) -> int:
        return int(self.q.nbytes + self.log_kernel.nbytes)

    def __call__(self, product):
        z = np.asarray(product, dtype=float)
        if np.any(~np.isfinite(z)) or np.any(z < 0.0):
            raise InputValidationError("ERROR: product must be finite and non-negative")

        flat = z.reshape(-1)
        out = np.zeros_like(flat)
        active = flat > 1.0
        if np.any(active):
            z_active = flat[active]
            q = np.log10(z_active - 1.0)
            inside = (q >= self.q_min) & (q <= self.q_max)
            values = np.empty_like(z_active)
            values[inside] = np.exp(
                np.interp(q[inside], self.q, self.log_kernel)
            )

            if np.any(~inside):
                if self.outside == "raise":
                    lo, hi = self.z_bounds
                    raise InputValidationError(
                        f"ERROR: kernel product outside cached interval [{lo:.12g}, {hi:.12g}]"
                    )
                from .isotropic import angle_averaged_cross_section

                values[~inside] = angle_averaged_cross_section(
                    z_active[~inside], n_angle=self.n_angle
                )
            out[active] = values

        out = out.reshape(z.shape)
        return float(out) if out.ndim == 0 else out


@lru_cache(maxsize=3)
def cached_kernel_preset(preset: str = "balanced") -> CachedAngleAveragedKernel:
    """Return one cached kernel for a named preset."""
    return CachedAngleAveragedKernel.from_preset(preset)
