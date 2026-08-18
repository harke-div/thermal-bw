"""Here we conduct a comparison of GAMERA opacity used by ``scripts/compare.py``."""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from thermal_bw import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    CallableSpectrum,
    PowerLawSpectrum,
    TabulatedSpectrum,
    angle_averaged_cross_section,
)
from thermal_bw.constants import ME_C2_KEV, SIGMA_T

GAMERA_COMMIT = "4d4753075ef4824b9d7994b7af33105e9bf56767"
GAMMA_TEST_RANGE_MEV = (0.5, 50.0)


def gamera_average_sigma(z):
    """Reproduce GAMERA's approximate isotropic angle-averaged cross section."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    mask = z >= 1.0
    x = z[mask]
    term = (
        (x + 0.5 * np.log(x) - 1.0 / 6.0 + 1.0 / (2.0 * x))
        * np.log(np.sqrt(x) + np.sqrt(x - 1.0))
        - (x + 4.0 / 9.0 - 1.0 / (9.0 * x)) * np.sqrt(1.0 - 1.0 / x)
    )
    out[mask] = 3.0 * SIGMA_T * term / (2.0 * x**2)
    return out


def double_peaked_density(epsilon_keV):
    """Smooth two-peak spectrum used only for the cross-code comparison."""
    epsilon = np.asarray(epsilon_keV, dtype=float)
    return (
        1.2e12 * np.exp(-0.5 * (np.log(epsilon / 35.0) / 0.42) ** 2)
        + 5.0e11 * np.exp(-0.5 * (np.log(epsilon / 260.0) / 0.30) ** 2)
        + 1.0e2
    )


def comparison_targets():
    """Return the four isotropic spectra used in the GAMERA comparison."""
    return {
        "blackbody_50keV": BlackbodySpectrum(50.0, u_min=1.0e-4, u_max=80.0),
        "powerlaw": PowerLawSpectrum(
            normalization=1.0e12,
            index=1.5,
            reference_keV=50.0,
            energy_bounds_keV=(5.0, 2000.0),
        ),
        "broken_powerlaw": BrokenPowerLawSpectrum(
            normalization_at_break=1.0e12,
            index_low=0.8,
            index_high=2.4,
            break_keV=80.0,
            energy_bounds_keV=(5.0, 2000.0),
        ),
        "double_peaked": CallableSpectrum(
            double_peaked_density,
            energy_bounds_keV=(2.0, 2000.0),
        ),
    }


def sample_target(target, size=401):
    """Sample one target on the stored grid used in the GAMERA comparison."""
    lower, upper = target.energy_bounds_keV
    energy = np.geomspace(lower, upper, size)
    density = np.asarray(target.number_density(energy), dtype=float)
    density = np.maximum(density, np.max(density) * 1.0e-300)
    return energy, density


def dense_tabulated_opacity(E_MeV, energy, density, *, n_energy=6001, n_angle=192):
    """Integrate a tabulated target with the thermal-bw angular kernel."""
    target = TabulatedSpectrum(energy, density)
    log_energy = np.linspace(np.log(energy[0]), np.log(energy[-1]), n_energy)
    epsilon = np.exp(log_energy)
    number_density = target.number_density(epsilon)
    z = E_MeV * 1.0e3 * epsilon / ME_C2_KEV**2
    kernel = angle_averaged_cross_section(z, n_angle=n_angle)
    return float(trapezoid(number_density * kernel * epsilon, log_energy))


def gamera_resampled_target(energy, density):
    """Reproduce GAMERA's log-space target interpolation and resampling."""
    log_energy = np.log10(energy)
    log_density = np.log10(density)
    size = len(energy)
    grid = log_energy[0] + np.arange(size) * (log_energy[-1] - log_energy[0]) / size
    epsilon = 10.0**grid
    number_density = 10.0 ** np.interp(grid, log_energy, log_density)
    return epsilon, number_density


def gamera_opacity(E_MeV, energy, density, *, exact_angle=False, n_angle=192):
    """Reproduce GAMERA's isotropic arbitrary-target opacity calculation."""
    epsilon, number_density = gamera_resampled_target(energy, density)
    z = E_MeV * 1.0e3 * epsilon / ME_C2_KEV**2
    if exact_angle:
        kernel = angle_averaged_cross_section(z, n_angle=n_angle)
    else:
        kernel = gamera_average_sigma(z)
    # GAMERA integrates a linear spline through these sampled values.
    return float(trapezoid(number_density * kernel, epsilon))


def relative_summary(relative):
    """Summarize one array of signed relative differences."""
    absolute = np.abs(relative)
    return {
        "median_abs_rel": float(np.median(absolute)),
        "p95_abs_rel": float(np.quantile(absolute, 0.95)),
        "p99_abs_rel": float(np.quantile(absolute, 0.99)),
        "max_abs_rel": float(np.max(absolute)),
        "median_signed_rel": float(np.median(relative)),
    }


def gamera_comparison():
    """Compare and decompose the isotropic opacity difference with GAMERA."""
    gamma_energy = np.geomspace(*GAMMA_TEST_RANGE_MEV, 31)
    rows = []
    summaries = []

    for name, target in comparison_targets().items():
        target_energy, target_density = sample_target(target)
        reference = np.asarray(
            [
                dense_tabulated_opacity(value, target_energy, target_density)
                for value in gamma_energy
            ]
        )
        full = np.asarray(
            [
                gamera_opacity(value, target_energy, target_density)
                for value in gamma_energy
            ]
        )
        control = np.asarray(
            [
                gamera_opacity(
                    value,
                    target_energy,
                    target_density,
                    exact_angle=True,
                )
                for value in gamma_energy
            ]
        )
        full_rel = full / reference - 1.0
        control_rel = control / reference - 1.0

        for values in zip(gamma_energy, reference, full, control, full_rel, control_rel):
            rows.append(
                {
                    "case": name,
                    "E_MeV": float(values[0]),
                    "thermal_bw_cm_inv": float(values[1]),
                    "gamera_cm_inv": float(values[2]),
                    "gamera_exact_angle_control_cm_inv": float(values[3]),
                    "gamera_over_thermal_bw_minus_1": float(values[4]),
                    "control_over_thermal_bw_minus_1": float(values[5]),
                }
            )

        summaries.append(
            {
                "case": name,
                "full": relative_summary(full_rel),
                "exact_angle_control": relative_summary(control_rel),
                "full_max_E_MeV": float(gamma_energy[np.argmax(np.abs(full_rel))]),
                "control_max_E_MeV": float(
                    gamma_energy[np.argmax(np.abs(control_rel))]
                ),
            }
        )

    # A smaller grid is enough for the convergence check and keeps reproduction quick.
    convergence_energy = np.geomspace(*GAMMA_TEST_RANGE_MEV, 7)
    angular = []
    target_sampling = []
    both = []
    for target in comparison_targets().values():
        energy, density = sample_target(target)
        for value in convergence_energy:
            baseline = dense_tabulated_opacity(value, energy, density)
            angular.append(
                abs(
                    dense_tabulated_opacity(
                        value, energy, density, n_energy=6001, n_angle=320
                    )
                    / baseline
                    - 1.0
                )
            )
            target_sampling.append(
                abs(
                    dense_tabulated_opacity(
                        value, energy, density, n_energy=12001, n_angle=192
                    )
                    / baseline
                    - 1.0
                )
            )
            both.append(
                abs(
                    dense_tabulated_opacity(
                        value, energy, density, n_energy=12001, n_angle=320
                    )
                    / baseline
                    - 1.0
                )
            )

    report = {
        "source": {
            "software": "GAMERA",
            "commit": GAMERA_COMMIT,
            "routine": "Radiation::ComputeAbsCoeff for an isotropic arbitrary target",
            "comparison": "reproduction",
        },
        "gamma_energy_MeV": list(GAMMA_TEST_RANGE_MEV),
        "gamma_points": len(gamma_energy),
        "target_samples": 401,
        "reference": {"target_samples": 6001, "angle_order": 192},
        "summaries": summaries,
        "convergence": {
            "cases": len(convergence_energy) * len(comparison_targets()),
            "angle_192_to_320_max_rel": float(np.max(angular)),
            "target_6001_to_12001_max_rel": float(np.max(target_sampling)),
            "both_max_rel": float(np.max(both)),
        },
    }
    return rows, report
