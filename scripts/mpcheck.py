#!/usr/bin/env python3
"""Here we conduct a comparison of package calculations with separate references of higher precision.

Blackbody opacities are recomputed by nested ``mpmath`` integration. We also evaluate
a selected set of differential pair-kernel values from the closed-form QED expression 
with 50-digit arithmetic, which includes strongly asymmetric photons.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mpmath as mp

from thermal_bw import (
    BrokenPowerLawSpectrum,
    PowerLawSpectrum,
    alpha_blackbody_gauss,
    alpha_exact,
    alpha_isotropic_adaptive,
    alpha_isotropic_gauss,
    pair_dsigma_dgamma,
    pair_gamma_bounds,
)

from thermal_bw.constants import ME_C2_KEV

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def mpmath_sigma(s):
    """Evaluate the unpolarized Breit--Wheeler cross section with mpmath."""
    if s <= 2:
        return mp.mpf("0")
    beta = mp.sqrt(1 - 2 / s)
    return (
        mp.mpf(3)
        / 16
        * mp.mpf("6.6524587321e-25")
        * (1 - beta**2)
        * (
            (3 - beta**4) * mp.log((1 + beta) / (1 - beta))
            - 2 * beta * (2 - beta**2)
        )
    )


def mpmath_opacity(energy_MeV, temp_keV):
    """Compute one blackbody opacity using independent nested integration."""
    c = mp.mpf("2.99792458e10")
    h = mp.mpf("6.62607015e-27")
    kev = mp.mpf("1.602176634e-9")
    mass = mp.mpf("511")
    energy = mp.mpf(str(energy_MeV))
    temp = mp.mpf(str(temp_keV))

    eta = (energy * 1000 / mass) * (temp / mass)

    # u = epsilon/(kT). The lower limit is the head-on pair threshold.
    lower = 1 / eta
    upper = max(mp.mpf(80), lower + 80)
    prefactor = 8 * mp.pi / (h**3 * c**3) * (temp * kev) ** 3

    def integrand(u):
        t0 = 2 / (eta * u)
        # t = 1 - cos(psi); this is the isotropic collision-angle integral.
        angular = mp.quad(
            lambda t: mp.mpf("0.5") * t * mpmath_sigma(eta * u * t),
            [t0, 2],
        )
        return u**2 / mp.expm1(u) * angular

    # Splitting the threshold/Wien-tail region makes adaptive integration more stable.
    breaks = [
        lower,
        lower + mp.mpf("0.25"),
        lower + 1,
        lower + 4,
        lower + 12,
        lower + 32,
        upper,
    ]
    intervals = sorted({x for x in breaks if lower <= x <= upper})
    return prefactor * mp.quad(integrand, intervals)



def mpmath_isotropic_opacity(energy_MeV, density, bounds_keV, breakpoints_keV=()):
    """Compute finite-spectrum isotropic opacity by independent nested integration."""
    mass = mp.mpf("511")
    energy_keV = mp.mpf(str(energy_MeV)) * 1000
    lower = max(mp.mpf(str(bounds_keV[0])), mass**2 / energy_keV)
    upper = mp.mpf(str(bounds_keV[1]))
    if lower >= upper:
        return mp.mpf("0")

    def angular(epsilon):
        z = energy_keV * epsilon / mass**2
        if z <= 1:
            return mp.mpf("0")
        return mp.quad(
            lambda t: mp.mpf("0.5") * t * mpmath_sigma(z * t),
            [2 / z, 2],
        )

    points = [lower]
    points.extend(
        mp.mpf(str(value))
        for value in breakpoints_keV
        if lower < mp.mpf(str(value)) < upper
    )
    points.append(upper)
    return mp.quad(lambda epsilon: density(epsilon) * angular(epsilon), points)


def general_spectrum_reference_rows():
    """Check two finite non-thermal target spectra with mpmath integration."""
    cutoff = PowerLawSpectrum(
        normalization=1e12,
        index=1.5,
        reference_keV=100.0,
        cutoff_keV=500.0,
        energy_bounds_keV=(1.0, 5000.0),
    )
    broken = BrokenPowerLawSpectrum(
        normalization_at_break=1e12,
        index_low=1.0,
        index_high=2.3,
        break_keV=100.0,
        cutoff_keV=2000.0,
        energy_bounds_keV=(1.0, 5000.0),
    )

    norm = mp.mpf("1e12")
    reference = mp.mpf("100")

    def cutoff_density(epsilon):
        return (
            norm
            * (epsilon / reference) ** mp.mpf("-1.5")
            * mp.exp(-epsilon / mp.mpf("500"))
        )

    def broken_density(epsilon):
        ratio = epsilon / reference
        index = mp.mpf("-1.0") if epsilon < reference else mp.mpf("-2.3")
        return norm * ratio**index * mp.exp(-epsilon / mp.mpf("2000"))

    cases = (
        ("cutoff_powerlaw", cutoff, cutoff_density, ()),
        ("broken_powerlaw", broken, broken_density, (100.0,)),
    )
    rows = []
    for name, target, density, breakpoints in cases:
        for energy in (0.5, 2.0, 10.0, 50.0):
            start = time.perf_counter()
            high = mpmath_isotropic_opacity(
                energy, density, target.energy_bounds_keV, breakpoints
            )
            elapsed = time.perf_counter() - start
            high_float = float(high)
            adaptive = alpha_isotropic_adaptive(
                energy, target, epsrel=1e-9, angle_epsrel=1e-9
            )
            gauss = alpha_isotropic_gauss(
                energy, target, n_energy=384, n_angle=320
            )
            rows.append(
                {
                    "target": name,
                    "E_MeV": energy,
                    "mpmath_cm_inv": mp.nstr(high, 22),
                    "adaptive_rel": abs(adaptive - high_float) / high_float,
                    "gauss_rel": abs(gauss - high_float) / high_float,
                    "seconds": elapsed,
                }
            )
    return rows


def _mp_i_term(alpha, product, c_value):
    if c_value > 0:
        return mp.asinh(alpha * mp.sqrt(c_value / product)) / mp.sqrt(c_value)
    return mp.asin(alpha * mp.sqrt((-c_value) / product)) / mp.sqrt(-c_value)


def _mp_h_term(alpha, x1, x2, gamma_e, branch):
    product = x1 * x2
    if branch > 0:
        c_value = (x1 - gamma_e) ** 2 - 1
        d_value = x1**2 + product + gamma_e * (x2 - x1)
    else:
        c_value = (x2 - gamma_e) ** 2 - 1
        d_value = x2**2 + product - gamma_e * (x2 - x1)

    if abs(c_value) < mp.mpf("1e-40"):
        return (
            (alpha**3 / 12 - alpha * d_value / 8) / product ** mp.mpf("1.5")
            + (alpha**3 / 6 + alpha / 2 + 1 / (4 * alpha)) / mp.sqrt(product)
        )

    root = mp.sqrt(product + c_value * alpha**2)
    i_value = _mp_i_term(alpha, product, c_value)
    return (
        -alpha / (8 * root) * (d_value / product + 2 / c_value)
        + mp.mpf("0.25") * (2 - (product - 1) / c_value) * i_value
        + root / 4 * (alpha / c_value + 1 / (alpha * product))
    )


def mpmath_pair_kernel(x1_value, x2_value, gamma_value):
    """Evaluate one differential pair cross section with 50-digit arithmetic."""
    x1 = mp.mpf(str(x1_value))
    x2 = mp.mpf(str(x2_value))
    gamma_e = mp.mpf(str(gamma_value))
    product = x1 * x2
    energy = x1 + x2

    y_value = gamma_e * (energy - gamma_e) + 1
    discriminant = (y_value - energy) * (y_value + energy)
    root_disc = mp.sqrt(discriminant)
    alpha_a = mp.sqrt((y_value + root_disc) / 2)
    alpha_b = energy / (2 * alpha_a)
    alpha_lo = max(mp.mpf(1), alpha_b)
    alpha_hi = min(mp.sqrt(product), alpha_a)

    def primitive(alpha):
        return (
            mp.sqrt(energy**2 - 4 * alpha**2) / 4
            + _mp_h_term(alpha, x1, x2, gamma_e, 1)
            + _mp_h_term(alpha, x1, x2, gamma_e, -1)
        )

    sigma_t = mp.mpf("6.6524587321e-25")
    return mp.mpf("1.5") * sigma_t / product**2 * (
        primitive(alpha_hi) - primitive(alpha_lo)
    )


def pair_reference_rows():
    """Check ordinary and strongly asymmetric differential pair kinematics."""
    rows = []
    cases = [
        (10.0, 0.3, 0.5),
        (mp.sqrt(mp.mpf("1.001e12")), mp.sqrt(mp.mpf("1.001e-12")), 0.1),
        (mp.sqrt(mp.mpf("1.001e12")), mp.sqrt(mp.mpf("1.001e-12")), 0.5),
        (mp.sqrt(mp.mpf("1.1e12")), mp.sqrt(mp.mpf("1.1e-12")), 0.9),
    ]
    for x1_mp, x2_mp, fraction in cases:
        x1 = float(x1_mp)
        x2 = float(x2_mp)
        lower, upper = pair_gamma_bounds(x1, x2)
        gamma_e = lower + fraction * (upper - lower)
        high = mpmath_pair_kernel(x1, x2, gamma_e)
        value = pair_dsigma_dgamma(x1, x2, gamma_e)
        high_float = float(high)
        rows.append(
            {
                "x1": x1,
                "x2": x2,
                "fraction": fraction,
                "gamma_e": gamma_e,
                "mpmath_cm2": mp.nstr(high, 22),
                "package_rel": abs(value - high_float) / high_float,
            }
        )
    return rows

def main() -> None:
    """Run the extended-precision checks and write ``mpmath.json``."""
    OUT.mkdir(parents=True, exist_ok=True)
    mp.mp.dps = 30
    opacity_rows = []

    # Probe the universal blackbody function from the Wien tail through the interior.
    temp = 1.0
    opacity_eta = (0.01, 0.03, 0.3, 3.0)
    opacity_points = [(eta * ME_C2_KEV**2 / 1.0e3, temp) for eta in opacity_eta]
    for energy, temp in opacity_points:
        start = time.perf_counter()
        high = mpmath_opacity(energy, temp)
        elapsed = time.perf_counter() - start
        value = float(high)
        adaptive = alpha_exact(energy, temp, epsrel=1e-9, angle_epsrel=1e-9)
        gauss = alpha_blackbody_gauss(energy, temp, n_angle=160, n_planck=320)
        opacity_rows.append(
            {
                "E_MeV": energy,
                "kT_keV": temp,
                "mpmath_cm_inv": mp.nstr(high, 22),
                "adaptive_rel": abs(adaptive - value) / value,
                "gauss_rel": abs(gauss - value) / value,
                "seconds": elapsed,
            }
        )

    general_rows = general_spectrum_reference_rows()

    mp.mp.dps = 50
    pair_rows = pair_reference_rows()
    report = {
        "blackbody_opacity": opacity_rows,
        "general_isotropic_opacity": general_rows,
        "pair_kernel": pair_rows,
    }
    (OUT / "mpmath.json").write_text(json.dumps(report, indent=2) + "\n")

    opacity_worst = max(
        max(row["adaptive_rel"], row["gauss_rel"]) for row in opacity_rows
    )
    general_worst = max(
        max(row["adaptive_rel"], row["gauss_rel"]) for row in general_rows
    )
    pair_worst = max(row["package_rel"] for row in pair_rows)
    if opacity_worst >= 1e-5:
        raise RuntimeError(
            "ERROR: package integrators differ from the mpmath opacity reference by at least 1e-5"
        )
    if general_worst >= 1e-5:
        raise RuntimeError(
            "ERROR: general-spectrum integrators differ from the mpmath reference by at least 1e-5"
        )
    if pair_worst >= 1e-6:
        raise RuntimeError(
            "ERROR: differential pair kernel differs from the 50-digit reference by at least 1e-6"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
