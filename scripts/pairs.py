#!/usr/bin/env python3
"""Here we validate differential Breit--Wheeler pair production by independent routes."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

from thermal_bw import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    DiscreteLineSpectrum,
    GreybodySpectrum,
    PowerLawSpectrum,
    alpha_isotropic_gauss,
    angle_averaged_cross_section,
    pair_dsigma_dgamma,
    pair_gamma_bounds,
    pair_injection,
    pair_spectrum,
)
from thermal_bw.constants import C_CGS, ME_C2_KEV, PI, SIGMA_T
from thermal_bw.targets import integration_breakpoints

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
R0_SQ = 3.0 * SIGMA_T / (8.0 * PI)
ME_C2_MEV = ME_C2_KEV / 1.0e3


def cm_dsigma_domega(cos_theta: float, s_mandel: float) -> float:
    """Unpolarized CM differential Breit--Wheeler cross section, cm^2 sr^-1."""
    if s_mandel <= 4.0:
        return 0.0
    beta = math.sqrt(1.0 - 4.0 / s_mandel)
    cos2 = cos_theta * cos_theta
    sin2 = 1.0 - cos2
    numerator = 1.0 + 2.0 * beta * beta * sin2 - beta**4 - beta**4 * sin2 * sin2
    denominator = (1.0 - beta * beta * cos2) ** 2
    return R0_SQ * beta / s_mandel * numerator / denominator


def phase_space_histogram(
    x1: float,
    x2: float,
    edges,
    *,
    n_incoming: int = 112,
    n_scatter: int = 112,
    n_phi: int = 384,
):
    """Integrate the CM cross section and boost each lepton to the lab frame."""
    edges = np.asarray(edges, dtype=float)
    histogram = np.zeros(edges.size - 1, dtype=float)
    product = x1 * x2
    if product <= 1.0:
        return histogram

    # Pair threshold limits the angle between the two incident photons.
    mu_hi = 1.0 - 2.0 / product
    nodes_mu, weights_mu = leggauss(n_incoming)
    half_mu = 0.5 * (mu_hi + 1.0)
    mid_mu = 0.5 * (mu_hi - 1.0)
    mus = half_mu * nodes_mu + mid_mu
    wmus = half_mu * weights_mu

    cost, wcost = leggauss(n_scatter)
    phi = (np.arange(n_phi, dtype=float) + 0.5) * (2.0 * PI / n_phi)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    dphi = 2.0 * PI / n_phi

    lab_energy = x1 + x2
    p1_lab = np.array([0.0, 0.0, x1])

    for mu, wmu in zip(mus, wmus):
        sint = math.sqrt(max(0.0, 1.0 - mu * mu))
        p_total = np.array([x2 * sint, 0.0, x1 + x2 * mu])
        beta_vec = p_total / lab_energy
        beta2 = float(np.dot(beta_vec, beta_vec))
        s_mandel = 2.0 * product * (1.0 - mu)
        root_s = math.sqrt(s_mandel)
        boost_gamma = lab_energy / root_s
        energy_cm = 0.5 * root_s
        momentum_cm = math.sqrt(max(0.0, energy_cm * energy_cm - 1.0))

        beta = math.sqrt(beta2)
        beta_hat = beta_vec / beta
        beta_dot_p1 = float(np.dot(beta_vec, p1_lab))
        p1_cm = p1_lab + (
            (boost_gamma - 1.0) * beta_dot_p1 / beta2 - boost_gamma * x1
        ) * beta_vec
        ez = p1_cm / np.linalg.norm(p1_cm)

        # Complete an orthonormal basis around the incident photon in the CM frame.
        projection = beta_hat - float(np.dot(beta_hat, ez)) * ez
        norm_projection = float(np.linalg.norm(projection))
        if norm_projection > 1.0e-13:
            ex = projection / norm_projection
        else:
            trial = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(trial, ez))) > 0.9:
                trial = np.array([0.0, 1.0, 0.0])
            ex = trial - float(np.dot(trial, ez)) * ez
            ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)

        bdot_ez = float(np.dot(beta_hat, ez))
        bdot_ex = float(np.dot(beta_hat, ex))
        bdot_ey = float(np.dot(beta_hat, ey))
        collision_weight = wmu * 0.5 * (1.0 - mu)

        for ctheta, wc in zip(cost, wcost):
            stheta = math.sqrt(max(0.0, 1.0 - ctheta * ctheta))
            cross_section = cm_dsigma_domega(float(ctheta), s_mandel)
            cos_to_boost = (
                ctheta * bdot_ez
                + stheta * cosphi * bdot_ex
                + stheta * sinphi * bdot_ey
            )
            gamma_lab = boost_gamma * (
                energy_cm + beta * momentum_cm * cos_to_boost
            )
            weights = np.full(
                n_phi,
                collision_weight * wc * dphi * cross_section,
            )
            histogram += np.histogram(gamma_lab, bins=edges, weights=weights)[0]
    return histogram


def analytic_histogram(x1: float, x2: float, edges):
    """Integrate the reduced analytical pair kernel over the same bins."""
    edges = np.asarray(edges, dtype=float)
    result = np.zeros(edges.size - 1, dtype=float)
    gamma_lo, gamma_hi = pair_gamma_bounds(x1, x2)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        a = max(float(lo), gamma_lo)
        b = min(float(hi), gamma_hi)
        if b <= a:
            continue
        result[index] = integrate.quad(
            lambda gamma: pair_dsigma_dgamma(x1, x2, gamma),
            a,
            b,
            epsabs=1.0e-38,
            epsrel=2.0e-8,
            limit=500,
        )[0]
    return result


def kernel_checks():
    """Compare the reduced kernel with direct CM phase-space integration."""
    rows = []
    for x1, x2 in ((10.0, 0.3), (2.0, 2.0), (20.0, 0.2), (5.0, 1.0)):
        gamma_lo, gamma_hi = pair_gamma_bounds(x1, x2)
        edges = np.linspace(gamma_lo, gamma_hi, 61)
        direct = phase_space_histogram(x1, x2, edges)
        analytic = analytic_histogram(x1, x2, edges)
        total_direct = float(np.sum(direct))
        total_analytic = float(np.sum(analytic))
        reference = float(angle_averaged_cross_section(x1 * x2, n_angle=256))
        cdf_direct = np.cumsum(direct) / total_direct
        cdf_analytic = np.cumsum(analytic) / total_analytic
        rows.append(
            {
                "x1": x1,
                "x2": x2,
                "analytic_total_ratio": total_analytic / reference,
                "phase_total_ratio": total_direct / reference,
                "cdf_max_abs": float(np.max(np.abs(cdf_direct - cdf_analytic))),
                "hist_l1": float(
                    np.sum(np.abs(direct / total_direct - analytic / total_analytic))
                ),
            }
        )
    return rows


def stress_checks():
    """Check normalization and the first moment for extreme photon-energy ratios."""
    rows = []
    for product, ratio in (
        (1.00001, 1.0e6),
        (1.001, 1.0e8),
        (1.001, 1.0e12),
        (1.1, 1.0e12),
        (2.0, 1.0e12),
        (10.0, 1.0e13),
    ):
        x1 = math.sqrt(product * ratio)
        x2 = math.sqrt(product / ratio)
        gamma_lo, gamma_hi = pair_gamma_bounds(x1, x2)
        midpoint = 0.5 * (gamma_lo + gamma_hi)
        total = integrate.quad(
            lambda gamma: pair_dsigma_dgamma(x1, x2, gamma),
            gamma_lo,
            gamma_hi,
            points=[midpoint],
            epsabs=1.0e-40,
            epsrel=2.0e-6,
            limit=800,
        )[0]
        first = integrate.quad(
            lambda gamma: gamma * pair_dsigma_dgamma(x1, x2, gamma),
            gamma_lo,
            gamma_hi,
            points=[midpoint],
            epsabs=1.0e-34,
            epsrel=4.0e-6,
            limit=800,
        )[0]
        reference = float(angle_averaged_cross_section(product, n_angle=512))
        rows.append(
            {
                "product": product,
                "energy_ratio": ratio,
                "opacity_ratio": total / reference,
                "mean_ratio": (first / total) / (0.5 * (x1 + x2)),
            }
        )
    return rows


def cold_blackbody_check():
    """Check a TeV gamma ray on a 1-eV thermal target."""
    E_MeV = 1.0e6
    kT_keV = 1.0e-3
    target = BlackbodySpectrum(kT_keV)
    reference = float(
        alpha_isotropic_gauss(E_MeV, target, n_energy=192, n_angle=192)
    )
    upper = E_MeV + 80.0 * kT_keV / 1.0e3 - ME_C2_MEV
    value = integrate.quad(
        lambda electron_energy: pair_spectrum(
            E_MeV, electron_energy, target, n_energy=96
        ),
        ME_C2_MEV,
        upper,
        points=[0.5 * E_MeV],
        epsabs=0.0,
        epsrel=2.0e-4,
        limit=300,
    )[0]
    return {
        "E_MeV": E_MeV,
        "kT_keV": kT_keV,
        "opacity_ratio": value / reference,
    }

def target_moment_reference(E_MeV: float, target) -> tuple[float, float]:
    """Return opacity and one-lepton first moment from the total cross section."""
    x1 = E_MeV / ME_C2_MEV
    threshold_keV = ME_C2_KEV / x1
    lo, hi = target.energy_bounds_keV
    lo = max(lo, threshold_keV)
    if getattr(target, "has_infinite_high_energy_tail", False):
        hi = max(hi, threshold_keV + 80.0 * float(target.kT_keV))
    if hi <= lo:
        return 0.0, 0.0

    def base(log_epsilon: float) -> float:
        epsilon_keV = math.exp(log_epsilon)
        density = float(target.number_density(epsilon_keV))
        product = x1 * epsilon_keV / ME_C2_KEV
        sigma_bar = float(angle_averaged_cross_section(product, n_angle=192))
        return epsilon_keV * density * sigma_bar

    log_lo = math.log(lo)
    log_hi = math.log(hi)
    points = [
        math.log(value)
        for value in integration_breakpoints(target)
        if lo < value < hi
    ]
    alpha = integrate.quad(
        base, log_lo, log_hi, epsabs=0.0, epsrel=2e-7, limit=300, points=points or None
    )[0]
    moment = 0.5 * integrate.quad(
        lambda log_eps: base(log_eps) * (E_MeV + math.exp(log_eps) / 1.0e3),
        log_lo,
        log_hi,
        epsabs=0.0,
        epsrel=2e-7,
        limit=300,
        points=points or None,
    )[0]
    return alpha, moment


def target_checks():
    """Check target integration against opacity and first-energy moments."""
    rows = []

    line = DiscreteLineSpectrum([100.0], [1.0e12])
    electron_energy = np.linspace(0.511, 4.589, 1401)
    spectrum = pair_spectrum(5.0, electron_energy, line)
    total = float(integrate.simpson(spectrum, x=electron_energy))
    first = float(integrate.simpson(electron_energy * spectrum, x=electron_energy))
    alpha = float(alpha_isotropic_gauss(5.0, line, n_angle=256))
    rows.append(
        {
            "target": "line_100keV",
            "E_MeV": 5.0,
            "opacity_ratio": total / alpha,
            "moment_ratio": (first / total) / (0.5 * (5.0 + 0.1)),
        }
    )

    cases = [
        ("blackbody_50keV", BlackbodySpectrum(50.0), 2.0, 128, 800),
        ("blackbody_50keV", BlackbodySpectrum(50.0), 10.0, 128, 800),
        ("blackbody_50keV", BlackbodySpectrum(50.0), 30.0, 128, 800),
        ("greybody_50keV", GreybodySpectrum(50.0, 0.3), 5.0, 96, 500),
        (
            "cutoff_powerlaw",
            PowerLawSpectrum(1.0e10, 1.5, (1.0, 1000.0), cutoff_keV=400.0),
            5.0,
            96,
            500,
        ),
        (
            "broken_powerlaw",
            BrokenPowerLawSpectrum(
                1.0e10, 1.0, 2.2, 30.0, (1.0, 1000.0), cutoff_keV=500.0
            ),
            5.0,
            96,
            500,
        ),
    ]
    for name, target, E_MeV, order, n_electron in cases:
        _, hi = target.energy_bounds_keV
        if getattr(target, "has_infinite_high_energy_tail", False):
            threshold_keV = ME_C2_KEV / (E_MeV / ME_C2_MEV)
            hi = max(hi, threshold_keV + 80.0 * float(target.kT_keV))
        upper = E_MeV + hi / 1.0e3 - ME_C2_MEV
        electron_energy = np.linspace(ME_C2_MEV, upper, n_electron)
        spectrum = pair_spectrum(E_MeV, electron_energy, target, n_energy=order)
        total = float(integrate.simpson(spectrum, x=electron_energy))
        first = float(integrate.simpson(electron_energy * spectrum, x=electron_energy))
        alpha_ref, first_ref = target_moment_reference(E_MeV, target)
        rows.append(
            {
                "target": name,
                "E_MeV": E_MeV,
                "opacity_ratio": total / alpha_ref,
                "moment_ratio": first / first_ref,
            }
        )
    return rows


def shape_checks():
    """Quantify when the common equal-energy picture ceases to be representative."""
    target = BlackbodySpectrum(50.0)
    rows = []
    for E_MeV in (2.0, 10.0, 30.0, 100.0):
        upper = E_MeV + 4.0 - 0.511
        electron_energy = np.linspace(0.511, upper, 1200)
        spectrum = pair_spectrum(E_MeV, electron_energy, target, n_energy=128)
        total = float(integrate.simpson(spectrum, x=electron_energy))
        probability = spectrum / total
        half = 0.5 * E_MeV
        central = (electron_energy >= 0.9 * half) & (electron_energy <= 1.1 * half)
        central_fraction = float(
            integrate.simpson(np.where(central, probability, 0.0), x=electron_energy)
        )
        left = electron_energy <= half
        right = electron_energy >= half
        left_index = np.flatnonzero(left)[np.argmax(probability[left])]
        right_index = np.flatnonzero(right)[np.argmax(probability[right])]
        rows.append(
            {
                "E_MeV": E_MeV,
                "kT_keV": 50.0,
                "fraction_within_10pct_half": central_fraction,
                "left_peak_MeV": float(electron_energy[left_index]),
                "right_peak_MeV": float(electron_energy[right_index]),
            }
        )
    return rows


def injection_conservation_check():
    """Check pair-source number and energy against the absorbed gamma-ray rate."""
    target = DiscreteLineSpectrum([100.0], [2.0e10])
    gamma_energy = np.linspace(2.0, 8.0, 41)
    gamma_density = 3.0e4 * (gamma_energy / 2.0) ** -2.0
    electron_energy = np.linspace(ME_C2_MEV, 7.589, 700)

    single = pair_injection(gamma_energy, gamma_density, electron_energy, target)
    combined = pair_injection(
        gamma_energy,
        gamma_density,
        electron_energy,
        target,
        combined=True,
    )
    alpha = alpha_isotropic_gauss(gamma_energy, target, n_angle=192)

    number = float(integrate.trapezoid(single, x=electron_energy))
    number_reference = C_CGS * float(
        integrate.trapezoid(gamma_density * alpha, x=gamma_energy)
    )
    energy = float(
        integrate.trapezoid(electron_energy * combined, x=electron_energy)
    )
    # Each absorbed gamma ray also transfers the 0.1-MeV target photon to the pair.
    energy_reference = C_CGS * float(
        integrate.trapezoid(
            gamma_density * alpha * (gamma_energy + 0.1),
            x=gamma_energy,
        )
    )
    return {
        "number_ratio": number / number_reference,
        "energy_ratio": energy / energy_reference,
    }


def injection_demo():
    """Compare the exact pair source with the equal-split approximation."""
    target = BlackbodySpectrum(50.0)
    gamma_energy = np.logspace(np.log10(0.7), np.log10(100.0), 120)
    gamma_density = (gamma_energy / 10.0) ** -2.0 * np.exp(-gamma_energy / 100.0)
    electron_energy = np.logspace(np.log10(0.511), np.log10(50.0), 150)
    exact = pair_injection(
        gamma_energy,
        gamma_density,
        electron_energy,
        target,
        n_energy=96,
    )

    equal_split = np.zeros_like(electron_energy)
    parent_energy = 2.0 * electron_energy
    active = (parent_energy >= gamma_energy[0]) & (parent_energy <= gamma_energy[-1])
    parent_density = (parent_energy[active] / 10.0) ** -2.0 * np.exp(
        -parent_energy[active] / 100.0
    )
    parent_alpha = alpha_isotropic_gauss(
        parent_energy[active], target, n_energy=160, n_angle=128
    )
    equal_split[active] = 2.0 * C_CGS * parent_density * parent_alpha

    exact_total = float(integrate.trapezoid(exact, x=electron_energy))
    split_total = float(integrate.trapezoid(equal_split, x=electron_energy))
    exact_shape = exact / exact_total
    split_shape = equal_split / split_total
    core = (exact_shape > 0.01 * float(np.max(exact_shape))) & (split_shape > 0.0)
    max_core_difference = float(
        np.max(np.abs(split_shape[core] - exact_shape[core]) / exact_shape[core])
    )
    l1 = float(integrate.trapezoid(np.abs(exact_shape - split_shape), x=electron_energy))

    rows = [
        {
            "E_e_MeV": float(Ee),
            "exact_shape_MeV-1": float(q1),
            "equal_split_shape_MeV-1": float(q2),
        }
        for Ee, q1, q2 in zip(electron_energy, exact_shape, split_shape)
    ]
    summary = {
        "target": "blackbody_50keV",
        "gamma_spectrum": "E^-2 exp(-E/100 MeV)",
        "gamma_range_MeV": [float(gamma_energy[0]), float(gamma_energy[-1])],
        "number_ratio_exact_to_split": exact_total / split_total,
        "normalized_l1_distance": l1,
        "max_core_fractional_difference": max_core_difference,
    }
    return rows, summary


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kernels = kernel_checks()
    stress = stress_checks()
    cold_blackbody = cold_blackbody_check()
    targets = target_checks()
    shapes = shape_checks()
    injection_conservation = injection_conservation_check()
    injection_rows, injection = injection_demo()
    report = {
        "version": "1.0.0",
        "kernel": kernels,
        "stress": stress,
        "cold_blackbody": cold_blackbody,
        "targets": targets,
        "blackbody_shapes": shapes,
        "injection_conservation": injection_conservation,
        "injection_demo": injection,
    }
    (OUT / "pairs.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(OUT / "pairs.csv", kernels)
    write_csv(OUT / "pair_stress.csv", stress)
    write_csv(OUT / "pair_injection.csv", injection_rows)

    worst_total = max(
        max(abs(row["analytic_total_ratio"] - 1.0), abs(row["phase_total_ratio"] - 1.0))
        for row in kernels
    )
    worst_cdf = max(row["cdf_max_abs"] for row in kernels)
    stress_error = max(
        max(abs(row["opacity_ratio"] - 1.0), abs(row["mean_ratio"] - 1.0))
        for row in stress
    )
    cold_error = abs(cold_blackbody["opacity_ratio"] - 1.0)
    target_error = max(
        max(abs(row["opacity_ratio"] - 1.0), abs(row["moment_ratio"] - 1.0))
        for row in targets
    )
    injection_error = max(
        abs(injection_conservation["number_ratio"] - 1.0),
        abs(injection_conservation["energy_ratio"] - 1.0),
    )
    if worst_total >= 2.0e-5:
        raise RuntimeError("ERROR: pair-kernel normalization check failed")
    if worst_cdf >= 5.0e-4:
        raise RuntimeError("ERROR: pair-kernel phase-space shape check failed")
    if stress_error >= 1.0e-5:
        raise RuntimeError("ERROR: extreme-ratio pair-kernel conservation check failed")
    if cold_error >= 5.0e-4:
        raise RuntimeError("ERROR: cold-blackbody pair-spectrum check failed")
    if target_error >= 1.0e-3:
        raise RuntimeError("ERROR: target-integrated pair spectrum failed conservation checks")
    if injection_error >= 5.0e-3:
        raise RuntimeError("ERROR: pair-injection conservation check failed")

    print(f"pair kernel total error: {worst_total:.3e}")
    print(f"pair kernel CDF difference: {worst_cdf:.3e}")
    print(f"extreme-ratio conservation error: {stress_error:.3e}")
    print(f"1 TeV / 1 eV blackbody opacity error: {cold_error:.3e}")
    print(f"target conservation error: {target_error:.3e}")
    print(f"pair-injection conservation error: {injection_error:.3e}")
    print(
        "equal-split shape L1 / max core difference: "
        f"{injection['normalized_l1_distance']:.3e} / "
        f"{injection['max_core_fractional_difference']:.3e}"
    )


if __name__ == "__main__":
    main()
