#!/usr/bin/env python3
"""Here we validate the universal blackbody scaling and eta range in our surrogate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from thermal_bw import (
    SURROGATE_ETA_DOMAIN,
    alpha_blackbody_gauss,
    alpha_fit,
    thermal_eta,
)
from thermal_bw.constants import C_CGS, H_CGS, KEV_TO_ERG, ME_C2_KEV, PI, SIGMA_T
from literature import celli2017_alpha, celli2017_exact_f

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def energy_for_eta(eta, kT_keV):
    """Return gamma-ray energy in MeV for a chosen eta and temperature."""
    return np.asarray(eta, dtype=float) * ME_C2_KEV**2 / (1.0e3 * kT_keV)


def relative_stats(prediction, reference):
    """Return relative-error statistics."""
    relative = np.abs(np.asarray(prediction) / np.asarray(reference) - 1.0)
    return {
        "median_rel": float(np.median(relative)),
        "p95_rel": float(np.quantile(relative, 0.95)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "max_rel": float(np.max(relative)),
    }


def write_csv(path, rows):
    """Write result dictionaries to CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the dimensionless blackbody validation and write release results."""
    OUT.mkdir(parents=True, exist_ok=True)
    lower, upper = SURROGATE_ETA_DOMAIN

    # The fit uses logarithmically spaced eta nodes. Validation is performed on
    # logarithmic cell centres, so none of these 4096 points is a training node.
    edges = np.geomspace(lower, upper, 4097)
    eta = np.sqrt(edges[:-1] * edges[1:])
    energy = energy_for_eta(eta, 1.0)
    exact = alpha_blackbody_gauss(energy, 1.0, n_angle=256, n_planck=384)
    fit = alpha_fit(energy, 1.0, bounds="ignore")
    relative = np.abs(fit / exact - 1.0)
    log_error = np.abs(np.log10(fit) - np.log10(exact))
    worst = int(np.argmax(relative))

    rows = [
        {
            "eta": float(e),
            "x_inverse_eta": float(1.0 / e),
            "exact_reduced_cm_inv_keV-3": float(ref),
            "fit_reduced_cm_inv_keV-3": float(value),
            "relative_error": float(rel),
            "log_error_dex": float(loge),
        }
        for e, ref, value, rel, loge in zip(eta, exact, fit, relative, log_error)
    ]
    write_csv(OUT / "universal.csv", rows)

    peak = int(np.argmax(exact))
    validation = {
        "sampling": "4096 logarithmic cell centres, separate from fit nodes",
        "eta_domain": [lower, upper],
        "points": len(eta),
        "median_rel": float(np.median(relative)),
        "p95_rel": float(np.quantile(relative, 0.95)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "max_rel": float(relative[worst]),
        "max_log_dex": float(np.max(log_error)),
        "worst_eta": float(eta[worst]),
        "exact_peak_eta": float(eta[peak]),
    }

    # A scrambled Sobol sequence provides a second point set with different sampling.
    sampler = qmc.Sobol(d=1, scramble=True, seed=20260813)
    u = sampler.random_base2(m=11).ravel()
    sobol_eta = 10.0 ** (
        np.log10(lower) + u * (np.log10(upper) - np.log10(lower))
    )
    sobol_energy = energy_for_eta(sobol_eta, 1.0)
    sobol_exact = alpha_blackbody_gauss(
        sobol_energy, 1.0, n_angle=256, n_planck=384
    )
    sobol_fit = alpha_fit(sobol_energy, 1.0, bounds="ignore")
    sobol_validation = {
        "sampling": "2048-point scrambled Sobol sequence in log eta",
        "eta_domain": [lower, upper],
        "points": len(sobol_eta),
        **relative_stats(sobol_fit, sobol_exact),
        "worst_eta": float(
            sobol_eta[np.argmax(np.abs(sobol_fit / sobol_exact - 1.0))]
        ),
    }

    # Check that the fixed-Gauss reference is converged on independent eta values.
    eta_conv = eta[::8]
    energy_conv = energy_for_eta(eta_conv, 1.0)
    ref_low = alpha_blackbody_gauss(
        energy_conv, 1.0, n_angle=256, n_planck=384
    )
    ref_high = alpha_blackbody_gauss(
        energy_conv, 1.0, n_angle=320, n_planck=480
    )
    convergence = np.abs(ref_low / ref_high - 1.0)
    reference_convergence = {
        "points": len(eta_conv),
        "orders": [[256, 384], [320, 480]],
        "p99_rel": float(np.quantile(convergence, 0.99)),
        "max_rel": float(np.max(convergence)),
        "worst_eta": float(eta_conv[np.argmax(convergence)]),
    }

    # The exact T^3 scaling should hold at the same eta across very different T.
    eta_collapse = np.asarray([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0])
    temperatures = np.asarray([2.35e-7, 1.0e-3, 1.0, 50.0, 300.0, 1000.0])
    collapse_max = 0.0
    for eta_value in eta_collapse:
        energies = energy_for_eta(eta_value, temperatures)
        values = alpha_blackbody_gauss(
            energies, temperatures, n_angle=128, n_planck=224
        ) / temperatures**3
        collapse_max = max(
            collapse_max,
            float(np.max(np.abs(values / values[3] - 1.0))),
        )
    scaling = {
        "eta_values": eta_collapse.tolist(),
        "kT_keV": temperatures.tolist(),
        "max_reduced_coefficient_rel": collapse_max,
    }

    # Celli et al. give an independent exact beta-integral for the same function.
    celli_eta = np.geomspace(lower, upper, 20)
    celli_energy = energy_for_eta(celli_eta, 1.0)
    package_exact = alpha_blackbody_gauss(
        celli_energy, 1.0, n_angle=320, n_planck=480
    )
    reduced_prefactor = (8.0 * PI) / (H_CGS**3 * C_CGS**3) * KEV_TO_ERG**3
    celli_exact = np.asarray(
        [
            reduced_prefactor * (3.0 * SIGMA_T / 8.0) * celli2017_exact_f(1.0 / e)
            for e in celli_eta
        ]
    )
    celli_rel = np.abs(package_exact / celli_exact - 1.0)
    celli_rows = [
        {
            "eta": float(e),
            "x_inverse_eta": float(1.0 / e),
            "thermal_bw_exact_reduced": float(ours),
            "celli_exact_reduced": float(theirs),
            "relative_difference": float(rel),
        }
        for e, ours, theirs, rel in zip(
            celli_eta, package_exact, celli_exact, celli_rel
        )
    ]
    write_csv(OUT / "universal_celli.csv", celli_rows)
    celli_reference = {
        "points": len(celli_eta),
        "max_rel": float(np.max(celli_rel)),
        "p99_rel": float(np.quantile(celli_rel, 0.99)),
        "relation": "J(eta) = (3 sigma_T / 8) f(1/eta)",
        "source": "Celli, Palladino & Vissani 2017, EPJC 77, 66, Appendix A",
    }

    # Compare analytic approximations only where the Celli fit states x <= 10.
    common_eta = np.geomspace(0.1, upper, 1024)
    common_energy = energy_for_eta(common_eta, 1.0)
    common_exact = alpha_blackbody_gauss(
        common_energy, 1.0, n_angle=256, n_planck=384
    )
    ours_common = alpha_fit(common_energy, 1.0, bounds="ignore")
    celli_common = celli2017_alpha(common_energy, 1.0)
    literature = {
        "eta_domain": [0.1, upper],
        "x_domain": [1.0 / upper, 10.0],
        "points": len(common_eta),
        "thermal_bw": relative_stats(ours_common, common_exact),
        "Celli2017": relative_stats(celli_common, common_exact),
    }

    # A twofold eta shell is a stress test, not part of the validated interval.
    shell_eta = np.concatenate(
        [
            np.geomspace(lower / 2.0, lower, 512, endpoint=False),
            np.geomspace(upper, upper * 2.0, 512),
        ]
    )
    shell_energy = energy_for_eta(shell_eta, 1.0)
    shell_exact = alpha_blackbody_gauss(
        shell_energy, 1.0, n_angle=192, n_planck=288
    )
    shell_fit = alpha_fit(shell_energy, 1.0, bounds="ignore")
    shell_rel = np.abs(shell_fit / shell_exact - 1.0)
    shell = {
        "eta_range": [lower / 2.0, upper * 2.0],
        "points": len(shell_eta),
        "p95_rel": float(np.quantile(shell_rel, 0.95)),
        "p99_rel": float(np.quantile(shell_rel, 0.99)),
        "max_rel": float(np.max(shell_rel)),
        "worst_eta": float(shell_eta[np.argmax(shell_rel)]),
    }

    report = {
        "version": "1.0.0",
        "definition": "eta = E_gamma kT / (m_e c^2)^2; alpha = (kT)^3 F(eta)",
        "validation": validation,
        "sobol_validation": sobol_validation,
        "reference_convergence": reference_convergence,
        "temperature_scaling": scaling,
        "celli_exact_crosscheck": celli_reference,
        "published_approximation_comparison": literature,
        "literature_peak_check": {
            "exact_peak_eta": validation["exact_peak_eta"],
            "published_value": "zeta about 2 for the exact Nikishov curve",
            "source": "Voisin, Mottez & Bonazzola 2018, MNRAS 474, 1436",
        },
        "twofold_eta_shell": shell,
    }
    (OUT / "universal.json").write_text(json.dumps(report, indent=2) + "\n")

    if validation["max_rel"] >= 4.0e-3 or sobol_validation["max_rel"] >= 4.0e-3:
        raise RuntimeError("ERROR: surrogate exceeds 0.4% in the validated eta range")
    if reference_convergence["max_rel"] >= 5.0e-7:
        raise RuntimeError("ERROR: universal reference quadrature did not converge")
    if celli_reference["max_rel"] >= 1.0e-6:
        raise RuntimeError("ERROR: independent Celli exact-function check failed")
    if not 1.8 <= validation["exact_peak_eta"] <= 2.2:
        raise RuntimeError("ERROR: blackbody opacity peak is inconsistent with the literature")

    lines = [
        "thermal-bw 1.0.0",
        f"validated eta range: {lower:g} -- {upper:g}",
        f"interleaved validation max rel: {validation['max_rel']:.8e}",
        f"interleaved validation p99 rel: {validation['p99_rel']:.8e}",
        f"Sobol validation max rel: {sobol_validation['max_rel']:.8e}",
        f"reference convergence max rel: {reference_convergence['max_rel']:.8e}",
        f"Celli exact-function max rel: {celli_reference['max_rel']:.8e}",
        f"twofold eta-shell max rel: {shell['max_rel']:.8e}",
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
