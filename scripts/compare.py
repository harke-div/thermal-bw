#!/usr/bin/env python3
"""Here we conduct a comparison of thermal-bw with compact numerical alternatives and public implementations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval

from bench import system_info, timed
from gamera import gamera_comparison
from literature import celli2017_alpha
from thermal_bw import (
    DEFAULT_PARAMS,
    SURROGATE_ETA_DOMAIN,
    alpha_blackbody_gauss,
    alpha_fit,
    sigma_breit_wheeler_s,
)
from thermal_bw.constants import ME_C2_KEV, SIGMA_T
from thermal_bw.approx import _reduced_surrogate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
AGNPY_COMMIT = "96f345a2c55acb3385a1d1f3aa722aaa0742f12c"


def energy_for_eta(eta, kT_keV=1.0):
    """Return gamma-ray energy in MeV for a chosen eta and temperature."""
    return np.asarray(eta, dtype=float) * ME_C2_KEV**2 / (1.0e3 * kT_keV)


def agnpy_sigma(s):
    """Evaluate the Breit--Wheeler formula using agnpy's invariant convention."""
    arr = np.asarray(s, dtype=float)
    out = np.zeros_like(arr)
    mask = arr >= 1.0
    beta = np.sqrt(1.0 - 1.0 / arr[mask])
    prefactor = 3.0 / 16.0 * SIGMA_T * (1.0 - beta**2)
    term = (3.0 - beta**4) * np.log((1.0 + beta) / (1.0 - beta))
    term -= 2.0 * beta * (2.0 - beta**2)
    out[mask] = prefactor * term
    return out


def map_eta(eta):
    """Map log eta over the validated interval to [-1, 1]."""
    lower, upper = SURROGATE_ETA_DOMAIN
    log_eta = np.log10(eta)
    return 2.0 * (log_eta - np.log10(lower)) / np.log10(upper / lower) - 1.0


def stats(prediction, reference):
    """Return relative-error statistics without an opacity cut."""
    relative = np.abs(np.asarray(prediction) / np.asarray(reference) - 1.0)
    return {
        "median_rel": float(np.median(relative)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "max_rel": float(np.max(relative)),
    }


def load_universal_validation():
    """Load the independent no-floor eta validation set."""
    rows = list(csv.DictReader((OUT / "universal.csv").open()))
    eta = np.asarray([float(row["eta"]) for row in rows])
    exact = np.asarray([float(row["exact_reduced_cm_inv_keV-3"]) for row in rows])
    return eta, exact


def main() -> None:
    """Run interpolation, literature, and cross-code comparisons."""
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "universal.csv").exists():
        from universal import main as universal_main

        universal_main()

    eta, exact = load_universal_validation()
    log_eta = np.log10(eta)
    log_exact = np.log10(exact)
    lower, upper = SURROGATE_ETA_DOMAIN

    # One-dimensional alternatives approximate the same universal blackbody function.
    regular = {}
    for size in (25, 50):
        nodes = np.geomspace(lower, upper, size)
        node_exact = alpha_blackbody_gauss(
            energy_for_eta(nodes), 1.0, n_angle=256, n_planck=384
        )
        regular[size] = (np.log10(nodes), np.log10(node_exact))

    train_eta = np.geomspace(lower, upper, 257)
    train_exact = alpha_blackbody_gauss(
        energy_for_eta(train_eta), 1.0, n_angle=256, n_planck=384
    )
    train_x = map_eta(train_eta)
    cheb = {
        degree: chebfit(train_x, np.log10(train_exact), degree)
        for degree in (8, 12)
    }

    energy = energy_for_eta(eta)
    methods = {
        "surrogate": (alpha_fit(energy, 1.0, bounds="ignore"), 6 * 8),
        "regular25": (
            10 ** np.interp(log_eta, regular[25][0], regular[25][1]),
            25 * 8,
        ),
        "regular50": (
            10 ** np.interp(log_eta, regular[50][0], regular[50][1]),
            50 * 8,
        ),
        "cheb8": (10 ** chebval(map_eta(eta), cheb[8]), cheb[8].nbytes),
        "cheb12": (10 ** chebval(map_eta(eta), cheb[12]), cheb[12].nbytes),
    }

    # Celli et al. state their approximation for x=1/eta <= 10.
    literature_mask = eta >= 0.1
    common_eta = eta[literature_mask]
    common_energy = energy[literature_mask]
    common_exact = exact[literature_mask]
    literature_predictions = {
        "thermal-bw": methods["surrogate"][0][literature_mask],
        "Celli2017": celli2017_alpha(common_energy, 1.0),
    }

    rng = np.random.default_rng(20260803)
    method_count = 1_000_000
    speed_eta = 10 ** rng.uniform(
        np.log10(lower), np.log10(upper), method_count
    )

    # Compare only the stored representations of the same universal function F(eta).
    # eta construction and the common (kT)^3 factor are excluded from every method,
    # so the timing isolates the actual analytic/interpolation representation.
    def regular_eval(size):
        values = np.interp(
            np.log10(speed_eta), regular[size][0], regular[size][1]
        )
        return np.power(10.0, values)

    def cheb_eval(degree):
        return np.power(10.0, chebval(map_eta(speed_eta), cheb[degree]))

    functions = {
        "surrogate": lambda: _reduced_surrogate(speed_eta, DEFAULT_PARAMS),
        "regular25": lambda: regular_eval(25),
        "regular50": lambda: regular_eval(50),
        "cheb8": lambda: cheb_eval(8),
        "cheb12": lambda: cheb_eval(12),
    }

    literature_count = 100_000
    common_speed_eta = 10 ** rng.uniform(
        np.log10(0.1), np.log10(upper), literature_count
    )
    common_speed_energy = energy_for_eta(common_speed_eta)
    literature_functions = {
        "thermal-bw": lambda: alpha_fit(common_speed_energy, 1.0, bounds="ignore"),
        "Celli2017": lambda: celli2017_alpha(common_speed_energy, 1.0),
    }

    rows = []
    for name, (prediction, storage) in methods.items():
        timing = timed(functions[name], repeats=21, warmups=4)
        rows.append(
            {
                "method": name,
                **stats(prediction, exact),
                "storage_bytes": int(storage),
                "rate_s-1": method_count / timing["median_s"],
                "timing_median_s": timing["median_s"],
                "timing_mad_s": timing["mad_s"],
                "timing_min_s": timing["min_s"],
                "timing_max_s": timing["max_s"],
                "timing_repeats": timing["repeats"],
                "timing_warmups": timing["warmups"],
            }
        )

    literature_rows = []
    for name, prediction in literature_predictions.items():
        timing = timed(literature_functions[name], repeats=7, warmups=1)
        literature_rows.append(
            {
                "method": name,
                "common_points": len(common_eta),
                "eta_range": f"0.1 <= eta <= {upper:g}",
                **stats(prediction, common_exact),
                "storage_bytes": 48 if name == "thermal-bw" else 24,
                "rate_s-1": literature_count / timing["median_s"],
                "timing_median_s": timing["median_s"],
                "timing_mad_s": timing["mad_s"],
                "published_accuracy": (
                    "this work" if name == "thermal-bw" else "~3% over 1e-10 <= x <= 10"
                ),
                "source": (
                    "thermal-bw 1.0.0"
                    if name == "thermal-bw"
                    else "Celli, Palladino & Vissani 2017, EPJC 77, 66, Eq. 8"
                ),
            }
        )

    # agnpy uses threshold s=1; thermal-bw uses a convention with threshold s=2.
    s = np.logspace(np.log10(1.0 + 1.0e-10), 6.0, 2000)
    ours = sigma_breit_wheeler_s(2.0 * s)
    external = agnpy_sigma(s)
    active = external > 0.0
    cross_section_rel = float(
        np.max(np.abs(ours[active] - external[active]) / external[active])
    )

    gamera_rows, gamera_report = gamera_comparison()

    for name, data in (("methods.csv", rows), ("literature.csv", literature_rows)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    with (OUT / "gamera.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gamera_rows[0]))
        writer.writeheader()
        writer.writerows(gamera_rows)
    (OUT / "gamera.json").write_text(json.dumps(gamera_report, indent=2) + "\n")

    report = {
        "version": "1.0.0",
        "blackbody_comparison": {
            "variable": "eta",
            "eta_domain": [lower, upper],
            "validation_points": len(eta),
            "opacity_floor": None,
            "timing_points": method_count,
            "timing_scope": (
                "stored representation of the universal reduced function F(eta); "
                "eta construction and the common (kT)^3 factor are excluded"
            ),
            "methods": rows,
        },
        "literature": literature_rows,
        "agnpy": {
            "version": "0.5.1",
            "source_commit": AGNPY_COMMIT,
            "cross_section_points": int(np.sum(active)),
            "cross_section_max_rel": cross_section_rel,
            "comparison": "source-expression convention check",
        },
        "gamera": gamera_report,
        "benchmark_system": system_info(),
    }
    (OUT / "compare.json").write_text(json.dumps(report, indent=2) + "\n")

    if cross_section_rel >= 1.0e-12:
        raise RuntimeError("ERROR: agnpy cross-section convention check failed")
    if gamera_report["convergence"]["both_max_rel"] >= 1.0e-5:
        raise RuntimeError("ERROR: GAMERA comparison reference did not converge")
    for row in gamera_report["summaries"]:
        if row["exact_angle_control"]["median_abs_rel"] >= 2.0e-3:
            raise RuntimeError("ERROR: GAMERA exact-angle control differs unexpectedly")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
