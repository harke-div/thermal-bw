#!/usr/bin/env python3
"""Here we validate the isotropic spectrum methods on representative target shapes.

The cases include smooth and broken continua, a Band-shaped callable, narrow
features, discrete lines, and a line-plus-continuum composite. The output is
written to ``results/spectra.csv`` and ``results/spectra.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from thermal_bw import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    CallableSpectrum,
    CompositeSpectrum,
    DiscreteLineSpectrum,
    PowerLawSpectrum,
    TabulatedSpectrum,
    alpha_isotropic_auto,
    alpha_isotropic_cached,
    alpha_isotropic_gauss,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
ENERGY_MEV = np.logspace(np.log10(0.5), np.log10(50.0), 40)


def band_target():
    """Return a Band-shaped local photon number-density spectrum."""
    alpha = -1.0
    beta = -2.3
    epeak = 300.0
    pivot = 100.0

    # The amplitude is a local differential number density, not an observed flux.
    amplitude = 1e12
    e0 = epeak / (2.0 + alpha)
    ebreak = (alpha - beta) * e0
    high_scale = (
        ((alpha - beta) * e0 / pivot) ** (alpha - beta) * np.exp(beta - alpha)
    )

    def density(epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        low = amplitude * (eps / pivot) ** alpha * np.exp(-eps / e0)
        high = amplitude * high_scale * (eps / pivot) ** beta
        return np.where(eps < ebreak, low, high)

    # The Band break is supplied explicitly so fixed quadrature can split there.
    return CallableSpectrum(density, (1.0, 5000.0), (ebreak,))


def narrow_targets():
    """Return tabulated and callable versions of one narrow feature."""
    center = 300.0
    width = 0.025
    log_center = np.log(center)

    def density(epsilon_keV):
        eps = np.asarray(epsilon_keV, dtype=float)
        peak = np.exp(-0.5 * ((np.log(eps) - log_center) / width) ** 2)
        return 1e8 + 1e12 * peak

    base = np.logspace(np.log10(1.0), np.log10(5000.0), 81)
    feature = np.exp(log_center + np.linspace(-0.15, 0.15, 61))
    nodes = np.unique(np.concatenate([base, feature]))
    table = TabulatedSpectrum(nodes, density(nodes))

    # Sharp features in a user callable are supplied as integration boundaries.
    points = tuple(np.exp(log_center + np.linspace(-0.15, 0.15, 13))[1:-1])
    callable_target = CallableSpectrum(density, (1.0, 5000.0), points)
    return table, callable_target


def targets():
    """Build the target fields used in the spectrum validation."""
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
    line = DiscreteLineSpectrum(
        energies_keV=(50.0, 500.0),
        number_densities_cm3=(2e12, 1e12),
    )
    table, narrow_callable = narrow_targets()
    composite = CompositeSpectrum((cutoff, line))

    return {
        "blackbody": BlackbodySpectrum(50.0),
        "cutoff_powerlaw": cutoff,
        "broken_powerlaw": broken,
        "band_callable": band_target(),
        "narrow_table": table,
        "narrow_callable": narrow_callable,
        "line": line,
        "composite": composite,
    }


def error_stats(value, reference):
    """Return relative errors where the reference opacity is non-zero."""
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    active = reference > 0.0
    relative = np.abs(value[active] - reference[active]) / reference[active]
    zero_error = np.abs(value[~active] - reference[~active])
    return {
        "points": int(reference.size),
        "active": int(active.sum()),
        "median_rel": float(np.median(relative)) if relative.size else 0.0,
        "p99_rel": float(np.quantile(relative, 0.99)) if relative.size else 0.0,
        "max_rel": float(np.max(relative)) if relative.size else 0.0,
        "zero_max_abs_cm_inv": float(np.max(zero_error)) if zero_error.size else 0.0,
    }


def main() -> None:
    """Run the spectrum tests and write the release validation tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    details = {}

    for name, target in targets().items():

        # Lines require only angular integration, so use a denser angular rule.
        angle_order = 512 if name == "line" else 192
        reference = alpha_isotropic_gauss(
            ENERGY_MEV,
            target,
            n_energy=256,
            n_angle=angle_order,
        )

        methods = {
            preset: alpha_isotropic_cached(ENERGY_MEV, target, preset=preset)
            for preset in ("fast", "balanced", "accurate")
        }

        methods["auto"] = alpha_isotropic_auto(
            ENERGY_MEV,
            target,
            rtol=1e-3,
            initial_order=24,
            max_order=384,
        )

        target_rows = {}
        for method, value in methods.items():
            stats = error_stats(value, reference)
            row = {"target": name, "method": method, **stats}
            rows.append(row)
            target_rows[method] = stats
        details[name] = target_rows

    smooth = {"blackbody", "cutoff_powerlaw", "broken_powerlaw", "band_callable"}
    fast_max = max(
        row["max_rel"]
        for row in rows
        if row["target"] in smooth and row["method"] == "fast"
    )
    auto_max = max(row["max_rel"] for row in rows if row["method"] == "auto")
    zero_max = max(row["zero_max_abs_cm_inv"] for row in rows)

    if fast_max >= 3e-3:
        raise RuntimeError("ERROR: fast preset exceeded 0.3 per cent on smooth targets")
    if auto_max >= 3e-3:
        raise RuntimeError("ERROR: automatic integration exceeded 0.3 per cent")
    if zero_max != 0.0:
        raise RuntimeError("ERROR: a method produced opacity where the reference is zero")

    with (OUT / "spectra.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "version": "1.0.0",
        "energy_MeV": ENERGY_MEV.tolist(),
        "reference": {
            "method": "fixed Gauss-Legendre",
            "n_energy": 256,
            "n_angle": 192,
            "line_n_angle": 512,
        },
        "targets": details,
        "target_note": (
            "Known narrow features in callable spectra are supplied as integration "
            "boundaries; the automatic method checks convergence of fixed quadrature."
        ),
        "limits": {
            "fast_smooth_max_rel": 3e-3,
            "auto_all_max_rel": 3e-3,
        },
    }
    (OUT / "spectra.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"smooth-target fast max rel: {fast_max:.8e}")
    print(f"all-target auto max rel: {auto_max:.8e}")


if __name__ == "__main__":
    main()
