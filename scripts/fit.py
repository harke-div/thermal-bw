#!/usr/bin/env python3
"""Here we fit the blackbody surrogate directly as a function of eta."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from thermal_bw import DEFAULT_PARAMS, SURROGATE_ETA_DOMAIN, alpha_blackbody_gauss, alpha_model
from thermal_bw.constants import ME_C2_KEV

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Parameter order: log(A), log(c), log(d), p, a, q. The final exponent is fixed at -5.
BOUNDS = [
    (-12.0, 2.0),
    (-12.0, 6.0),
    (-16.0, 4.0),
    (0.2, 3.0),
    (0.1, 3.0),
    (0.5, 2.0),
]
TRAIN_POINTS = 2048


def energy_for_eta(eta, kT_keV=1.0):
    """Return gamma-ray energy in MeV for a chosen eta and temperature."""
    return np.asarray(eta, dtype=float) * ME_C2_KEV**2 / (1.0e3 * kT_keV)


def fit_stats(prediction, reference):
    """Return relative and logarithmic errors for positive reference values."""
    relative = np.abs(np.asarray(prediction) / np.asarray(reference) - 1.0)
    log_error = np.abs(np.log10(prediction) - np.log10(reference))
    return {
        "median_rel": float(np.median(relative)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "max_rel": float(np.max(relative)),
        "max_log_dex": float(np.max(log_error)),
    }


def main() -> None:
    """Fit the surrogate directly to the reduced blackbody function F(eta)."""
    OUT.mkdir(parents=True, exist_ok=True)
    lower, upper = SURROGATE_ETA_DOMAIN
    eta = np.geomspace(lower, upper, TRAIN_POINTS)
    energy = energy_for_eta(eta)

    # At kT=1 keV the opacity is numerically the reduced function alpha/(kT)^3.
    exact = alpha_blackbody_gauss(energy, 1.0, n_angle=320, n_planck=480)
    log_exact = np.log(exact)

    def objective(params):
        prediction = alpha_model(energy * 1.0e3, 1.0, params=params)
        residual = np.log(np.maximum(prediction, 1.0e-300)) - log_exact
        return float(np.max(np.abs(residual)))

    global_fit = differential_evolution(
        objective,
        BOUNDS,
        seed=20260813,
        popsize=24,
        maxiter=1100,
        tol=3.0e-10,
        polish=False,
        workers=1,
        updating="immediate",
    )
    local_fit = minimize(
        objective,
        global_fit.x,
        method="Nelder-Mead",
        options={"maxiter": 40000, "xatol": 5.0e-14, "fatol": 5.0e-14},
    )

    params = np.asarray(local_fit.x, dtype=float)
    prediction = alpha_model(energy * 1.0e3, 1.0, params=params)
    stats = fit_stats(prediction, exact)
    worst = int(np.argmax(np.abs(prediction / exact - 1.0)))

    report = {
        "version": "1.0.0",
        "model": "six-parameter blackbody surrogate with r=-5",
        "fit_variable": "eta = E_gamma kT / (m_e c^2)^2",
        "eta_domain": [lower, upper],
        "fit_points": TRAIN_POINTS,
        "opacity_floor": None,
        "reference": {"n_angle": 320, "n_planck": 480, "kT_keV": 1.0},
        "objective": "minimize maximum absolute logarithmic residual",
        "global_optimizer": {
            "method": "scipy.optimize.differential_evolution",
            "seed": 20260813,
            "bounds": BOUNDS,
        },
        "local_optimizer": "scipy.optimize.minimize(method='Nelder-Mead')",
        "params": params.tolist(),
        "metrics": stats,
        "worst_eta": float(eta[worst]),
        "packaged_difference": (params - DEFAULT_PARAMS).tolist(),
    }
    (OUT / "fit.json").write_text(json.dumps(report, indent=2) + "\n")

    if stats["max_rel"] >= 4.0e-3:
        raise RuntimeError("ERROR: universal fit exceeds the release error target")
    if np.max(np.abs(params - DEFAULT_PARAMS)) >= 5.0e-6:
        raise RuntimeError("ERROR: packaged coefficients do not match the reproduced fit")

    print(json.dumps({"params": params.tolist(), "metrics": stats}, indent=2))


if __name__ == "__main__":
    main()
