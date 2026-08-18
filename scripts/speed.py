#!/usr/bin/env python3
"""Here we benchmark scalar, array, and chunked opacity evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from bench import system_info, timed
from thermal_bw import (
    BlackbodySpectrum,
    TargetOpacityTable,
    alpha_exact,
    alpha_fit,
    alpha_isotropic_cached,
    alpha_isotropic_gauss,
    pair_injection,
    pair_spectrum,
    prepare_target,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


BENCHMARK_ENERGY_RANGE_MEV = (1.0, 100.0)

def main() -> None:
    """Run the benchmark and write timing and hardware records."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = BlackbodySpectrum(50.0)
    energy_range = BENCHMARK_ENERGY_RANGE_MEV
    prepared = prepare_target(target, energy_range[0], preset="balanced")
    table = TargetOpacityTable.build(target, energy_range, rtol=1e-3)
    rows = []

    # Array-size scaling distinguishes scalar overhead from batched throughput.
    for size in (1, 16, 256, 4096, 16384):
        energy = np.logspace(*np.log10(energy_range), size)
        methods = {
            "surrogate": lambda: alpha_fit(energy, 50.0, bounds="ignore"),
            "cached_fast": lambda: alpha_isotropic_cached(
                energy, target, preset="fast"
            ),
            "cached_balanced": lambda: alpha_isotropic_cached(
                energy, target, preset="balanced"
            ),
            "cached_accurate": lambda: alpha_isotropic_cached(
                energy, target, preset="accurate"
            ),
            "prepared": lambda: prepared.opacity(energy),
            "table": lambda: table(energy),
        }
        if size <= 4096:
            methods["gauss"] = lambda: alpha_isotropic_gauss(
                energy, target, n_energy=128, n_angle=96
            )

        for name, function in methods.items():
            repeats = 11 if size <= 256 else 7
            warmups = 2
            if name == "gauss":
                repeats = 3 if size <= 256 else 1
                warmups = 1
            result = timed(function, repeats=repeats, warmups=warmups)
            rows.append(
                {
                    "method": name,
                    "size": size,
                    **result,
                    "rate_s-1": size / result["median_s"],
                }
            )

    adaptive = timed(
        lambda: alpha_exact(2.0, 50.0, epsrel=3e-4), repeats=5, warmups=1
    )
    rows.append(
        {
            "method": "adaptive",
            "size": 1,
            **adaptive,
            "rate_s-1": 1.0 / adaptive["median_s"],
        }
    )

    # Differential-pair timings use the same 50-keV blackbody target.
    pair_rows = []
    for size in (1, 16, 64, 256):
        if size == 1:
            electron_energy = np.asarray([5.0])
        else:
            electron_energy = np.linspace(0.511, 13.489, size)
        result = timed(
            lambda electron_energy=electron_energy: pair_spectrum(
                10.0, electron_energy, target, n_energy=96
            ),
            repeats=5 if size <= 64 else 3,
            warmups=1,
        )
        pair_rows.append(
            {
                "operation": "pair_spectrum",
                "n_gamma": 1,
                "n_electron": size,
                "kernel_points": size,
                **result,
                "rate_s-1": size / result["median_s"],
            }
        )

    gamma_energy = np.logspace(0.0, 2.0, 32)
    gamma_density = 1.0e6 * (gamma_energy / 10.0) ** -2.0
    electron_energy = np.logspace(np.log10(0.511), np.log10(50.0), 64)
    result = timed(
        lambda: pair_injection(
            gamma_energy, gamma_density, electron_energy, target, n_energy=96
        ),
        repeats=3,
        warmups=1,
    )
    pair_rows.append(
        {
            "operation": "pair_injection",
            "n_gamma": len(gamma_energy),
            "n_electron": len(electron_energy),
            "kernel_points": len(gamma_energy) * len(electron_energy),
            **result,
            "rate_s-1": len(gamma_energy) * len(electron_energy) / result["median_s"],
        }
    )

    # Chunk scaling records the memory and timing of the cached method.
    chunk_rows = []
    energy = np.logspace(*np.log10(energy_range), 16384)
    for chunk in (64, 256, 1024, 2048, 4096, 8192, 16384):
        result = timed(
            lambda chunk=chunk: alpha_isotropic_cached(
                energy, target, preset="balanced", chunk_size=chunk
            ),
            repeats=5,
            warmups=1,
        )
        chunk_rows.append(
            {
                "chunk_size": chunk,
                **result,
                "rate_s-1": len(energy) / result["median_s"],
            }
        )

    for name, data in (
        ("speed.csv", rows),
        ("chunks.csv", chunk_rows),
        ("pair_speed.csv", pair_rows),
    ):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    report = {
        "version": "1.0.0",
        "note": "Rerun on dedicated hardware for publication timing values.",
        "system": system_info(),
        "table_nodes": table.n_nodes,
        "table_bytes": table.representation_bytes,
        "table_validation_max_rel": table.validation_max_relative_error,
        "prepared_bytes": prepared.representation_bytes,
        "rows": rows,
        "pair_rows": pair_rows,
        "chunks": chunk_rows,
    }
    (OUT / "speed.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
