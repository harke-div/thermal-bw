from __future__ import annotations

import warnings

import numpy as np
import pytest

from thermal_bw.constants import ME_C2_KEV
from thermal_bw import (
    BlackbodySpectrum,
    CompositeSpectrum,
    GreybodySpectrum,
    OutOfDomainError,
    OutOfDomainWarning,
    PowerLawSpectrum,
    SURROGATE_ETA_DOMAIN,
    TabulatedSpectrum,
    alpha_blackbody_gauss,
    alpha_exact,
    alpha_fit,
    alpha_model,
    alpha_isotropic_adaptive,
    alpha_isotropic_gauss,
    angle_averaged_cross_section,
    sigma_breit_wheeler_s,
    thermal_eta,
    within_validated_domain,
)


def test_cross_section_threshold_and_positive_branch():
    values = sigma_breit_wheeler_s(np.array([0.0, 2.0, 2.1, 10.0]))
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert np.all(values[2:] > 0.0)


def test_angle_averaged_kernel_threshold():
    values = angle_averaged_cross_section(np.array([0.5, 1.0, 2.0]), n_angle=96)
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] > 0.0


@pytest.mark.parametrize(
    "energy,temp",
    [(0.5, 10.0), (1.0, 20.0), (2.0, 50.0), (10.0, 100.0), (30.0, 300.0)],
)
def test_independent_blackbody_gauss_matches_adaptive(energy, temp):
    gauss = alpha_blackbody_gauss(energy, temp, n_angle=64, n_planck=96)
    adaptive = alpha_exact(energy, temp, epsrel=1e-8)
    assert gauss == pytest.approx(adaptive, rel=8e-5)


def test_general_isotropic_blackbody_matches_specialized_gauss():
    target = BlackbodySpectrum(50.0)
    energies = np.array([0.5, 1.0, 2.0, 10.0])
    general = alpha_isotropic_gauss(energies, target, n_energy=160, n_angle=96)
    specialized = alpha_blackbody_gauss(energies, 50.0, n_angle=96, n_planck=128)
    assert np.allclose(general, specialized, rtol=2e-3, atol=0.0)


def test_general_adaptive_matches_general_gauss():
    target = BlackbodySpectrum(50.0)
    adaptive = alpha_isotropic_adaptive(2.0, target, epsrel=2e-7)
    gauss = alpha_isotropic_gauss(2.0, target, n_energy=160, n_angle=128)
    assert gauss == pytest.approx(adaptive, rel=3e-4)


def test_greybody_scales_linearly():
    base = alpha_isotropic_gauss(2.0, BlackbodySpectrum(50.0))
    diluted = alpha_isotropic_gauss(2.0, GreybodySpectrum(50.0, dilution=1e-3))
    assert diluted == pytest.approx(base * 1e-3, rel=1e-12)


def test_composite_additivity():
    first = GreybodySpectrum(30.0, dilution=0.2)
    second = GreybodySpectrum(80.0, dilution=0.05)
    composite = CompositeSpectrum([first, second])
    energy = 3.0
    total = alpha_isotropic_gauss(energy, composite)
    separate = alpha_isotropic_gauss(energy, first) + alpha_isotropic_gauss(energy, second)
    assert total == pytest.approx(separate, rel=2e-10)


def test_tabulated_spectrum_reproduces_powerlaw():
    spectrum = PowerLawSpectrum(
        normalization=1e12,
        index=1.5,
        energy_bounds_keV=(0.1, 1000.0),
        cutoff_keV=500.0,
    )
    grid = np.logspace(-1, 3, 200)
    tabulated = TabulatedSpectrum(grid, spectrum.number_density(grid))
    exact = alpha_isotropic_gauss(5.0, spectrum, n_energy=160, n_angle=96)
    interpolated = alpha_isotropic_gauss(5.0, tabulated, n_energy=160, n_angle=96)
    assert interpolated == pytest.approx(exact, rel=2e-4)


def test_general_isotropic_threshold_at_target_upper_bound_is_zero():
    target = PowerLawSpectrum(1e12, 1.5, (1.0, 1000.0))
    threshold_energy = (ME_C2_KEV**2) / (1000.0 * 1.0e3)
    value = alpha_isotropic_gauss(threshold_energy, target, n_energy=64, n_angle=64)
    assert np.isfinite(value)
    assert value >= 0.0


def test_surrogate_warns_or_raises_outside_domain():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = alpha_fit(0.01, 5.0)
    assert value >= 0.0
    warning = next(
        item for item in caught if issubclass(item.category, OutOfDomainWarning)
    )
    assert "validated dimensionless range" in str(warning.message)
    assert "alpha_blackbody_gauss or alpha_exact" in str(warning.message)
    with pytest.raises(OutOfDomainError):
        alpha_fit(0.01, 5.0, bounds="raise")


def test_surrogate_inside_domain_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = alpha_fit(2.0, 50.0)
    assert value > 0.0
    assert not caught


def test_surrogate_domain_follows_eta():
    eta = 1.0
    temperatures = np.array([100.0, 1.0e-3])
    energies = eta * ME_C2_KEV**2 / (1.0e3 * temperatures)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = alpha_fit(energies, temperatures)

    assert not caught
    assert np.all(within_validated_domain(energies, temperatures))
    assert thermal_eta(energies, temperatures) == pytest.approx([eta, eta])
    reduced = values / temperatures**3
    assert reduced[0] == pytest.approx(reduced[1], rel=2e-14)


def test_surrogate_eta_boundaries():
    lower, upper = SURROGATE_ETA_DOMAIN
    energies = np.array([lower, upper]) * ME_C2_KEV**2 / 1.0e3
    assert np.all(within_validated_domain(energies, 1.0))

    outside = np.array([lower * 0.99, upper * 1.01]) * ME_C2_KEV**2 / 1.0e3
    assert not np.any(within_validated_domain(outside, 1.0))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        alpha_fit(-1.0, 50.0)
    with pytest.raises(ValueError):
        alpha_blackbody_gauss(1.0, 0.0)


def test_cached_kernel_matches_direct_random_points():
    from thermal_bw import CachedAngleAveragedKernel

    cached = CachedAngleAveragedKernel(n_nodes=2048, n_angle=240)
    rng = np.random.default_rng(20260803)
    q = rng.uniform(-9.5, 7.5, 2000)
    z = 1.0 + 10.0**q
    direct = angle_averaged_cross_section(z, n_angle=256)
    approx = cached(z)
    rel = np.abs(approx - direct) / direct
    assert np.quantile(rel, 0.99) < 5e-5
    assert np.max(rel) < 1e-4


def test_cached_isotropic_matches_direct_gauss_blackbody():
    from thermal_bw import alpha_isotropic_cached

    target = BlackbodySpectrum(50.0)
    energies = np.logspace(np.log10(0.4), np.log10(25.0), 20)
    direct = alpha_isotropic_gauss(energies, target, n_energy=160, n_angle=128)
    cached = alpha_isotropic_cached(energies, target, n_energy=160)
    assert np.allclose(cached, direct, rtol=5e-5, atol=0.0)


def test_general_isotropic_blackbody_preserves_wien_tail_beyond_u80():
    target = BlackbodySpectrum(3.0)
    adaptive = alpha_exact(0.5, 3.0, epsrel=2e-7)
    general = alpha_isotropic_gauss(0.5, target, n_energy=256, n_angle=160)
    assert general > 0.0
    assert general == pytest.approx(adaptive, rel=2e-4)


def test_cached_presets_expose_speed_accuracy_tradeoff():
    from thermal_bw import cached_kernel_preset

    fast = cached_kernel_preset("fast")
    balanced = cached_kernel_preset("balanced")
    accurate = cached_kernel_preset("accurate")
    assert fast.representation_bytes < balanced.representation_bytes < accurate.representation_bytes
    z = 1.0 + 10.0 ** np.linspace(-8.0, 6.0, 200)
    direct = angle_averaged_cross_section(z, n_angle=256)
    fast_rel = np.abs(fast(z) - direct) / direct
    accurate_rel = np.abs(accurate(z) - direct) / direct
    assert np.max(fast_rel) < 1.0e-2
    assert np.max(accurate_rel) < 1.0e-4


def test_vectorized_cached_chunking_is_invariant():
    from thermal_bw import alpha_isotropic_cached

    target = BlackbodySpectrum(50.0)
    energies = np.logspace(np.log10(0.5), np.log10(50.0), 73)
    one_chunk = alpha_isotropic_cached(
        energies, target, preset="balanced", chunk_size=1000
    )
    many_chunks = alpha_isotropic_cached(
        energies, target, preset="balanced", chunk_size=7
    )
    assert np.allclose(one_chunk, many_chunks, rtol=2e-13, atol=0.0)


def test_fast_preset_is_subpercent_for_smooth_reference_cases():
    from thermal_bw import alpha_isotropic_cached

    target = BlackbodySpectrum(50.0)
    energies = np.logspace(np.log10(0.5), np.log10(50.0), 32)
    reference = alpha_isotropic_gauss(energies, target, n_energy=256, n_angle=192)
    fast = alpha_isotropic_cached(energies, target, preset="fast")
    relative = np.abs(fast - reference) / reference
    assert np.max(relative) < 2.0e-3

def test_alpha_model_custom_parameters_and_validation():
    value = alpha_model(2000.0, 50.0, params=np.array([-4.6, -1.5, -7.4, 0.68, 0.99, 1.0]))
    assert np.isfinite(value) and value > 0.0
    with pytest.raises(ValueError):
        alpha_model(2000.0, 50.0, params=[1.0, 2.0])


def test_thermal_eta_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        thermal_eta(0.0, 10.0)
    with pytest.raises(ValueError):
        thermal_eta(1.0, 0.0)


def test_cached_preset_validation_and_scalar_output():
    from thermal_bw import cached_kernel_preset

    with pytest.raises(ValueError):
        cached_kernel_preset("unknown")
    value = cached_kernel_preset("balanced")(2.0)
    assert isinstance(value, float) and value > 0.0


def test_opacity_table_reports_failure_when_too_small():
    from thermal_bw import TargetOpacityTable

    with pytest.raises(RuntimeError):
        TargetOpacityTable.build(
            BlackbodySpectrum(50.0),
            (1.0, 100.0),
            rtol=1e-8,
            initial_nodes=8,
            max_nodes=8,
        )

