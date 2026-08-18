import numpy as np
import pytest
from scipy import integrate

from thermal_bw import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    CompositeSpectrum,
    DiscreteLineSpectrum,
    GreybodySpectrum,
    PowerLawSpectrum,
    alpha_isotropic_gauss,
    angle_averaged_cross_section,
    pair_distribution,
    pair_dsigma_dgamma,
    pair_gamma_bounds,
    pair_injection,
    pair_spectrum,
    pair_spectrum_adaptive,
)
from thermal_bw.constants import C_CGS

def test_pair_gamma_bounds_match_known_kinematic_limits():
    assert pair_gamma_bounds(2.0, 2.0) == pytest.approx((1.0, 3.0), rel=0.0, abs=1e-14)
    assert pair_gamma_bounds(10.0, 0.3) == pytest.approx(
        (1.1899915825005292, 9.110008417499472), rel=2e-14
    )
    assert pair_gamma_bounds(1.0, 1.0) == (0.0, 0.0)

def test_pair_kernel_is_symmetric_under_photon_and_lepton_exchange():
    x1, x2 = 10.0, 0.3
    lo, hi = pair_gamma_bounds(x1, x2)
    gamma = np.linspace(lo + 1e-5, hi - 1e-5, 41)
    direct = pair_dsigma_dgamma(x1, x2, gamma)
    swapped = pair_dsigma_dgamma(x2, x1, gamma)
    reflected = pair_dsigma_dgamma(x1, x2, x1 + x2 - gamma)
    assert np.allclose(direct, swapped, rtol=2e-11, atol=0.0)
    assert np.allclose(direct, reflected, rtol=2e-9, atol=0.0)


@pytest.mark.parametrize(
    "x1,x2",
    [(10.0, 0.3), (5.0, 1.0), (2.0, 2.0), (20.0, 0.2), (3.0, 0.5)],
)
def test_pair_kernel_integrates_to_total_cross_section(x1, x2):
    upper = x1 + x2 - 1.0
    total = integrate.quad(
        lambda gamma: pair_dsigma_dgamma(x1, x2, gamma),
        1.0,
        upper,
        epsabs=1e-38,
        epsrel=1e-6,
        limit=800,
    )[0]
    reference = angle_averaged_cross_section(x1 * x2, n_angle=256)
    assert total == pytest.approx(reference, rel=2e-6)



@pytest.mark.parametrize(
    "product,ratio",
    [
        (1.00001, 1.0e6),
        (1.001, 1.0e8),
        (1.001, 1.0e12),
        (1.1, 1.0e12),
        (2.0, 1.0e12),
        (10.0, 1.0e13),
    ],
)
def test_pair_kernel_extreme_asymmetry_preserves_normalization_and_mean(product, ratio):
    x1 = np.sqrt(product * ratio)
    x2 = np.sqrt(product / ratio)
    lower, upper = pair_gamma_bounds(x1, x2)
    midpoint = 0.5 * (lower + upper)
    total = integrate.quad(
        lambda gamma: pair_dsigma_dgamma(x1, x2, gamma),
        lower,
        upper,
        points=[midpoint],
        epsabs=1e-40,
        epsrel=2e-6,
        limit=800,
    )[0]
    first = integrate.quad(
        lambda gamma: gamma * pair_dsigma_dgamma(x1, x2, gamma),
        lower,
        upper,
        points=[midpoint],
        epsabs=1e-34,
        epsrel=4e-6,
        limit=800,
    )[0]
    reference = angle_averaged_cross_section(product, n_angle=512)
    assert total == pytest.approx(reference, rel=5e-6)
    assert first / total == pytest.approx(0.5 * (x1 + x2), rel=5e-6)

def test_pair_kernel_first_moment_obeys_energy_symmetry():
    x1, x2 = 10.0, 0.3
    upper = x1 + x2 - 1.0
    total = integrate.quad(
        lambda gamma: pair_dsigma_dgamma(x1, x2, gamma),
        1.0,
        upper,
        epsabs=1e-38,
        epsrel=1e-6,
        limit=800,
    )[0]
    first = integrate.quad(
        lambda gamma: gamma * pair_dsigma_dgamma(x1, x2, gamma),
        1.0,
        upper,
        epsabs=1e-38,
        epsrel=1e-6,
        limit=800,
    )[0]
    assert first / total == pytest.approx(0.5 * (x1 + x2), rel=2e-6)

def test_pair_spectrum_line_integrates_to_opacity_and_correct_mean_energy():
    target = DiscreteLineSpectrum([100.0], [1e12])
    gamma_energy = 5.0
    upper = gamma_energy + 0.1 - 0.511
    electron_energy = np.linspace(0.511, upper, 1201)
    spectrum = pair_spectrum(gamma_energy, electron_energy, target)
    total = integrate.simpson(spectrum, x=electron_energy)
    mean = integrate.simpson(electron_energy * spectrum, x=electron_energy) / total
    reference = alpha_isotropic_gauss(gamma_energy, target, n_angle=256)
    assert total == pytest.approx(reference, rel=4e-6)
    assert mean == pytest.approx(0.5 * (gamma_energy + 0.1), rel=5e-6)

def test_pair_gauss_matches_adaptive_target_integration():
    target = BlackbodySpectrum(50.0)
    electron_energy = np.array([0.6, 0.8, 1.0, 1.5, 2.0])
    gauss = pair_spectrum(2.0, electron_energy, target, n_energy=192)
    adaptive = pair_spectrum_adaptive(2.0, electron_energy, target, epsrel=2e-7)
    assert np.allclose(gauss, adaptive, rtol=2e-4, atol=0.0)

def test_pair_distribution_is_normalized_for_line_target():
    target = DiscreteLineSpectrum([100.0], [1e12])
    electron_energy = np.linspace(0.511, 4.589, 1201)
    probability = pair_distribution(5.0, electron_energy, target, n_angle=256)
    assert integrate.simpson(probability, x=electron_energy) == pytest.approx(1.0, rel=4e-6)

def test_blackbody_pair_spectrum_recovers_total_opacity():
    target = BlackbodySpectrum(50.0)
    electron_energy = np.linspace(0.511, 5.489, 360)
    spectrum = pair_spectrum(2.0, electron_energy, target, n_energy=128)
    total = integrate.simpson(spectrum, x=electron_energy)
    reference = alpha_isotropic_gauss(2.0, target, n_energy=192, n_angle=192)
    assert total == pytest.approx(reference, rel=8e-4)

@pytest.mark.parametrize(
    "target",
    [
        GreybodySpectrum(50.0, dilution=0.3),
        PowerLawSpectrum(1e10, 1.5, (1.0, 1000.0), cutoff_keV=400.0),
        BrokenPowerLawSpectrum(1e10, 1.0, 2.2, 30.0, (1.0, 1000.0), cutoff_keV=500.0),
    ],
)
def test_pair_spectrum_recovers_opacity_for_continuum_targets(target):
    electron_energy = np.linspace(0.511, 5.489, 500)
    spectrum = pair_spectrum(5.0, electron_energy, target, n_energy=96)
    total = integrate.simpson(spectrum, x=electron_energy)
    reference = alpha_isotropic_gauss(5.0, target, n_energy=192, n_angle=192)
    assert total == pytest.approx(reference, rel=2e-4)


def test_pair_spectrum_is_additive_for_composite_targets():
    first = GreybodySpectrum(30.0, dilution=0.2)
    second = GreybodySpectrum(80.0, dilution=0.05)
    composite = CompositeSpectrum([first, second])
    electron_energy = np.array([0.7, 1.0, 1.5, 2.0])
    total = pair_spectrum(3.0, electron_energy, composite, n_energy=96)
    separate = pair_spectrum(3.0, electron_energy, first, n_energy=96) + pair_spectrum(
        3.0, electron_energy, second, n_energy=96
    )
    assert np.allclose(total, separate, rtol=2e-12, atol=0.0)


def test_pair_injection_integrates_to_gamma_absorption_rate_for_line_target():
    target = DiscreteLineSpectrum([100.0], [2e10])
    gamma_energy = np.linspace(2.0, 8.0, 31)
    gamma_density = 3e4 * (gamma_energy / 2.0) ** -2.0
    electron_energy = np.linspace(0.511, 7.589, 500)
    source = pair_injection(
        gamma_energy,
        gamma_density,
        electron_energy,
        target,
        n_energy=64,
    )
    total_source = integrate.simpson(source, x=electron_energy)
    alpha = alpha_isotropic_gauss(gamma_energy, target, n_angle=192)
    reference = C_CGS * integrate.trapezoid(gamma_density * alpha, x=gamma_energy)
    assert total_source == pytest.approx(reference, rel=5e-3)


def test_combined_pair_injection_conserves_energy_for_line_target():
    target = DiscreteLineSpectrum([100.0], [2e10])
    gamma_energy = np.linspace(2.0, 8.0, 41)
    gamma_density = 3e4 * (gamma_energy / 2.0) ** -2.0
    electron_energy = np.linspace(0.511, 7.589, 700)
    source = pair_injection(
        gamma_energy,
        gamma_density,
        electron_energy,
        target,
        combined=True,
    )
    injected_energy = integrate.trapezoid(electron_energy * source, x=electron_energy)
    alpha = alpha_isotropic_gauss(gamma_energy, target, n_angle=192)
    reference = C_CGS * integrate.trapezoid(
        gamma_density * alpha * (gamma_energy + 0.1), x=gamma_energy
    )
    assert injected_energy == pytest.approx(reference, rel=5e-3)


def test_combined_pair_injection_is_twice_single_species():
    target = DiscreteLineSpectrum([100.0], [2e10])
    gamma_energy = np.linspace(2.0, 8.0, 9)
    gamma_density = np.ones_like(gamma_energy)
    electron_energy = np.linspace(0.511, 7.589, 80)
    single = pair_injection(gamma_energy, gamma_density, electron_energy, target)
    combined = pair_injection(
        gamma_energy,
        gamma_density,
        electron_energy,
        target,
        combined=True,
    )
    assert np.allclose(combined, 2.0 * single, rtol=1e-14, atol=0.0)


def test_same_population_pair_injection_applies_collision_counting_factor():
    target = DiscreteLineSpectrum([100.0], [2e10])
    gamma_energy = np.linspace(2.0, 8.0, 9)
    gamma_density = np.ones_like(gamma_energy)
    electron_energy = np.linspace(0.511, 7.589, 80)
    distinct = pair_injection(gamma_energy, gamma_density, electron_energy, target)
    same = pair_injection(
        gamma_energy,
        gamma_density,
        electron_energy,
        target,
        same_photon_population=True,
    )
    same_combined = pair_injection(
        gamma_energy,
        gamma_density,
        electron_energy,
        target,
        combined=True,
        same_photon_population=True,
    )
    assert np.allclose(same, 0.5 * distinct, rtol=1e-14, atol=0.0)
    assert np.allclose(same_combined, distinct, rtol=1e-14, atol=0.0)


def test_pair_spectrum_handles_kinematic_bound_at_target_endpoint():
    target = PowerLawSpectrum(1e10, 1.5, (1.0, 1000.0), cutoff_keV=400.0)
    electron_energy = np.linspace(0.511, 2.489, 300)
    spectrum = pair_spectrum(2.0, electron_energy, target, n_energy=64)
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)


def test_pair_inputs_are_validated():
    target = BlackbodySpectrum(50.0)
    with pytest.raises(ValueError):
        pair_spectrum(-1.0, [1.0], target)
    with pytest.raises(ValueError):
        pair_spectrum(2.0, [-1.0], target)
    with pytest.raises(ValueError):
        pair_spectrum(2.0, [0.5], target)
    with pytest.raises(ValueError):
        pair_dsigma_dgamma(-1.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        pair_injection([2.0, 1.0], [1.0, 1.0], [1.0], target)
    with pytest.raises(ValueError):
        pair_injection([1.0, 2.0], [1.0, 1.0], [0.5], target)
    with pytest.raises(ValueError):
        pair_injection([1.0, 2.0], [1.0, 1.0], [1.0], target, combined="yes")


@pytest.mark.parametrize("center", [1.3, 9.0])
def test_pair_kernel_is_continuous_at_c_zero(center):
    exact = pair_dsigma_dgamma(10.0, 0.3, center)
    left = pair_dsigma_dgamma(10.0, 0.3, center - 1.0e-8)
    right = pair_dsigma_dgamma(10.0, 0.3, center + 1.0e-8)
    assert exact == pytest.approx(0.5 * (left + right), rel=2e-7)
