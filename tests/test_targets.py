from __future__ import annotations

import numpy as np
import pytest

from thermal_bw import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    CachedAngleAveragedKernel,
    CallableSpectrum,
    CompositeSpectrum,
    GreybodySpectrum,
    InputValidationError,
    PowerLawSpectrum,
    TabulatedSpectrum,
    alpha_blackbody_gauss,
    alpha_exact_keV,
    alpha_fit,
    alpha_grid,
    alpha_isotropic_adaptive,
    alpha_isotropic_cached,
    alpha_isotropic_gauss,
    sigma_breit_wheeler,
    sigma_breit_wheeler_beta,
)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_blackbody_invalid_temperature(bad):
    with pytest.raises(InputValidationError):
        BlackbodySpectrum(bad)


def test_blackbody_invalid_reduced_bounds():
    with pytest.raises(InputValidationError):
        BlackbodySpectrum(10.0, u_min=1.0, u_max=0.5)


def test_blackbody_density_scalar_and_array():
    bb = BlackbodySpectrum(10.0)
    assert bb.number_density(1.0) > 0.0
    arr = bb.number_density(np.array([-1.0, 1.0, 10000.0]))
    assert arr.shape == (3,)
    assert arr[0] == 0.0
    assert arr[1] > 0.0
    assert arr[2] >= 0.0


def test_greybody_validation_and_bounds():
    with pytest.raises(InputValidationError):
        GreybodySpectrum(10.0, -1.0)
    g = GreybodySpectrum(10.0, 0.2)
    assert g.energy_bounds_keV == pytest.approx((1e-7, 800.0))
    assert g.number_density(1.0) == pytest.approx(0.2 * BlackbodySpectrum(10.0).number_density(1.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(normalization=-1.0, index=1.0, energy_bounds_keV=(1.0, 10.0)),
        dict(normalization=1.0, index=np.nan, energy_bounds_keV=(1.0, 10.0)),
        dict(normalization=1.0, index=1.0, energy_bounds_keV=(10.0, 1.0)),
        dict(normalization=1.0, index=1.0, energy_bounds_keV=(1.0, 10.0), reference_keV=0.0),
        dict(normalization=1.0, index=1.0, energy_bounds_keV=(1.0, 10.0), cutoff_keV=-1.0),
    ],
)
def test_powerlaw_validation(kwargs):
    with pytest.raises(InputValidationError):
        PowerLawSpectrum(**kwargs)


def test_powerlaw_scalar_and_support():
    p = PowerLawSpectrum(2.0, 1.0, (1.0, 10.0), reference_keV=2.0, cutoff_keV=100.0)
    assert p.number_density(2.0) > 0.0
    assert p.number_density(0.5) == 0.0
    vals = p.number_density(np.array([0.5, 2.0, 20.0]))
    assert vals[0] == vals[2] == 0.0
    assert vals[1] > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            normalization_at_break=-1.0,
            index_low=1,
            index_high=2,
            break_keV=2,
            energy_bounds_keV=(1, 10),
        ),
        dict(
            normalization_at_break=1.0,
            index_low=1,
            index_high=2,
            break_keV=20,
            energy_bounds_keV=(1, 10),
        ),
        dict(
            normalization_at_break=1.0,
            index_low=np.nan,
            index_high=2,
            break_keV=2,
            energy_bounds_keV=(1, 10),
        ),
        dict(
            normalization_at_break=1.0,
            index_low=1,
            index_high=2,
            break_keV=2,
            energy_bounds_keV=(1, 10),
            cutoff_keV=0,
        ),
    ],
)
def test_broken_powerlaw_validation(kwargs):
    with pytest.raises(InputValidationError):
        BrokenPowerLawSpectrum(**kwargs)


def test_broken_powerlaw_continuity_and_support():
    b = BrokenPowerLawSpectrum(3.0, 1.0, 2.0, 10.0, (1.0, 100.0), cutoff_keV=1000.0)
    assert b.number_density(10.0) > 0.0
    left = b.number_density(10.0 * (1 - 1e-10))
    right = b.number_density(10.0 * (1 + 1e-10))
    assert left == pytest.approx(right, rel=1e-8)
    vals = b.number_density(np.array([0.1, 5.0, 50.0, 1000.0]))
    assert vals[0] == vals[-1] == 0.0
    assert np.all(vals[1:3] > 0.0)


@pytest.mark.parametrize(
    "eps,dens",
    [
        ([1.0], [1.0]),
        ([1.0, 2.0], [1.0]),
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, 0.0]),
        ([2.0, 1.0], [1.0, 2.0]),
    ],
)
def test_tabulated_validation(eps, dens):
    with pytest.raises(InputValidationError):
        TabulatedSpectrum(eps, dens)


def test_tabulated_scalar_and_outside_support():
    t = TabulatedSpectrum([1.0, 10.0, 100.0], [10.0, 1.0, 0.1])
    assert t.energy_bounds_keV == (1.0, 100.0)
    assert t.number_density(10.0) == pytest.approx(1.0)
    vals = t.number_density(np.array([0.1, 10.0, 1000.0]))
    assert np.array_equal(vals[[0, 2]], [0.0, 0.0])


def test_callable_validation_and_output_checks():
    with pytest.raises(InputValidationError):
        CallableSpectrum(3, (1.0, 10.0))
    with pytest.raises(InputValidationError):
        CallableSpectrum(lambda x: x, (10.0, 1.0))
    c = CallableSpectrum(lambda x: np.ones_like(x) * 2.0, (1.0, 10.0))
    assert c.number_density(2.0) == 2.0
    assert np.array_equal(c.number_density(np.array([0.5, 2.0, 20.0])), [0.0, 2.0, 0.0])
    with pytest.raises(InputValidationError):
        CallableSpectrum(lambda x: np.array([1.0, 2.0]), (1.0, 10.0)).number_density(np.ones(3))
    with pytest.raises(InputValidationError):
        CallableSpectrum(lambda x: -np.ones_like(x), (1.0, 10.0)).number_density(2.0)


def test_composite_validation_bounds_and_density():
    with pytest.raises(InputValidationError):
        CompositeSpectrum([])
    with pytest.raises(InputValidationError):
        CompositeSpectrum([object()])
    a = PowerLawSpectrum(1.0, 1.0, (1.0, 10.0))
    b = PowerLawSpectrum(2.0, 2.0, (0.1, 100.0))
    c = CompositeSpectrum([a, b])
    assert c.energy_bounds_keV == (0.1, 100.0)
    assert c.number_density(2.0) == pytest.approx(a.number_density(2.0) + b.number_density(2.0))
    assert c.number_density(np.array([2.0, 3.0])).shape == (2,)


def test_cross_section_beta_scalar_array_and_wrapper():
    assert sigma_breit_wheeler_beta(0.0) == 0.0
    arr = sigma_breit_wheeler_beta(np.array([-1.0, 0.5, 1.0, np.nan]))
    assert arr[1] > 0.0
    assert np.count_nonzero(arr) == 1
    assert sigma_breit_wheeler(0.5) == pytest.approx(sigma_breit_wheeler_beta(0.5))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(E_MeV=0.0, kT_keV=10.0),
        dict(E_MeV=1.0, kT_keV=0.0),
        dict(E_MeV=1.0, kT_keV=10.0, n_angle=2),
        dict(E_MeV=1.0, kT_keV=10.0, n_planck=2),
        dict(E_MeV=1.0, kT_keV=10.0, u_base_max=0.0),
    ],
)
def test_gauss_blackbody_invalid_inputs(kwargs):
    with pytest.raises(InputValidationError):
        alpha_blackbody_gauss(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(E_keV=0.0, kT_keV=10.0),
        dict(E_keV=1000.0, kT_keV=0.0),
        dict(E_keV=1000.0, kT_keV=10.0, u_base_max=0.0),
        dict(E_keV=1000.0, kT_keV=10.0, epsrel=0.0),
    ],
)
def test_adaptive_exact_invalid_inputs(kwargs):
    with pytest.raises(InputValidationError):
        alpha_exact_keV(**kwargs)


def test_alpha_grid_shape_and_values():
    grid = alpha_grid([1.0, 2.0], [20.0, 50.0], epsrel=1e-5)
    assert grid.shape == (2, 2)
    assert np.all(grid > 0.0)


def test_kernel_validation_and_outside_policy():
    with pytest.raises(InputValidationError):
        CachedAngleAveragedKernel(q_min=1.0, q_max=0.0)
    with pytest.raises(InputValidationError):
        CachedAngleAveragedKernel(n_nodes=10)
    with pytest.raises(InputValidationError):
        CachedAngleAveragedKernel(n_angle=4)
    with pytest.raises(InputValidationError):
        CachedAngleAveragedKernel(outside="bad")
    kernel = CachedAngleAveragedKernel(q_min=-2, q_max=2, n_nodes=128, n_angle=64, outside="raise")
    with pytest.raises(InputValidationError):
        kernel(1.0 + 1e-5)
    with pytest.raises(InputValidationError):
        kernel(-1.0)


def test_isotropic_invalid_target_and_bounds():
    with pytest.raises(InputValidationError):
        alpha_isotropic_gauss(1.0, object())
    target = PowerLawSpectrum(1.0, 1.0, (1.0, 10.0))
    with pytest.raises(InputValidationError):
        alpha_isotropic_gauss(1.0, target, energy_bounds_keV=(10.0, 1.0))
    with pytest.raises(InputValidationError):
        alpha_isotropic_cached(1.0, target, kernel=3)
    with pytest.raises(InputValidationError):
        alpha_isotropic_adaptive(0.0, target)


def test_fit_ignore_and_invalid_bounds_mode():
    assert alpha_fit(0.1, 50.0, bounds="ignore") >= 0.0
    with pytest.raises(InputValidationError):
        alpha_fit(1.0, 50.0, bounds="bad")
