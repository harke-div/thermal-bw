from __future__ import annotations

import numpy as np
import pytest

from thermal_bw import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    CallableSpectrum,
    CompositeSpectrum,
    DiscreteLineSpectrum,
    InputValidationError,
    OutOfDomainError,
    OutOfDomainWarning,
    PowerLawSpectrum,
    PreparedTarget,
    TabulatedSpectrum,
    TargetOpacityTable,
    alpha_isotropic_adaptive,
    alpha_isotropic_auto,
    alpha_isotropic_cached,
    alpha_isotropic_gauss,
    cached_kernel_preset,
    prepare_target,
)
from thermal_bw.constants import ME_C2_KEV
from thermal_bw.targets import discrete_lines, integration_breakpoints


def test_target_breakpoints_and_callable_validation():
    broken = BrokenPowerLawSpectrum(1.0, 1.0, 2.0, 10.0, (1.0, 100.0))
    assert integration_breakpoints(broken) == (10.0,)
    tab = TabulatedSpectrum([1.0, 2.0, 4.0, 8.0], [1.0, 3.0, 2.0, 1.0])
    assert integration_breakpoints(tab) == (2.0, 4.0)
    call = CallableSpectrum(lambda x: np.ones_like(x), (1.0, 10.0), (2.0, 5.0))
    assert integration_breakpoints(call) == (2.0, 5.0)
    with pytest.raises(InputValidationError):
        CallableSpectrum(lambda x: x, (1.0, 10.0), (0.5,))


def test_discrete_line_opacity():
    line = DiscreteLineSpectrum([100.0, 10.0], [2e20, 1e20])
    energies, densities = discrete_lines(line)
    E = np.array([0.5, 2.0, 50.0])
    kernel = cached_kernel_preset("accurate")
    z = E[:, None] * 1e3 * energies[None, :] / ME_C2_KEV**2
    expected = np.sum(densities[None, :] * kernel(z), axis=1)
    assert np.allclose(alpha_isotropic_cached(E, line, kernel=kernel), expected)
    assert np.allclose(alpha_isotropic_gauss(E, line, n_angle=256), expected, rtol=1e-4)


def test_auto_converges_to_adaptive_for_callable():
    target = CallableSpectrum(
        lambda eps: 1e18 * np.exp(-np.asarray(eps) / 100.0),
        (1.0, 1000.0),
    )
    value = alpha_isotropic_auto(2.0, target, rtol=2e-3, initial_order=16, max_order=128)
    reference = alpha_isotropic_adaptive(2.0, target, epsrel=2e-7)
    assert value == pytest.approx(reference, rel=2e-3)


def test_auto_validation_errors():
    target = PowerLawSpectrum(1e18, 1.5, (1.0, 1000.0))
    with pytest.raises(InputValidationError):
        alpha_isotropic_auto(2.0, target, rtol=0.0)
    with pytest.raises(InputValidationError):
        alpha_isotropic_auto(2.0, target, initial_order=16, max_order=20)


def test_prepared_target_accuracy_and_composite():
    E = np.logspace(np.log10(0.5), np.log10(50.0), 48)
    bb = BlackbodySpectrum(50.0)
    prepared = prepare_target(bb, 0.5, preset="fast")
    assert isinstance(prepared, PreparedTarget)
    assert prepared.representation_bytes > 0
    value = prepared.opacity(E, chunk_size=17)
    reference = alpha_isotropic_cached(E, bb, preset="accurate", n_energy=192)
    assert np.max(np.abs(value - reference) / reference) < 1e-2
    with pytest.raises(InputValidationError):
        prepared.opacity(0.4)

    continuum = PowerLawSpectrum(1e18, 1.0, (1.0, 1000.0))
    line = DiscreteLineSpectrum([50.0], [1e18])
    composite = CompositeSpectrum([continuum, line])
    got = prepare_target(composite, 0.3).opacity(E)
    expected = alpha_isotropic_cached(E, composite)
    assert np.allclose(got, expected, rtol=3e-3)


def test_prepared_target_invalid_inputs():
    target = PowerLawSpectrum(1e18, 1.0, (1.0, 100.0))
    with pytest.raises(InputValidationError):
        PreparedTarget(object(), 0.3)
    with pytest.raises(InputValidationError):
        PreparedTarget(target, 0.3, preset="bad")
    with pytest.raises(InputValidationError):
        PreparedTarget(target, 0.3, kernel=3)
    with pytest.raises(InputValidationError):
        prepare_target(target, 0.0)


def test_opacity_table_accuracy_and_bounds():
    target = BlackbodySpectrum(50.0)
    table = TargetOpacityTable.build(
        target,
        (1.0, 100.0),
        rtol=1e-3,
        initial_nodes=24,
        max_nodes=512,
        reference_n_energy=128,
    )
    assert table.n_nodes > 24
    assert table.validation_max_relative_error < 1e-3
    E = np.logspace(0.0, 2.0, 300)
    reference = alpha_isotropic_cached(E, target, preset="accurate", n_energy=128)
    assert np.max(np.abs(table(E) - reference) / reference) < 1.5e-3
    with pytest.warns(OutOfDomainWarning):
        table(0.5)
    with pytest.raises(OutOfDomainError):
        table(0.5, bounds="raise")
    with pytest.raises(InputValidationError):
        table(2.0, bounds="bad")


def test_opacity_table_finite_threshold_and_invalid_builds():
    target = PowerLawSpectrum(1e18, 1.0, (1.0, 100.0))
    threshold = ME_C2_KEV**2 / (100.0 * 1e3)
    table = TargetOpacityTable.build(
        target,
        (threshold * 0.5, 30.0),
        rtol=3e-3,
        initial_nodes=16,
        max_nodes=1024,
        reference_n_energy=128,
    )
    assert table(threshold * 0.9) == 0.0
    assert table(threshold * 1.5) > 0.0

    with pytest.raises(InputValidationError):
        TargetOpacityTable.build(object(), (1.0, 2.0))
    with pytest.raises(InputValidationError):
        TargetOpacityTable.build(target, (2.0, 1.0))
    with pytest.raises(InputValidationError):
        TargetOpacityTable.build(target, (1.0, 2.0), rtol=2.0)
    with pytest.raises(InputValidationError):
        TargetOpacityTable.build(target, (1.0, 2.0), initial_nodes=2)
    with pytest.raises(InputValidationError):
        TargetOpacityTable.build(target, (0.1, threshold * 0.9))


def test_composite_adaptive_is_additive():
    target = CompositeSpectrum(
        [
            PowerLawSpectrum(1e18, 1.0, (1.0, 1000.0)),
            DiscreteLineSpectrum([100.0], [2e18]),
        ]
    )
    full = alpha_isotropic_adaptive(2.0, target, epsrel=1e-6)
    pieces = sum(alpha_isotropic_adaptive(2.0, item, epsrel=1e-6) for item in target.components)
    assert full == pytest.approx(pieces, rel=1e-12)
