from __future__ import annotations

import numpy as np
import pytest

from thermal_bw import (
    BlackbodySpectrum,
    DiscreteLineSpectrum,
    InputValidationError,
    alpha_blackbody_gauss,
    alpha_exact,
    alpha_fit,
    pair_injection,
)


class Quantity:
    scale = {
        ("MeV", "MeV"): 1.0,
        ("keV", "keV"): 1.0,
        ("MeV", "keV"): 1000.0,
        ("keV", "MeV"): 0.001,
        ("1 / (cm3 MeV)", "1 / (cm3 MeV)"): 1.0,
    }

    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def to_value(self, unit):
        try:
            return np.asarray(self.value) * self.scale[(self.unit, unit)]
        except KeyError as exc:
            raise ValueError("incompatible units") from exc

def test_quantity_inputs_match_floats():
    energy = Quantity(2.0, "MeV")
    temp = Quantity(50.0, "keV")
    assert alpha_fit(energy, temp) == pytest.approx(alpha_fit(2.0, 50.0))
    assert alpha_blackbody_gauss(energy, temp) == pytest.approx(
        alpha_blackbody_gauss(2.0, 50.0)
    )
    assert alpha_exact(energy, temp) == pytest.approx(alpha_exact(2.0, 50.0))
    assert BlackbodySpectrum(temp).kT_keV == 50.0

def test_quantity_arrays_and_lines():
    energy = Quantity([0.5, 2.0, 50.0], "MeV")
    values = alpha_fit(energy, Quantity(50.0, "keV"))
    assert values.shape == (3,)
    lines = DiscreteLineSpectrum(Quantity([1.0, 10.0], "keV"), [1e10, 2e10])
    assert lines.energies_keV == (1.0, 10.0)

def test_incompatible_quantity_rejected():
    with pytest.raises(InputValidationError):
        alpha_fit(Quantity(2.0, "second"), 50.0)

def test_pair_injection_accepts_density_quantity():
    target = DiscreteLineSpectrum([100.0], [2e10])
    energy = Quantity([2.0, 4.0, 8.0], "MeV")
    density = Quantity([1.0, 0.5, 0.25], "1 / (cm3 MeV)")
    electron_energy = Quantity([0.7, 1.0, 2.0], "MeV")
    quantity = pair_injection(energy, density, electron_energy, target)
    plain = pair_injection(
        [2.0, 4.0, 8.0], [1.0, 0.5, 0.25], [0.7, 1.0, 2.0], target
    )
    assert np.allclose(quantity, plain, rtol=0.0, atol=0.0)

def test_extreme_inputs_are_stable():
    low = alpha_fit(1e-8, 1e-4, bounds="ignore")
    high = alpha_fit(1e8, 1e8, bounds="ignore")
    assert np.isfinite(low) and low >= 0.0
    assert np.isfinite(high) and high >= 0.0
    tail = alpha_blackbody_gauss(0.01, 1.0, n_angle=32, n_planck=64)
    assert np.isfinite(tail) and tail >= 0.0

def test_astropy_quantities_when_available():
    units = pytest.importorskip("astropy.units")
    value = alpha_fit(2.0 * units.MeV, 50.0 * units.keV)
    assert value == pytest.approx(alpha_fit(2.0, 50.0))
