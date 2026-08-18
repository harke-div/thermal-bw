# thermal-bw: v1.0.0

## Brief

Here I host the code for `thermal-bw`, which is a lightweight microphysics library computing local (1) Breit-Wheeler photon absorption and (2) the energy spectrum of the produced electron-positron pairs. Currently, THIS IS ONLY FOR ISOTROPIC RADIATION FIELDS. There are numerical reference methods (adaptive, Gauss-Legendre quadrature, etc.) for general isotropic specta, and for blackbody fields, I have supplied a lightweight, sub-percent accuracy analytic surrogate. The aim of this code is to provide local and lightweight calculations that may reduce computational cost for repeated evaluations and that may be integrated into larger radiative transfer, kinetic, or Monte Carlo modelling. For more information on this code, you may visit its corresponding paper in SoftwareX as an Original Software Publication: (to be added upon online publication).

## Installation

```bash
python -m pip install .
```

Optional dependencies:

```bash
python -m pip install ".[units]"              # Astropy quantities
python -m pip install -e ".[dev]"           # development and validation
```

Note that the core package only requires NumPy and SciPy.



## Blackbody opacity

```python
from thermal_bw import alpha_blackbody_gauss, alpha_exact, alpha_fit

fast = alpha_fit(2.0, 50.0)                    # cm^-1
adaptive = alpha_exact(2.0, 50.0, epsrel=1e-8)
gauss = alpha_blackbody_gauss(2.0, 50.0)
```

Plain inputs use MeV for the test photon and keV for `kT`. Astropy quantities are accepted when Astropy is installed.

For a blackbody target, there is a universal blackbody-scaling and dimensionless interaction variable

```text
eta = E_gamma kT / (m_e c^2)^2
alpha_gamma-gamma = (kT)^3 F(eta).
```

I have validated the surrogate with subpercent accuracy over the domain `0.01 <= eta <= 50`. For more context, consider `kT` in keV, which yields a corresponding gamma-ray energy of `E_gamma [MeV] = 261.121 eta / kT [keV]`.

### METRICS & RESULTS: 

Independent validation on 4096 interleaved logarithmic points gives median, 99th-percentile, and maximum relative errors of 0.225%, 0.316%, and 0.316%, respectively. A separate 2048-point scrambled Sobol validation gives the same 0.316% maximum error.

### Evaluation outside validated domain

The code default is to warn, `bounds="raise"` rejects extrapolation, and `bounds="ignore"` permits it without any warning. For more information on performance and validation, see the SoftwareX paper.

For users wanting high accuracy, I suggest using `alpha_blackbody_gauss` (Gauss-Legendre quadrature) or `alpha_exact` (adaptive quadrature).



## Isotropic target spectra

Target callables return differential photon number density in units of `cm^-3 keV^-1`.

```python
import numpy as np
from thermal_bw import PowerLawSpectrum, alpha_isotropic_cached

target = PowerLawSpectrum(
    normalization=1e12,
    index=1.5,
    reference_keV=1.0,
    cutoff_keV=500.0,
    energy_bounds_keV=(0.1, 5000.0),
)

energy = np.logspace(-1, 2, 200)
alpha = alpha_isotropic_cached(energy, target, preset="balanced")
```

The code contains the following targets: blackbody, greybody, power-law, broken-power-law, tabulated, callable, discrete-line, and composite spectra. A callable may represent other local spectral shapes, provided its values have the same interpretation of number density.

Numerical methods available are

- `alpha_isotropic_adaptive`
- `alpha_isotropic_gauss`
- `alpha_isotropic_cached`
- `alpha_isotropic_auto`

When using `alpha_isotropic_cached`, you will be provided with 3 presets (`fast`, `balanced`, and `accurate`) that provide a combination of speed and accuracy. These differ only in the number of tabulated angular-kernel values and the target-energy quadrature order. When using `alpha_isotropic_auto`, it will increase the target-energy quadrature order until the result converges. For a callable without stated breakpoints, it also conducts comparison of the result with adaptive integration before returning it.



## Secondary pair spectra
