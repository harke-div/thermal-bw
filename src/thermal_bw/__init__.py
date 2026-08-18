"""thermal-bw: Breit--Wheeler opacity and secondary pair spectra."""
from __future__ import annotations

from .approx import (
    DEFAULT_PARAMS,
    SURROGATE_ETA_DOMAIN,
    alpha_fit,
    alpha_model,
    thermal_eta,
    within_validated_domain,
)
from .blackbody import alpha_blackbody_gauss
from .cross_sections import sigma_breit_wheeler_beta, sigma_breit_wheeler_s
from .exact import alpha_exact, alpha_exact_keV, alpha_grid, sigma_breit_wheeler
from .exceptions import (
    InputValidationError,
    OutOfDomainError,
    OutOfDomainWarning,
    ThermalBWError,
)
from .isotropic import (
    ResolutionWarning,
    alpha_isotropic_adaptive,
    alpha_isotropic_auto,
    alpha_isotropic_cached,
    alpha_isotropic_gauss,
    angle_averaged_cross_section,
)
from .kernel import CachedAngleAveragedKernel, cached_kernel_preset
from .opacity_table import TargetOpacityTable
from .pairs import (
    pair_distribution,
    pair_dsigma_dgamma,
    pair_gamma_bounds,
    pair_injection,
    pair_spectrum,
    pair_spectrum_adaptive,
)
from .prepared import PreparedTarget, prepare_target
from .targets import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    CallableSpectrum,
    CompositeSpectrum,
    DiscreteLineSpectrum,
    GreybodySpectrum,
    PhotonSpectrum,
    PowerLawSpectrum,
    TabulatedSpectrum,
)

__all__ = [
    "DEFAULT_PARAMS",
    "SURROGATE_ETA_DOMAIN",
    "ThermalBWError",
    "InputValidationError",
    "OutOfDomainError",
    "OutOfDomainWarning",
    "ResolutionWarning",
    "PhotonSpectrum",
    "BlackbodySpectrum",
    "GreybodySpectrum",
    "PowerLawSpectrum",
    "BrokenPowerLawSpectrum",
    "TabulatedSpectrum",
    "CallableSpectrum",
    "CompositeSpectrum",
    "DiscreteLineSpectrum",
    "alpha_fit",
    "alpha_model",
    "alpha_exact",
    "alpha_exact_keV",
    "alpha_grid",
    "alpha_blackbody_gauss",
    "alpha_isotropic_gauss",
    "alpha_isotropic_cached",
    "alpha_isotropic_auto",
    "alpha_isotropic_adaptive",
    "angle_averaged_cross_section",
    "CachedAngleAveragedKernel",
    "cached_kernel_preset",
    "PreparedTarget",
    "prepare_target",
    "TargetOpacityTable",
    "pair_gamma_bounds",
    "pair_dsigma_dgamma",
    "pair_spectrum",
    "pair_spectrum_adaptive",
    "pair_distribution",
    "pair_injection",
    "sigma_breit_wheeler",
    "sigma_breit_wheeler_beta",
    "sigma_breit_wheeler_s",
    "thermal_eta",
    "within_validated_domain",
]

__version__ = "1.0.0"
