"""Warnings and exceptions raised by :mod:`thermal_bw`"""


class ThermalBWError(Exception):
    """base exception for thermal-bw."""


class InputValidationError(ThermalBWError, ValueError):
    """raised when an input is non-finite, non-positive, or dimensionally invalid."""


class OutOfDomainError(ThermalBWError, ValueError):
    """raised when the analytic surrogate is evaluated outside its validated domain."""


class OutOfDomainWarning(UserWarning):
    """warning emitted when the analytic surrogate is used outside its validated domain."""
