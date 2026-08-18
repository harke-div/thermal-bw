"""Optional quantity conversion."""
from __future__ import annotations

import numpy as np

from .exceptions import InputValidationError


def as_value(value, unit: str):
    """Return a NumPy value in ``unit`` while keeping Astropy optional."""
    if hasattr(value, "to_value"):
        try:
            value = value.to_value(unit)
        except Exception as exc:
            raise InputValidationError(f"ERROR: input is not convertible to {unit}") from exc
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("ERROR: input must be numeric") from exc


def positive_values(value, unit: str, message: str):
    """return finite positive numerical values"""
    value = as_value(value, unit)
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise InputValidationError(f"ERROR: {message}")
    return value

