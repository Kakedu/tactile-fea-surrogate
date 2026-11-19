"""Sphere indentation of a thin elastic layer bonded to a rigid base.

Hertz assumes a half-space. A tactile sensor's gel is a few millimetres thick
and glued to acrylic, so the base stiffens the response. Dimitriadis et al.
(Biophysical Journal 82, 2002) give a correction for an incompressible layer:

    F = F_hertz * (1 + 1.133 x + 1.283 x^2 + 0.769 x^3 + 0.0975 x^4),
    x = sqrt(R * depth) / thickness

Assumptions: incompressible (nu = 0.5), linear elastic, small strain,
frictionless indenter, layer bonded to the base, paraboloidal indenter.
The authors derived it as a series in x, so trust it for x well below 1.

Units: mm, N, MPa.
"""
from __future__ import annotations

import numpy as np

import hertz

BONDED = (1.0, 1.133, 1.283, 0.769, 0.0975)


def chi(R: float, depth, thickness: float):
    """Contact radius over layer thickness."""
    return np.sqrt(R * np.asarray(depth, dtype=float)) / thickness


def correction(R: float, depth, thickness: float):
    """Factor by which a bonded incompressible layer exceeds the Hertz force."""
    x = chi(R, depth, thickness)
    return sum(c * x**k for k, c in enumerate(BONDED))


def force(R: float, depth, mu: float, thickness: float):
    """Indentation force for a bonded incompressible layer of shear modulus mu."""
    return hertz.force(R, depth, hertz.shear_to_young(mu, 0.5), 0.5) * correction(R, depth, thickness)


if __name__ == "__main__":
    R, mu, t = 4.0, 0.05, 4.0
    for d in (0.1, 0.3, 0.5, 1.0):
        print(f"depth={d:.1f} mm  chi={float(chi(R, d, t)):.3f}  "
              f"factor={float(correction(R, d, t)):.3f}  F={float(force(R, d, mu, t)):.4f} N")
