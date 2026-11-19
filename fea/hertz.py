"""Hertz contact: rigid sphere pressed into an elastic half-space.

Used for two things in this project:
1. Checking the FEA model at small indentation depth (week 3).
2. A stand-in "solver" so the data and surrogate pipeline can be tested
   before the ANSYS model exists (backend="hertz" in run_batch.py).

Assumptions (know them before you compare against FEA or experiment):
- half-space: the silicone layer is much thicker than the contact radius
- small strain, linear elastic
- frictionless
A thin bonded layer is stiffer than Hertz predicts. That is physics, not a bug.

Units: mm, N, MPa throughout (1 MPa = 1 N/mm^2).
"""
from __future__ import annotations

import numpy as np


def effective_modulus(E: float, nu: float) -> float:
    """E* for a rigid indenter on an elastic body, in the units of E."""
    return E / (1.0 - nu**2)


def shear_to_young(mu: float, nu: float = 0.5) -> float:
    """Small-strain Young's modulus from the initial shear modulus."""
    return 2.0 * mu * (1.0 + nu)


def contact_radius(R: float, depth: float) -> float:
    """Contact radius a = sqrt(R * depth)."""
    return float(np.sqrt(R * depth))


def force(R: float, depth, E: float, nu: float = 0.5):
    """Normal force F = 4/3 * E* * sqrt(R) * depth^1.5."""
    depth = np.asarray(depth, dtype=float)
    return 4.0 / 3.0 * effective_modulus(E, nu) * np.sqrt(R) * depth**1.5


def peak_pressure(R: float, depth: float, E: float, nu: float = 0.5) -> float:
    """Maximum contact pressure p0 = 3F / (2 pi a^2)."""
    a = contact_radius(R, depth)
    return float(3.0 * force(R, depth, E, nu) / (2.0 * np.pi * a**2))


def surface_deflection(r, R: float, depth: float):
    """Downward deflection of the surface at radius r (positive = pushed down).

    Inside the contact the surface follows the sphere; outside it relaxes
    back towards zero.
    """
    r = np.abs(np.asarray(r, dtype=float))
    a = contact_radius(R, depth)
    uz = np.empty_like(r)
    inside = r <= a
    uz[inside] = depth - r[inside] ** 2 / (2.0 * R)
    ro = r[~inside]
    uz[~inside] = ((2.0 * a**2 - ro**2) * np.arcsin(a / ro) + a * np.sqrt(ro**2 - a**2)) / (np.pi * R)
    return uz


if __name__ == "__main__":
    mu = 0.05  # MPa, a soft silicone; replace with your calibrated value
    E = shear_to_young(mu)
    R = 4.0
    for d in (0.1, 0.3, 0.5, 1.0):
        print(f"R={R} mm  depth={d:.1f} mm  a={contact_radius(R, d):.3f} mm  "
              f"F={float(force(R, d, E)):.4f} N  p0={peak_pressure(R, d, E) * 1e3:.2f} kPa")
