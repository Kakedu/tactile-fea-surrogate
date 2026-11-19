"""Run with: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fea"))
sys.path.insert(0, str(ROOT / "materials"))

import hertz  # noqa: E402
import fit_hyperelastic as fh  # noqa: E402
import run_batch  # noqa: E402


def test_hertz_force_scales_with_depth_to_three_halves():
    f1, f2 = hertz.force(4.0, 0.2, 0.15), hertz.force(4.0, 0.8, 0.15)
    assert np.isclose(f2 / f1, 4.0**1.5)


def test_hertz_surface_is_continuous_at_contact_edge():
    R, d = 4.0, 0.5
    a = hertz.contact_radius(R, d)
    inside, outside = hertz.surface_deflection([a * (1 - 1e-9), a * (1 + 1e-9)], R, d)
    assert np.isclose(inside, outside, atol=1e-6)
    assert np.isclose(inside, d / 2)


def test_hertz_pressure_integrates_to_force():
    R, d, E = 4.0, 0.5, 0.15
    a, p0 = hertz.contact_radius(R, d), hertz.peak_pressure(R, d, E)
    r = np.linspace(0, a, 20001)
    p = p0 * np.sqrt(1 - (r / a) ** 2)
    assert np.isclose(np.trapezoid(p * 2 * np.pi * r, r), hertz.force(R, d, E), rtol=1e-4)


def test_models_agree_with_small_strain_limit():
    # every model must reduce to P = 3 mu (L - 1) near L = 1
    L, mu = 1.0 - 1e-4, 0.05
    for P in (fh.neo_hookean(L, mu), fh.yeoh(L, mu / 2, 0.0, 0.0), fh.ogden1(L, mu, 2.0)):
        assert np.isclose(P, 3 * mu * (L - 1), rtol=1e-3)


def test_fit_recovers_known_neo_hookean():
    L = np.linspace(0.7, 0.99, 30)
    res = fh.fit(L, fh.neo_hookean(L, 0.05), "neo_hookean")
    assert np.isclose(res["params"][0], 0.05, rtol=1e-4)


def test_case_check_rejects_bad_parameters():
    for bad in ((4.0, 5.0), (4.0, 3.0), (-1.0, 0.1)):
        try:
            run_batch.check_case(*bad, layer_thickness=4.0)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
