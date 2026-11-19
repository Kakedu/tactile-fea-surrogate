"""Fit hyperelastic models to a uniaxial compression test (week 6-7).

Input CSV, one row per reading (see experiments/template_compression.csv):
    specimen, displacement_mm, force_N
plus the specimen height and diameter given on the command line.

Models (all incompressible, nominal stress P as a function of stretch L):
    Neo-Hookean  P = mu (L - L^-2)
    Yeoh (3)     P = 2 (L - L^-2) (C10 + 2 C20 (I1-3) + 3 C30 (I1-3)^2),  I1 = L^2 + 2/L
    Ogden (1)    P = (2 mu / alpha) (L^(alpha-1) - L^(-alpha/2-1))

Compression: L < 1 and P < 0.

Caveats you must write into the article:
- This assumes frictionless platens. Friction makes the specimen barrel and
  the fitted material look stiffer than it is. Lubricate the end faces.
- One loading mode cannot pin down a multi-parameter model. Prefer the model
  with the fewest parameters that fits, and check it against indentation data.

Usage:
    python fit_hyperelastic.py data.csv --height 20 --diameter 15
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import least_squares


def neo_hookean(L, mu):
    return mu * (L - L**-2)


def yeoh(L, c10, c20, c30):
    i1 = L**2 + 2.0 / L - 3.0
    return 2.0 * (L - L**-2) * (c10 + 2.0 * c20 * i1 + 3.0 * c30 * i1**2)


def ogden1(L, mu, alpha):
    return (2.0 * mu / alpha) * (L ** (alpha - 1.0) - L ** (-alpha / 2.0 - 1.0))


MODELS = {
    # name: (function, initial guess, lower bounds, upper bounds, initial shear modulus from params)
    "neo_hookean": (neo_hookean, [0.05], [1e-6], [100.0], lambda p: p[0]),
    "yeoh": (yeoh, [0.025, 0.0, 0.0], [1e-6, -10.0, -10.0], [100.0, 10.0, 10.0], lambda p: 2.0 * p[0]),
    "ogden1": (ogden1, [0.05, 2.0], [1e-6, 0.1], [100.0, 20.0], lambda p: p[0]),
}


def to_stretch_stress(displacement_mm, force_N, height_mm, diameter_mm):
    """Compression readings -> (stretch, nominal stress in MPa). Compression is negative stress."""
    area = np.pi * (diameter_mm / 2.0) ** 2
    stretch = 1.0 - np.asarray(displacement_mm, float) / height_mm
    stress = -np.asarray(force_N, float) / area
    return stretch, stress


def fit(stretch, stress, name: str):
    func, x0, lo, hi, mu0 = MODELS[name]
    res = least_squares(lambda p: func(stretch, *p) - stress, x0, bounds=(lo, hi))
    resid = func(stretch, *res.x) - stress
    rmse = float(np.sqrt(np.mean(resid**2)))
    rel = rmse / float(np.max(np.abs(stress)))
    return {"model": name, "params": res.x, "rmse_MPa": rmse, "rmse_rel": rel, "mu0_MPa": float(mu0(res.x))}


def fit_all(stretch, stress):
    return [fit(stretch, stress, name) for name in MODELS]


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return data["displacement_mm"], data["force_N"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--height", type=float, required=True, help="specimen height, mm")
    ap.add_argument("--diameter", type=float, required=True, help="specimen diameter, mm")
    ap.add_argument("--plot", default=None, help="save a comparison figure to this path")
    args = ap.parse_args()

    disp, frc = load_csv(args.csv)
    L, P = to_stretch_stress(disp, frc, args.height, args.diameter)
    results = fit_all(L, P)
    print(f"{len(L)} readings, stretch range {L.min():.3f} .. {L.max():.3f}")
    for r in results:
        params = ", ".join(f"{v:.5g}" for v in r["params"])
        print(f"{r['model']:<12} params=[{params}]  mu0={r['mu0_MPa'] * 1e3:.2f} kPa  "
              f"RMSE={r['rmse_MPa'] * 1e3:.3f} kPa ({r['rmse_rel'] * 100:.2f}% of max stress)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ls = np.linspace(L.min(), 1.0, 200)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(L, P * 1e3, "k.", label="measured")
        for r in results:
            ax.plot(Ls, MODELS[r["model"]][0](Ls, *r["params"]) * 1e3, label=r["model"])
        ax.set_xlabel("stretch (-)")
        ax.set_ylabel("nominal stress (kPa)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
