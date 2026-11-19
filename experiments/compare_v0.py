"""v0 figure: measured vs solver vs surrogate, force against depth for one sphere.

Measured CSV (see template_indentation.csv):
    indenter, radius_mm, position, depth_mm, repeat, force_N, temperature_C, note
Force from a scale: force_N = grams * 9.81e-3.

Usage:
    python compare_v0.py measured.csv --radius 4 --mu 0.05 \
        --surrogate ../surrogate/force_v0.joblib --out v0_comparison.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fea"))
import hertz  # noqa: E402


def load_measured(path, radius):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    d = d[np.isclose(d["radius_mm"], radius)]
    depths = np.unique(d["depth_mm"])
    mean = np.array([d["force_N"][d["depth_mm"] == x].mean() for x in depths])
    std = np.array([d["force_N"][d["depth_mm"] == x].std(ddof=1) for x in depths])
    synthetic = bool(np.any(np.char.upper(d["note"].astype(str)) == "SYNTHETIC"))
    return depths, mean, std, synthetic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--mu", type=float, required=True, help="calibrated initial shear modulus, MPa")
    ap.add_argument("--surrogate", default=None)
    ap.add_argument("--out", default="v0_comparison.png")
    args = ap.parse_args()

    depth, f_mean, f_std, synthetic = load_measured(args.csv, args.radius)
    grid = np.linspace(depth.min(), depth.max(), 100)
    f_solver = hertz.force(args.radius, grid, hertz.shear_to_young(args.mu))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(depth, f_mean, yerr=f_std, fmt="ko", capsize=3, label="measured (mean, 1 s.d.)")
    ax.plot(grid, f_solver, label="Hertz half-space")
    if args.surrogate:
        bundle = joblib.load(args.surrogate)
        pred = bundle["model"].predict(np.column_stack([np.full_like(grid, args.radius), grid]))
        ax.plot(grid, np.exp(pred) if bundle["log_target"] else pred, "--", label="surrogate v0")
    ax.set_xlabel("indentation depth (mm)")
    ax.set_ylabel("force (N)")
    ax.set_title(f"sphere R = {args.radius:g} mm" + ("  |  SYNTHETIC EXAMPLE, NOT A RESULT" if synthetic else ""))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)

    at_meas = hertz.force(args.radius, depth, hertz.shear_to_young(args.mu))
    err = (at_meas - f_mean) / f_mean * 100
    print("depth_mm  measured_N  solver_N  solver_vs_measured_%")
    for row in zip(depth, f_mean, at_meas, err):
        print("  {:.2f}     {:.4f}     {:.4f}    {:+.1f}".format(*row))
    print("saved", args.out)


if __name__ == "__main__":
    main()
