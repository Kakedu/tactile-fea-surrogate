"""Figure: open-source FEM against the Hertz and thin-layer formulas.

Left: force against depth for one sphere on the default 4 mm bonded layer.
Right: surface profile at one depth, FEM against the Hertz half-space shape.

There is no measurement and no ANSYS result in this figure.

Usage:
    python plot_fem_vs_analytical.py --out ../docs/fem_vs_analytical.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fea"))
import hertz  # noqa: E402
import thin_layer  # noqa: E402
from axisym_fem import solve_indentation  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--thickness", type=float, default=4.0)
    ap.add_argument("--nu", type=float, default=0.49)
    ap.add_argument("--out", default="../docs/fem_vs_analytical.png")
    args = ap.parse_args()
    R, mu, t = args.radius, args.mu, args.thickness

    depths = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    runs = [solve_indentation(R, d, mu, layer_thickness=t, nu=args.nu) for d in depths]
    f_fem = np.array([r["force"] for r in runs])
    grid = np.linspace(0.02, depths.max(), 200)
    f_hertz = hertz.force(R, grid, hertz.shear_to_young(mu, 0.5), 0.5)
    f_layer = thin_layer.force(R, grid, mu, t)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(grid, f_hertz, label="Hertz, half-space")
    a.plot(grid, f_layer, label="Dimitriadis, bonded layer")
    a.plot(depths, f_fem, "ko", label=f"open-source FEM (nu = {args.nu})")
    a.set_xlabel("indentation depth (mm)")
    a.set_ylabel("force (N)")
    a.set_title(f"sphere R = {R:g} mm on a {t:g} mm bonded layer")
    a.legend()

    k = 2  # depth 0.3 mm
    r, uz = runs[k]["r"], runs[k]["uz"]
    rr = np.linspace(0, 6, 300)
    b.plot(rr, -hertz.surface_deflection(rr, R, depths[k]), label="Hertz, half-space")
    b.plot(r, -uz, "k.", ms=4, label="open-source FEM")
    b.axvline(hertz.contact_radius(R, depths[k]), color="0.6", ls=":", label="Hertz contact radius")
    b.set_xlim(0, 6)
    b.set_xlabel("radius (mm)")
    b.set_ylabel("surface displacement (mm)")
    b.set_title(f"surface profile at depth {depths[k]:g} mm")
    b.legend()
    fig.suptitle("Simulation only: no measurement, no ANSYS result", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)

    print("depth_mm  fem_N     vs_hertz_%  vs_dimitriadis_%")
    for d, f in zip(depths, f_fem):
        h = float(hertz.force(R, d, hertz.shear_to_young(mu, 0.5), 0.5))
        dm = float(thin_layer.force(R, d, mu, t))
        print(f"  {d:<7} {f:<9.5f} {(f - h) / h * 100:+9.1f}  {(f - dm) / dm * 100:+9.1f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
