"""Checks of axisym_fem.py against known solutions. Prints a report.

1. Half-space limit against Hertz (thick, wide layer; small depth).
2. Bonded thin layer against the Dimitriadis correction (nu -> 0.5).
3. Mesh convergence.
4. Sensitivity to the contact penalty.

Run:  python verify_fem.py            (about two minutes on a laptop)
"""
from __future__ import annotations

import time

import hertz
import thin_layer
from axisym_fem import solve_indentation

MU, R = 0.05, 4.0


def pct(a, b):
    return (a - b) / b * 100.0


def main():
    print("1. Half-space limit vs Hertz (layer 40 mm thick, 40 mm radius, nu = 0.49)")
    print("   depth_mm  fem_N      hertz_N    diff_%")
    for d in (0.05, 0.1, 0.2):
        out = solve_indentation(R, d, MU, layer_thickness=40.0, layer_radius=40.0, nu=0.49, nr=48, nz=40)
        ref = float(hertz.force(R, d, hertz.shear_to_young(MU, 0.49), 0.49))
        print(f"   {d:<9} {out['force']:<10.5f} {ref:<10.5f} {pct(out['force'], ref):+.2f}")

    print("\n2. Bonded 4 mm layer vs Dimitriadis (which assumes nu = 0.5, small strain)")
    print("   depth_mm  chi    nu      fem_N      dimitriadis_N  diff_%   hertz_only_diff_%")
    for d in (0.1, 0.3, 0.5):
        ref = float(thin_layer.force(R, d, MU, 4.0))
        hz = float(hertz.force(R, d, hertz.shear_to_young(MU, 0.5), 0.5))
        for nu in (0.49, 0.495):
            out = solve_indentation(R, d, MU, nu=nu)
            print(f"   {d:<9} {float(thin_layer.chi(R, d, 4.0)):<6.3f} {nu:<7} {out['force']:<10.5f} "
                  f"{ref:<14.5f} {pct(out['force'], ref):+6.2f}   {pct(out['force'], hz):+.1f}")

    print("\n3. Mesh convergence (depth 0.3 mm, 4 mm layer, nu = 0.49)")
    print("   nr x nz   dof     force_N    change_%   seconds")
    prev = None
    for nr, nz in ((18, 10), (24, 14), (36, 20), (54, 30)):  # finest takes about a minute
        t0 = time.perf_counter()
        out = solve_indentation(R, 0.3, MU, nr=nr, nz=nz)
        change = "" if prev is None else f"{pct(out['force'], prev):+.3f}"
        print(f"   {nr}x{nz:<6} {out['n_dof']:<7} {out['force']:<10.6f} {change:<10} {time.perf_counter() - t0:.1f}")
        prev = out["force"]

    print("\n4. Contact penalty (depth 0.3 mm, default mesh)")
    print("   penalty_MPa_per_mm  force_N    max_penetration_mm")
    for scale in (0.25, 1.0, 4.0):
        out = solve_indentation(R, 0.3, MU, penalty=4.0e3 * MU * scale)
        print(f"   {4.0e3 * MU * scale:<19.0f} {out['force']:<10.6f} {out['max_penetration']:.2e}")


if __name__ == "__main__":
    main()
