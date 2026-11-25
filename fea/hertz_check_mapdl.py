"""First ANSYS run of the contact model: a thick layer (close to a half-space) against Hertz.

Pass: FEA force within 5 % of Hertz at the smallest depth. At larger depth FEA may deviate
(large strain, finite layer); that deviation is reported, not hidden.
Thick layer: t = L = 20 mm vs contact radius 0.4-0.9 mm, so the half-space assumption nearly holds.

Usage (from the repo root):
    python fea/hertz_check_mapdl.py --exec "E:/ANSYS Inc/v251/ansys/bin/winx64/ANSYS251.exe"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hertz  # noqa: E402
from axisym_indentation_pymapdl import solve_indentation  # noqa: E402

R, MU, NU = 4.0, 0.05, 0.499          # TEST values (mu = soft silicone, as in hertz.py)
DEPTHS = (0.05, 0.1, 0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", required=True)
    ap.add_argument("--esize", type=float, default=0.025, help="element size near the contact (mm)")
    ap.add_argument("--lagrange", action="store_true", help="Lagrange contact (no penetration)")
    args = ap.parse_args()
    from ansys.mapdl.core import launch_mapdl

    mapdl = launch_mapdl(exec_file=args.exec, run_location=os.path.abspath("mapdl_run"),
                         override=True, loglevel="ERROR")
    flip = False
    try:
        print(f"R {R} mm, mu {MU} MPa, nu {NU}, layer 20 x 20 mm, element {args.esize} mm near contact")
        print("depth   a_Hertz   F_FEA(N)    F_Hertz(N)  FEA/Hertz")
        for d in DEPTHS:
            kw = dict(R=R, depth=d, mu=MU, nu=NU, layer_thickness=20.0, layer_radius=20.0,
                      elem_size=args.esize, refine=2.0, far_size=0.5, n_substeps=10, mapdl=mapdl,
                      lagrange=args.lagrange)
            out = solve_indentation(flip_target=flip, **kw)
            if abs(out["force"]) < 1e-9 and not flip:          # no contact found: normals were wrong
                print("  no contact with the default target normals -> reversing them")
                flip = True
                out = solve_indentation(flip_target=flip, **kw)
            fh = float(hertz.force(R, d, hertz.shear_to_young(MU, NU), NU))
            print(f"{d:5.2f}   {hertz.contact_radius(R, d):7.3f}   {out['force']:10.5f}  {fh:10.5f}  "
                  f"{out['force'] / fh:8.3f}")
        r, uz = out["r"], out["uz"]
        uz_h = hertz.surface_deflection(r, R, DEPTHS[-1])
        m = r < 3.0
        print(f"surface deflection at depth {DEPTHS[-1]}: max |FEA - Hertz| within r < 3 mm = "
              f"{np.max(np.abs(uz[m] - uz_h[m])) * 1e3:.1f} um")
        print(f"target normals reversed: {flip}; Lagrange contact: {args.lagrange}")
    finally:
        mapdl.exit()


if __name__ == "__main__":
    main()