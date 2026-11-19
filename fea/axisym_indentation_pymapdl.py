"""2D axisymmetric model: rigid sphere pressed into a bonded silicone layer.

STATUS: REFERENCE DRAFT, NEVER RUN. It was written without access to ANSYS.
The command sequence follows standard MAPDL practice, but expect to debug it
in week 3-4. Things most likely to need fixing are marked CHECK.

Build the model by hand in the GUI first (week 3). Then use this file to
compare against your own script, not instead of writing one.

Geometry (x = radial, y = axial, axis of symmetry is the Y axis):
    silicone layer: 0 <= x <= layer_radius, -layer_thickness <= y <= 0
    sphere: radius R, lowest point touching the surface at (0, 0)

Units: mm, N, MPa.
"""
from __future__ import annotations

import numpy as np


def solve_indentation(R: float, depth: float, mu: float, layer_thickness: float = 4.0,
                      layer_radius: float = 12.0, nu: float = 0.499, friction: float = 0.0,
                      elem_size: float | None = None, n_substeps: int = 20, mapdl=None):
    """Return {"force": N, "r": mm array, "uz": mm array (positive = pushed down)}."""
    own = mapdl is None
    if own:
        from ansys.mapdl.core import launch_mapdl
        mapdl = launch_mapdl()
    try:
        return _solve(mapdl, R, depth, mu, layer_thickness, layer_radius, nu, friction,
                      elem_size or min(R, layer_thickness) / 20.0, n_substeps)
    finally:
        if own:
            mapdl.exit()


def _solve(mapdl, R, depth, mu, t, L, nu, friction, esize, nsub):
    mapdl.clear()
    mapdl.prep7()

    # --- silicone: 4-node axisymmetric solid, mixed u-P for near incompressibility
    mapdl.et(1, "PLANE182")
    mapdl.keyopt(1, 3, 1)  # axisymmetric
    mapdl.keyopt(1, 6, 1)  # mixed u-P formulation, avoids volumetric locking
    bulk = 2.0 * mu * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
    mapdl.tb("HYPER", 1, 1, 2, "NEO")
    mapdl.tbdata(1, mu, 2.0 / bulk)  # Neo-Hookean: initial shear modulus, d = 2/K

    mapdl.rectng(0, L, -t, 0)
    mapdl.esize(esize)  # uniform for now; refine the contact zone in week 8
    mapdl.type(1)
    mapdl.mat(1)
    mapdl.amesh("ALL")

    # --- contact pair: rigid target arc (sphere) + contact elements on the top surface
    mapdl.et(2, "TARGE169")
    mapdl.et(3, "CONTA171")  # 2-node contact, matches PLANE182
    mapdl.keyopt(3, 10, 2)   # update contact stiffness each iteration  CHECK
    mapdl.mp("MU", 2, friction)
    mapdl.r(2)               # real constant set 2 ties target and contact together

    # arc from the bottom of the sphere up to 60 degrees
    k_center = mapdl.k("", 0, R)
    k_bottom = mapdl.k("", 0, 0)
    k_side = mapdl.k("", R * np.sin(np.pi / 3), R - R * np.cos(np.pi / 3))
    arc = mapdl.larc(k_bottom, k_side, k_center, R)
    mapdl.type(2)
    mapdl.real(2)
    mapdl.mat(2)
    mapdl.lesize(arc, "", "", 60)
    mapdl.lmesh(arc)
    # CHECK: target normals must point towards the silicone. If the solution
    # reports no contact, reverse them with ESURF,,REVERSE on the target elements.

    # pilot node carries the rigid-body motion and the reaction force
    pilot = int(mapdl.get_value("NODE", 0, "NUM", "MAXD")) + 1
    mapdl.n(pilot, 0, R)
    mapdl.tshap("PILO")
    mapdl.e(pilot)

    # contact elements on the top surface nodes of the silicone
    mapdl.esel("S", "TYPE", "", 1)
    mapdl.nsle("S")
    mapdl.nsel("R", "LOC", "Y", 0)
    mapdl.type(3)
    mapdl.real(2)
    mapdl.mat(2)
    mapdl.esurf()
    mapdl.allsel()

    # --- boundary conditions
    mapdl.nsel("S", "LOC", "Y", -t)
    mapdl.d("ALL", "UX", 0)
    mapdl.d("ALL", "UY", 0)      # bonded to the rigid base
    mapdl.nsel("S", "LOC", "X", 0)
    mapdl.d("ALL", "UX", 0)      # symmetry axis
    mapdl.allsel()
    mapdl.d(pilot, "UX", 0)
    mapdl.d(pilot, "ROTZ", 0)
    mapdl.d(pilot, "UY", -depth)  # displacement control, more stable than force control
    mapdl.finish()

    # --- solve
    mapdl.slashsolu()
    mapdl.antype("STATIC")
    mapdl.nlgeom("ON")
    mapdl.nsubst(nsub, 1000, max(5, nsub // 4))
    mapdl.outres("ALL", "LAST")
    mapdl.solve()
    mapdl.finish()

    # --- results. Axisymmetric forces are reported for the full 360 degrees.
    mapdl.post1()
    mapdl.set("LAST")
    force = -float(mapdl.get_value("NODE", pilot, "RF", "FY"))  # CHECK sign convention

    mapdl.esel("S", "TYPE", "", 1)
    mapdl.nsle("S")
    mapdl.nsel("R", "LOC", "Y", 0)
    r = np.asarray(mapdl.mesh.nodes)[:, 0]
    uy = np.asarray(mapdl.post_processing.nodal_displacement("Y"))
    mapdl.allsel()
    order = np.argsort(r)
    return {"force": force, "r": r[order], "uz": -uy[order]}


if __name__ == "__main__":
    out = solve_indentation(R=4.0, depth=0.3, mu=0.05)
    print("force (N):", out["force"])
