"""Generate a dataset of indentation cases (week 4, then week 10).

Each case = one sphere radius and one indentation depth. For each case we store
the total force and the surface deflection on a fixed radial grid, so every
case has the same output shape whatever the mesh was.

Backends:
    hertz  analytical half-space solution. Runs anywhere, no licence needed.
           Use it to test the pipeline end to end.
    fem    the open-source finite element model in axisym_fem.py (needs jax).
           About 12 s per case; use --workers to run cases in parallel.
    mapdl  the PyMAPDL model in axisym_indentation_pymapdl.py (needs ANSYS).

A case that fails is logged and skipped; the batch keeps going.

Usage:
    python run_batch.py --backend hertz --n 200 --out ../data/dataset_hertz.npz
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import qmc

import hertz

R_GRID = np.linspace(0.0, 10.0, 128)  # mm, fixed output grid for the surface


def sample_cases(n: int, r_range, depth_range, seed: int = 0):
    """Latin hypercube over (sphere radius, depth)."""
    unit = qmc.LatinHypercube(d=2, seed=seed).random(n)
    lo = [r_range[0], depth_range[0]]
    hi = [r_range[1], depth_range[1]]
    return qmc.scale(unit, lo, hi)


def check_case(R: float, depth: float, layer_thickness: float):
    """Reject parameter combinations that make no physical sense."""
    if R <= 0 or depth <= 0:
        raise ValueError("radius and depth must be positive")
    if depth >= R:
        raise ValueError("depth must be smaller than the sphere radius")
    if depth > 0.5 * layer_thickness:
        raise ValueError("depth exceeds half the layer thickness")


def solve_hertz(R, depth, mu, **_):
    E = hertz.shear_to_young(mu)
    return float(hertz.force(R, depth, E)), hertz.surface_deflection(R_GRID, R, depth)


def solve_mapdl(R, depth, mu, layer_thickness, layer_radius, mapdl=None, **_):
    from axisym_indentation_pymapdl import solve_indentation

    out = solve_indentation(R=R, depth=depth, mu=mu, layer_thickness=layer_thickness,
                            layer_radius=layer_radius, mapdl=mapdl)
    surface = np.interp(R_GRID, out["r"], out["uz"], right=0.0)
    return out["force"], surface


def solve_fem(R, depth, mu, layer_thickness, layer_radius, nu=0.49, **_):
    from axisym_fem import solve_indentation

    out = solve_indentation(R=R, depth=depth, mu=mu, layer_thickness=layer_thickness,
                            layer_radius=layer_radius, nu=nu)
    surface = np.interp(R_GRID, out["r"], out["uz"], right=0.0)
    return out["force"], surface


SOLVERS = {"hertz": solve_hertz, "mapdl": solve_mapdl, "fem": solve_fem}


def run_case(job):
    """One case, safe to call in a worker process. Returns (index, result or error text, seconds)."""
    i, R, depth, backend, kwargs = job
    t0 = time.perf_counter()
    try:
        check_case(R, depth, kwargs["layer_thickness"])
        result = SOLVERS[backend](R=R, depth=depth, **kwargs)
    except Exception as exc:  # keep the batch alive, remember what failed
        return i, repr(exc), time.perf_counter() - t0
    return i, result, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(SOLVERS), default="hertz")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (fem backend only)")
    ap.add_argument("--nu", type=float, default=0.49, help="Poisson's ratio (fem backend only)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mu", type=float, default=0.05, help="initial shear modulus, MPa")
    ap.add_argument("--r-range", type=float, nargs=2, default=[2.0, 6.0], help="sphere radius range, mm")
    ap.add_argument("--depth-range", type=float, nargs=2, default=[0.05, 1.0], help="depth range, mm")
    ap.add_argument("--layer-thickness", type=float, default=4.0)
    ap.add_argument("--layer-radius", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../data/dataset.npz")
    args = ap.parse_args()

    cases = sample_cases(args.n, args.r_range, args.depth_range, args.seed)
    kwargs = {"mu": args.mu, "layer_thickness": args.layer_thickness, "layer_radius": args.layer_radius}
    if args.backend == "fem":
        kwargs["nu"] = args.nu

    mapdl = None
    if args.backend == "mapdl":
        from ansys.mapdl.core import launch_mapdl
        mapdl = launch_mapdl()
        kwargs["mapdl"] = mapdl

    jobs = [(i, float(R), float(depth), args.backend, kwargs) for i, (R, depth) in enumerate(cases)]
    try:
        if args.backend == "fem" and args.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(args.workers) as pool:
                results = list(pool.imap(run_case, jobs))
        else:
            results = [run_case(job) for job in jobs]
    finally:
        if mapdl is not None:
            mapdl.exit()

    params, forces, surfaces, seconds, failures = [], [], [], [], []
    for (i, R, depth, _, _), (_, result, secs) in zip(jobs, results):
        if isinstance(result, str):
            failures.append({"case": i, "R": R, "depth": depth, "error": result})
            continue
        params.append([R, depth])
        forces.append(result[0])
        surfaces.append(result[1])
        seconds.append(secs)

    meta = {"backend": args.backend, "mu_MPa": args.mu, "seed": args.seed,
            "nu": args.nu if args.backend == "fem" else None,
            "r_range": args.r_range, "depth_range": args.depth_range,
            "layer_thickness": args.layer_thickness, "layer_radius": args.layer_radius,
            "n_requested": args.n, "n_ok": len(forces), "failures": failures}
    np.savez_compressed(args.out, params=np.array(params), force=np.array(forces),
                        surface=np.array(surfaces), r_grid=R_GRID,
                        solve_seconds=np.array(seconds), meta=json.dumps(meta))
    print(f"{len(forces)}/{args.n} cases solved, {len(failures)} failed -> {args.out}")
    if seconds:
        print(f"mean solve time {np.mean(seconds) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
