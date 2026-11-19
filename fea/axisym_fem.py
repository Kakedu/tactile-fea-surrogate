"""Open-source axisymmetric finite element model of sphere indentation.

Purpose: an independent check on the ANSYS model, and a way for anyone to
reproduce the dataset without an ANSYS licence. It is not a replacement for
the ANSYS model.

What it solves
- A silicone layer, radius L and thickness t, bonded to a rigid base.
- A rigid, frictionless sphere of radius R pushed in by a prescribed depth.
- Large deformation, compressible Neo-Hookean material:
      W = mu/2 (J^(-2/3) I1 - 3) + K/2 (J - 1)^2
  with K set from mu and Poisson's ratio nu (use nu close to 0.5 for silicone).

How
- Six-node (quadratic) triangles on a mesh graded towards the contact zone.
- Contact by a penalty on penetration, integrated along the top surface.
- Residual and tangent stiffness from automatic differentiation (JAX) of the
  element energy, assembled into a sparse matrix, solved with Newton's method
  and a backtracking line search, loading the depth in steps.

Limits you must state when you use its results
- Frictionless and axisymmetric only: no shear, no off-axis or edge indenters.
- Penalty contact allows a small penetration, reported in the output.
- Displacement-only elements stiffen as nu -> 0.5; keep nu <= 0.495 and
  check mesh convergence (see verify_fem.py).

Units: mm, N, MPa.
"""
from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

# ---- quadrature -----------------------------------------------------------
_A1, _W1 = 0.445948490915965, 0.223381589678011
_A2, _W2 = 0.091576213509771, 0.109951743655322
# barycentric (L1, L2, L3) points of a degree-4 rule; weights sum to 1
TRI_L = np.array([[_A1, _A1, 1 - 2 * _A1], [_A1, 1 - 2 * _A1, _A1], [1 - 2 * _A1, _A1, _A1],
                  [_A2, _A2, 1 - 2 * _A2], [_A2, 1 - 2 * _A2, _A2], [1 - 2 * _A2, _A2, _A2]])
TRI_W = np.array([_W1] * 3 + [_W2] * 3) * 0.5  # reference triangle has area 1/2
GAUSS_S, GAUSS_W = np.polynomial.legendre.leggauss(4)


def _tri_shape(L):
    """Quadratic triangle: values N (6,) and reference gradients dN (6, 2)."""
    L1, L2, L3 = L
    d1, d2, d3 = np.array([-1.0, -1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    N = np.array([L1 * (2 * L1 - 1), L2 * (2 * L2 - 1), L3 * (2 * L3 - 1), 4 * L1 * L2, 4 * L2 * L3, 4 * L3 * L1])
    dN = np.array([(4 * L1 - 1) * d1, (4 * L2 - 1) * d2, (4 * L3 - 1) * d3,
                   4 * (L2 * d1 + L1 * d2), 4 * (L3 * d2 + L2 * d3), 4 * (L1 * d3 + L3 * d1)])
    return N, dN


_TRI_N = jnp.array([_tri_shape(L)[0] for L in TRI_L])
_TRI_DN = jnp.array([_tri_shape(L)[1] for L in TRI_L])
_TRI_WJ = jnp.array(TRI_W)
# quadratic line element on s in [-1, 1], nodes ordered (start, middle, end)
_EDGE_N = jnp.array([[s * (s - 1) / 2, 1 - s * s, s * (s + 1) / 2] for s in GAUSS_S])
_EDGE_W = jnp.array(GAUSS_W)


# ---- mesh -----------------------------------------------------------------
def make_mesh(L: float, t: float, nr: int, nz: int, grade_r: float = 1.8, grade_z: float = 1.5):
    """Graded mesh of quadratic triangles. Fine near the axis and the top surface.

    Returns nodes (n, 2) as (r, z) with the top surface at z = 0, triangles (m, 6),
    top-surface edges (k, 3), and index arrays for the boundary conditions.
    """
    def with_midpoints(corners):
        out = np.empty(2 * len(corners) - 1)
        out[0::2] = corners
        out[1::2] = 0.5 * (corners[:-1] + corners[1:])
        return out

    r = with_midpoints(L * (np.arange(nr + 1) / nr) ** grade_r)
    z = with_midpoints(-t * (np.arange(nz + 1) / nz) ** grade_z)
    ni, nj = len(r), len(z)
    nodes = np.array([[ri, zj] for ri in r for zj in z])

    def idx(i, j):
        return i * nj + j

    tris, edges = [], []
    for a in range(nr):
        i = 2 * a
        edges.append([idx(i, 0), idx(i + 1, 0), idx(i + 2, 0)])
        for b in range(nz):
            j = 2 * b
            tris.append([idx(i, j), idx(i + 2, j), idx(i + 2, j + 2),
                         idx(i + 1, j), idx(i + 2, j + 1), idx(i + 1, j + 1)])
            tris.append([idx(i, j), idx(i + 2, j + 2), idx(i, j + 2),
                         idx(i + 1, j + 1), idx(i + 1, j + 2), idx(i, j + 1)])
    return {"nodes": nodes, "tris": np.array(tris), "edges": np.array(edges),
            "axis": np.array([idx(0, j) for j in range(nj)]),
            "bottom": np.array([idx(i, nj - 1) for i in range(ni)]),
            "top": np.array([idx(i, 0) for i in range(ni)])}


# ---- energies -------------------------------------------------------------
def _element_energy(u, X, mu, K):
    """Strain energy of one triangle, integrated over the full revolution."""
    def at_point(N, dN, w):
        jac = X.T @ dN                      # d(r, z) / d(xi, eta)
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        grad = dN @ jnp.linalg.inv(jac)     # shape-function gradients in (r, z)
        r = N @ X[:, 0]
        H = u.T @ grad                      # displacement gradient
        F11, F12, F21, F22 = 1 + H[0, 0], H[0, 1], H[1, 0], 1 + H[1, 1]
        F33 = 1 + (N @ u[:, 0]) / r         # hoop stretch
        J = (F11 * F22 - F12 * F21) * F33
        I1 = F11**2 + F12**2 + F21**2 + F22**2 + F33**2
        W = 0.5 * mu * (J ** (-2.0 / 3.0) * I1 - 3.0) + 0.5 * K * (J - 1.0) ** 2
        return W * 2 * jnp.pi * r * jnp.abs(det) * w
    return jnp.sum(jax.vmap(at_point)(_TRI_N, _TRI_DN, _TRI_WJ))


def _edge_energy(u, X, R, centre_z, penalty):
    """Penalty energy of one top-surface edge against the rigid sphere."""
    half = 0.5 * jnp.abs(X[2, 0] - X[0, 0])

    def at_point(N, w):
        r_ref = N @ X[:, 0]
        x = N @ (X + u)
        gap = jnp.sqrt(x[0] ** 2 + (x[1] - centre_z) ** 2 + 1e-30) - R
        pen = jnp.maximum(-gap, 0.0)
        return 0.5 * penalty * pen**2 * 2 * jnp.pi * r_ref * half * w, pen
    e, pen = jax.vmap(at_point)(_EDGE_N, _EDGE_W)
    return jnp.sum(e), jnp.max(pen)


@functools.lru_cache(maxsize=None)
def _compiled():
    elem = lambda u, X, mu, K: _element_energy(u, X, mu, K)
    edge = lambda u, X, R, cz, pen: _edge_energy(u, X, R, cz, pen)[0]
    return {
        "elem_e": jax.jit(jax.vmap(elem, (0, 0, None, None))),
        "elem_g": jax.jit(jax.vmap(jax.grad(elem), (0, 0, None, None))),
        "elem_h": jax.jit(jax.vmap(jax.hessian(elem), (0, 0, None, None))),
        "edge_e": jax.jit(jax.vmap(edge, (0, 0, None, None, None))),
        "edge_g": jax.jit(jax.vmap(jax.grad(edge), (0, 0, None, None, None))),
        "edge_h": jax.jit(jax.vmap(jax.hessian(edge), (0, 0, None, None, None))),
        "edge_pen": jax.jit(jax.vmap(lambda u, X, R, cz, pen: _edge_energy(u, X, R, cz, pen)[1],
                                     (0, 0, None, None, None))),
    }


# ---- solver ---------------------------------------------------------------
def solve_indentation(R: float, depth: float, mu: float, layer_thickness: float = 4.0,
                      layer_radius: float = 12.0, nu: float = 0.49, nr: int = 36, nz: int = 20,
                      penalty: float | None = None, n_steps: int | None = None,
                      tol: float = 1e-8, max_iter: int = 40, verbose: bool = False):
    """Return force, surface deflection and diagnostics for one indentation.

    {"force": N, "r": mm, "uz": mm (positive = pushed down), "max_penetration": mm,
     "newton_iterations": int, "n_dof": int}
    """
    fn = _compiled()
    mesh = make_mesh(layer_radius, layer_thickness, nr, nz)
    X = mesh["nodes"]
    tris, edges = mesh["tris"], mesh["edges"]
    n = len(X)
    K = 2.0 * mu * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
    if penalty is None:
        penalty = 4.0e3 * mu  # MPa/mm: penetration stays near 0.1 % of typical depths
    if n_steps is None:
        n_steps = max(6, int(np.ceil(depth / 0.05)))

    fixed = np.zeros((n, 2), dtype=bool)
    fixed[mesh["axis"], 0] = True      # symmetry axis: no radial motion
    fixed[mesh["bottom"], :] = True    # bonded to the rigid base
    free = np.flatnonzero(~fixed.ravel())

    Xt, Xe = jnp.array(X[tris]), jnp.array(X[edges])
    tri_dof = (2 * tris[:, :, None] + np.arange(2)).reshape(len(tris), 12)
    edge_dof = (2 * edges[:, :, None] + np.arange(2)).reshape(len(edges), 6)
    rows_t, cols_t = np.repeat(tri_dof, 12, axis=1).ravel(), np.tile(tri_dof, (1, 12)).ravel()
    rows_e, cols_e = np.repeat(edge_dof, 6, axis=1).ravel(), np.tile(edge_dof, (1, 6)).ravel()

    def energy(u, cz):
        return float(jnp.sum(fn["elem_e"](u[tris], Xt, mu, K)) +
                     jnp.sum(fn["edge_e"](u[edges], Xe, R, cz, penalty)))

    def contact_gradient(u, cz):
        g = np.zeros(2 * n)
        np.add.at(g, edge_dof.ravel(), np.asarray(fn["edge_g"](u[edges], Xe, R, cz, penalty)).ravel())
        return g

    def gradient(u, cz):
        g = contact_gradient(u, cz)
        np.add.at(g, tri_dof.ravel(), np.asarray(fn["elem_g"](u[tris], Xt, mu, K)).ravel())
        return g

    def stiffness(u, cz):
        ht = np.asarray(fn["elem_h"](u[tris], Xt, mu, K)).reshape(len(tris), -1).ravel()
        he = np.asarray(fn["edge_h"](u[edges], Xe, R, cz, penalty)).reshape(len(edges), -1).ravel()
        return sp.coo_matrix((np.concatenate([ht, he]),
                              (np.concatenate([rows_t, rows_e]), np.concatenate([cols_t, cols_e]))),
                             shape=(2 * n, 2 * n)).tocsr()

    force_scale = mu * R * R

    def equilibrate(u, cz):
        """Newton iterations at a fixed sphere position. Returns (u, iterations) or None."""
        for it in range(max_iter):
            g = gradient(jnp.array(u), cz)
            res = np.linalg.norm(g[free]) / force_scale
            if verbose:
                print(f"    iter {it:2d} residual {res:.2e}")
            if res < tol:
                return u, it
            Kff = stiffness(jnp.array(u), cz)[free][:, free]
            du = np.zeros(2 * n)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", spla.MatrixRankWarning)
                du[free] = spla.spsolve(Kff.tocsc(), -g[free])
            if not np.all(np.isfinite(du)):
                return None                 # singular tangent: let the caller cut the increment
            e0, alpha = energy(jnp.array(u), cz), 1.0
            while True:                     # backtrack if the step inverts an element or raises energy
                e1 = energy(jnp.array(u + alpha * du.reshape(n, 2)), cz)
                if np.isfinite(e1) and e1 <= e0 + 1e-12 * abs(e0) + 1e-14:
                    break
                alpha *= 0.5
                if alpha < 1e-4:
                    return None
            u = u + alpha * du.reshape(n, 2)
        return None

    # load the depth in increments; halve an increment that fails to converge
    u = np.zeros((n, 2))
    total_iters, done, inc = 0, 0.0, 1.0 / n_steps
    while done < 1.0 - 1e-12:
        target = min(1.0, done + inc)
        if verbose:
            print(f"  depth fraction {target:.4f}")
        result = equilibrate(u, R - depth * target)
        if result is None:
            inc *= 0.5
            if inc < 1.0 / (n_steps * 64):
                raise RuntimeError(f"Newton did not converge near depth fraction {target:.4f}")
            continue
        u, iters = result
        total_iters += iters
        done = target
        if iters <= 6:
            inc = min(inc * 1.5, 1.0 / n_steps)

    cz = R - depth
    uj = jnp.array(u)
    force = float(contact_gradient(uj, cz).reshape(n, 2)[:, 1].sum())
    top = mesh["top"]
    return {"force": force, "r": X[top, 0], "uz": -u[top, 1],
            "max_penetration": float(jnp.max(fn["edge_pen"](uj[edges], Xe, R, cz, penalty))),
            "newton_iterations": total_iters, "n_dof": int(len(free))}


if __name__ == "__main__":
    out = solve_indentation(R=4.0, depth=0.3, mu=0.05, verbose=True)
    print({k: v for k, v in out.items() if k not in ("r", "uz")})
