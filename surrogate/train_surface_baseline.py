"""Baseline field surrogate: predict the whole surface profile, not just the force.

Method: compress the surface profiles with PCA, then learn the map from
(sphere radius, depth) to the PCA coefficients with a small MLP. This is the
simplest thing that predicts a field. The model you write for v1 has to beat it.

Reports the relative L2 error of the predicted profile on an in-range test set
and on an out-of-range set (the deepest cases, never seen in training).

Usage:
    python train_surface_baseline.py ../data/fem_open_v0.npz
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from train_force_v0 import split


def rel_l2(pred, true):
    """Relative L2 error per case, in percent."""
    return np.linalg.norm(pred - true, axis=1) / np.linalg.norm(true, axis=1) * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.dataset)
    X, S = data["params"], data["surface"]
    meta = json.loads(str(data["meta"]))
    tr, te, ood = split(X, data["force"], seed=args.seed)

    # profiles scale roughly with depth, so learn the shape of surface / depth
    shape = S / X[:, 1:2]
    pca = PCA(n_components=args.modes).fit(shape[tr])
    explained = pca.explained_variance_ratio_.sum() * 100
    mlp = make_pipeline(FunctionTransformer(np.log), StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 64), solver="lbfgs", max_iter=5000,
                                     random_state=args.seed))
    coef_scale = pca.transform(shape[tr]).std(axis=0)
    mlp.fit(X[tr], pca.transform(shape[tr]) / coef_scale)

    def predict(a):
        return pca.inverse_transform(mlp.predict(a) * coef_scale) * a[:, 1:2]

    print(f"{meta['backend']} dataset: {len(tr)} train, {len(te)} in-range test, {len(ood)} out-of-range test")
    print(f"{args.modes} PCA modes keep {explained:.3f}% of the variance")
    for name, idx in (("in-range", te), ("out-of-range", ood)):
        e = rel_l2(predict(X[idx]), S[idx])
        print(f"{name:<13} relative L2 error: mean {e.mean():.2f}%  max {e.max():.2f}%")

    t0 = time.perf_counter()
    for _ in range(20):
        predict(X[te])
    per_pred = (time.perf_counter() - t0) / (20 * len(te))
    per_solve = float(np.mean(data["solve_seconds"]))
    print(f"surrogate {per_pred * 1e6:.0f} us per case; {meta['backend']} solver {per_solve:.2f} s per case")
    print("note: do not quote a speed-up from these two numbers. The hertz backend is a formula, and the "
          "fem backend is unoptimised Python. The speed-up that matters is against the ANSYS model.")


if __name__ == "__main__":
    main()
