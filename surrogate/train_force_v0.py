"""v0 surrogate: predict contact force from (sphere radius, depth).

Deliberately small. It exists to prove the chain data -> model -> evaluation
works. The field-predicting model of week 11-13 is yours to write in PyTorch.

What it reports:
- error on an in-range test set
- error on an out-of-range test set (deeper than anything seen in training)
- the same numbers for a linear-regression baseline
- time per prediction next to the solver's time per case

Usage:
    python train_force_v0.py ../data/dataset_hertz.npz --out force_v0.joblib
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def rel_err(pred, true):
    """Mean and worst relative error, in percent."""
    e = np.abs(pred - true) / np.abs(true)
    return float(e.mean() * 100), float(e.max() * 100)


def split(params, force, ood_quantile=0.85, test_frac=0.2, seed=0):
    """Hold out the deepest cases as out-of-range; split the rest at random."""
    depth = params[:, 1]
    ood = depth > np.quantile(depth, ood_quantile)
    idx = np.flatnonzero(~ood)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    return idx[n_test:], idx[:n_test], np.flatnonzero(ood)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="force_v0.joblib")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.dataset)
    X, y = data["params"], data["force"]
    meta = json.loads(str(data["meta"]))
    tr, te, ood = split(X, y, seed=args.seed)
    print(f"{len(tr)} train, {len(te)} in-range test, {len(ood)} out-of-range test "
          f"(depth > {X[ood, 1].min():.3f} mm)")

    # Work in log space on both sides: force spans orders of magnitude, must stay
    # positive, and contact laws are close to power laws. L-BFGS suits small data.
    mlp = make_pipeline(FunctionTransformer(np.log), StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(32, 32), solver="lbfgs", max_iter=5000,
                                     random_state=args.seed))
    mlp.fit(X[tr], np.log(y[tr]))
    lin = LinearRegression().fit(X[tr], y[tr])

    def predict_mlp(a):
        return np.exp(mlp.predict(a))

    print(f"{'model':<18}{'in-range mean / max %':<28}{'out-of-range mean / max %'}")
    for name, fn in (("MLP (2x32)", predict_mlp), ("linear baseline", lin.predict)):
        a, b = rel_err(fn(X[te]), y[te]), rel_err(fn(X[ood]), y[ood])
        print(f"{name:<18}{a[0]:>7.2f} / {a[1]:<17.2f}{b[0]:>7.2f} / {b[1]:.2f}")

    t0 = time.perf_counter()
    for _ in range(20):
        predict_mlp(X[te])
    per_pred = (time.perf_counter() - t0) / (20 * len(te))
    per_solve = float(np.mean(data["solve_seconds"]))
    print(f"surrogate {per_pred * 1e6:.1f} us per case; {meta['backend']} solver {per_solve * 1e3:.3f} ms per case")
    if meta["backend"] == "hertz":
        print("note: the hertz backend is a closed-form formula, so this timing says nothing "
              "about the speed-up over FEA. Re-run on the mapdl dataset for that.")

    joblib.dump({"model": mlp, "log_target": True, "meta": meta}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
