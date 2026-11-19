# Tactile sensor silicone layer: FEA to surrogate model

**Author:** [Haoran_Du] · [duhaoran20021212@foxmail.com]

**Status: in progress, started November 2026. No results yet.**
中文说明见 [README_zh.md](README_zh.md).

Vision-based tactile sensors read contact through the deformation of a soft
silicone layer. Simulating that deformation with finite elements is accurate but
slow, which is a problem when a robot policy needs thousands of contacts per
second in simulation. This project builds and validates a fast replacement:

1. **Calibrate** the silicone from compression tests on specimens cast from the same batch.
2. **Simulate** hyperelastic contact in ANSYS, verified against the Hertz solution and a mesh-convergence study.
3. **Learn** a surrogate that predicts the surface displacement field and the contact force.
4. **Validate** FEA and surrogate against indentation measurements on a self-built sensor.
5. **Report where the model fails**, with data, not only where it works.

## How this differs from the closest prior work

[FEM-Validated Real-Time Modelling of Soft Tactile Deformation](https://arxiv.org/pdf/2610.16667)
(October 2026) follows the same idea: validate an ANSYS model by experiment, then
train a surrogate on it. That paper studies a 2D planar TPU structure, takes its
material parameters from a comparable material, and lists 3D sensors and force
estimation as future work. This project targets a 3D silicone gel layer, calibrates
the material on the sensor's own silicone, predicts force as well as displacement,
and publishes code, data and failure cases.

## Roadmap

| Version | Target date | Contents | Done when |
| --- | --- | --- | --- |
| v0 minimal loop | 10 Jan 2027 | 2D sphere indentation FEA, force measured on a silicone block, small force model | three force-depth curves on one figure |
| v1 FEA + surrogate | 28 Feb 2027 | calibrated, converged 3D model; dataset; surrogate for displacement field and force | speed-up, error and range of validity stated in one sentence |
| v2 full release | 23 May 2027 | comparison against a self-built sensor, failure table, article, video | a newcomer can restate the conclusion after 3 minutes with this README |

## Repository layout

```
fea/          Hertz solution, PyMAPDL contact model, batch dataset generation
materials/    hyperelastic model fitting from compression tests
surrogate/    surrogate models (v0: force only)
experiments/  measurement templates, comparison scripts, synthetic examples
docs/         experiment protocol, shopping list, development log
tests/        unit tests for the analytical parts
```

## Try it without ANSYS

Everything below runs on synthetic data, so it shows that the pipeline works
and nothing else. Files named `example_synthetic_*` are not measurements.

```bash
pip install -r requirements.txt
python -m pytest tests
python experiments/make_synthetic_examples.py
cd fea && python run_batch.py --backend hertz --n 300 --out ../data/dataset_hertz.npz && cd ..
cd surrogate && python train_force_v0.py ../data/dataset_hertz.npz && cd ..
cd materials && python fit_hyperelastic.py ../experiments/example_synthetic_compression.csv --height 20 --diameter 15 && cd ..
```

## What has and has not been verified

| Part | State |
| --- | --- |
| Hertz solution, model fitting, batch runner, v0 force model | run and unit-tested on synthetic data |
| `fea/axisym_indentation_pymapdl.py` | reference draft, never run against ANSYS |
| Material parameters, FEA results, measurements | none yet |

## Licence

To be chosen before the v1 release. Until then, all rights reserved.
