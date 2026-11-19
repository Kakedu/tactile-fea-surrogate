"""Write SYNTHETIC example files so every script can be tried before real data exists.

Nothing here was measured. Delete these files once you have your own readings,
and never show a figure made from them as a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fea"))
sys.path.insert(0, str(HERE.parent / "materials"))
import hertz  # noqa: E402
import fit_hyperelastic as fh  # noqa: E402

rng = np.random.default_rng(1)

# compression test on a 20 mm tall, 15 mm diameter cylinder, Yeoh material + noise
height, diameter = 20.0, 15.0
disp = np.tile(np.linspace(0.5, 6.0, 12), 3)
stress = fh.yeoh(1 - disp / height, 0.025, -0.002, 0.004)
force = -stress * np.pi * (diameter / 2) ** 2 * (1 + rng.normal(0, 0.02, disp.size))
with open(HERE / "example_synthetic_compression.csv", "w") as f:
    f.write("specimen,displacement_mm,force_N\n")
    for i, (d, p) in enumerate(zip(disp, force)):
        f.write(f"S{i // 12 + 1},{d:.3f},{p:.4f}\n")

# indentation with a 4 mm sphere; a layer stiffer than half-space by a depth-dependent factor
R, mu = 4.0, 0.05
with open(HERE / "example_synthetic_indentation.csv", "w") as f:
    f.write("indenter,radius_mm,position,depth_mm,repeat,force_N,temperature_C,note\n")
    for depth in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        base = float(hertz.force(R, depth, hertz.shear_to_young(mu))) * (1 + 0.6 * depth)
        for rep in (1, 2, 3):
            f.write(f"ball_8mm,{R},centre,{depth},{rep},{base * (1 + rng.normal(0, 0.03)):.4f},22,SYNTHETIC\n")

print("wrote synthetic examples")
