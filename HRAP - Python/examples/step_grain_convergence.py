# Purpose: Grid-convergence report for a 3D grain imported from STEP (or STL).
# Usage:   python step_grain_convergence.py [grain.step|grain.stl] [--units mm] [--axis z]
#          [--mdot-ox 1.0] [--t-end 5.0] [--res 32,48,64,96]
# With no file, a twisted 6-point star grain (80 mm OD, 300 mm, one full twist) is generated.
import argparse
from pathlib import Path

import numpy as np

from hrap.burn3d import cad, RegressionLaw
from hrap.burn3d.convergence import grid_convergence

ap = argparse.ArgumentParser()
ap.add_argument('file', nargs='?')
ap.add_argument('--units', default=None)
ap.add_argument('--axis', default='z')
ap.add_argument('--mdot-ox', type=float, default=1.0)
ap.add_argument('--t-end', type=float, default=5.0)
ap.add_argument('--res', default='32,48,64,96')
ap.add_argument('--nz-ratio', type=float, default=0.75, help='axial stations per in-plane cell count')
args = ap.parse_args()

out = Path('results'); out.mkdir(exist_ok=True)
path = args.file
if path is None:
    path = out/'twisted_star_grain.step'
    cad.write_star_grain_step(path, 0.08, 0.3, 6, 0.012, 0.025, twist_turns=1.0, units='m')
    print('Generated', path)

# Validate once and print the mesh report
_, report = cad.load_grain(path, units=args.units, axis=args.axis, n_xy=16, n_z=4)
law = RegressionLaw()  # ABS defaults (a, n from hrap/resources/propellant_configs/ABS.mat), total flux
res = [int(r) for r in args.res.split(',')]
rep = grid_convergence(
    lambda n: cad.load_grain(path, units=args.units, axis=args.axis, n_xy=n,
                             n_z=max(4, int(round(args.nz_ratio*n))), verbose=False)[0],
    res, law, args.mdot_ox, args.t_end)
print(rep)
csv = out/(Path(path).stem + '_convergence.csv')
rep.to_csv(csv)
print('Wrote', csv)
