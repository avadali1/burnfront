"""3D level-set burn simulation for hybrid rocket fuel grains (numpy/scipy only, no JAX).

Conventions
-----------
- Arrays are indexed (x, y, z); z is the port (flow) axis, z[0] is the head (injector) end.
- ``phi`` is a signed distance field of the burning front: negative in the port (gas),
  positive in fuel. ``chi`` is a static signed distance field of the inert envelope
  (case wall): negative inside. Fuel = {phi > 0} & {chi < 0}; port = {max(phi, chi) < 0}.
- The front regresses into the fuel at the local normal speed rdot(z):
  phi_t + rdot |grad phi| = 0, advanced with its exact Hopf-Lax solution (default) or WENO5.
  End faces are inhibited: only the port surface inside z in [z0, z0+L] burns, and the surface is
  continued past the ends for the distance field.

Verified accuracy (tests/test_burn3d.py): port areas, burn-through time and burned mass converge at
second order in the cell size; the initial burning perimeter at infinitely sharp corners is first
order (grid-scale clipping of the corner during the first cell of regression).
"""
from hrap.burn3d.grid import Grid3D, Grain3D
from hrap.burn3d.law import RegressionLaw
from hrap.burn3d import shapes, measure, flow, levelset, meshdist
from hrap.burn3d.burn import burn_prescribed, BurnResult
