"""Analytic port shapes.

A Shape provides
  f(X, Y, Z)  -> implicit function, negative inside the port (used for inside/outside sign), and
  loops(z, ds)-> list of counter-clockwise (N, 2) boundary polygons of the cross-section at z with
                 segment length <= ds (same vertex count for every z, so rings can be stitched).
The exact 3D signed distance is computed to a fine ruled-surface mesh stitched from the loops
(see meshdist), in a band around the surface; for z-varying shapes (twist, helix) the slice-wise
f is NOT a 3D distance and is only used for the sign.
"""
import numpy as np

from hrap.sdf import sd_poly
from hrap.burn3d.grid import Grid3D, Grain3D
from hrap.burn3d.meshdist import signed_distance_on_grid


class Shape:
    def __init__(self, f, loops, exact_3d_sdf=False):
        self.f = f
        self.loops = loops
        self.exact_3d_sdf = exact_3d_sdf  # f is already an exact 3D signed distance

    def __call__(self, X, Y, Z):
        return self.f(X, Y, Z)


def _subdivide(v, ds):
    """Insert points so every edge of closed polygon v is at most ds long."""
    out = []
    n = len(v)
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / ds)))
        t = np.arange(m)[:, None] / m
        out.append(a + t*(b - a))
    return np.concatenate(out)


def circle(r, cx=0.0, cy=0.0):
    def f(X, Y, Z):
        return np.hypot(X - cx, Y - cy) - r

    def loops(z, ds):
        n = max(16, int(np.ceil(2*np.pi*r/ds)))
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        return [np.stack([cx + r*np.cos(t), cy + r*np.sin(t)], axis=1)]
    return Shape(f, loops, exact_3d_sdf=True)


def star_vertices(n_tips, r_valley, r_tip, rotation=0.0):
    """Counter-clockwise star polygon alternating tip/valley radii (tips at rotation + 2 pi k/n)."""
    r = np.tile([r_tip, r_valley], n_tips)
    t = rotation + np.linspace(0.0, 2*np.pi, 2*n_tips, endpoint=False)
    return np.stack([r*np.cos(t), r*np.sin(t)], axis=1)


def polygon(vertices):
    v = np.asarray(vertices, dtype=float)
    # Enforce counter-clockwise orientation (shoelace area > 0)
    area = 0.5*np.sum(v[:, 0]*np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1)*v[:, 1])
    if area < 0:
        v = v[::-1].copy()

    def f(X, Y, Z):
        p = np.stack([np.ravel(X), np.ravel(Y)], axis=-1)
        out = np.empty(p.shape[0])
        chunk = max(1, 4_000_000 // max(1, v.shape[0]))  # bound the (points x vertices) temporaries
        for s0 in range(0, p.shape[0], chunk):
            out[s0:s0 + chunk] = sd_poly(v, p[s0:s0 + chunk])
        return out.reshape(np.shape(X))

    def loops(z, ds):
        return [_subdivide(v, ds)]
    # A straight prism's in-slice distance equals its 3D distance
    return Shape(f, loops, exact_3d_sdf=True)


def star(n_tips, r_valley, r_tip, rotation=0.0):
    return polygon(star_vertices(n_tips, r_valley, r_tip, rotation))


def twist(shape, rate, z0=0.0):
    """Rotate the cross-section by ``rate`` rad per metre along z (positive = counter-clockwise)."""
    def f(X, Y, Z):
        a = -rate*(Z - z0)
        c, s = np.cos(a), np.sin(a)
        return shape.f(c*X - s*Y, s*X + c*Y, Z)

    def loops(z, ds):
        a = rate*(z - z0)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return [l @ R.T for l in shape.loops(z, ds)]
    return Shape(f, loops, exact_3d_sdf=(rate == 0.0 and shape.exact_3d_sdf))


def helix(shape, radius, pitch, phase=0.0):
    """Translate the cross-section around a helix of given radius and axial pitch (m per turn)."""
    def f(X, Y, Z):
        a = phase + 2*np.pi*Z/pitch
        return shape.f(X - radius*np.cos(a), Y - radius*np.sin(a), Z)

    def loops(z, ds):
        a = phase + 2*np.pi*z/pitch
        off = np.array([radius*np.cos(a), radius*np.sin(a)])
        return [l + off for l in shape.loops(z, ds)]
    return Shape(f, loops)


def union(*shapes):
    """Union of non-overlapping ports (multi-port grains)."""
    def f(X, Y, Z):
        return np.minimum.reduce([s.f(X, Y, Z) for s in shapes])

    def loops(z, ds):
        return [l for s in shapes for l in s.loops(z, ds)]
    return Shape(f, loops, exact_3d_sdf=all(s.exact_3d_sdf for s in shapes) and len(shapes) == 1)


def evaluate(f, grid):
    """Evaluate f slice-by-slice (bounded memory) -> (nx, ny, nz)."""
    X, Y = grid.xy_mesh()
    out = np.empty(grid.shape)
    for k, zk in enumerate(grid.z):
        out[:, :, k] = f(X, Y, np.full_like(X, zk))
    return out


def surface_mesh(shape, z_lo, z_hi, ds, dz):
    """Stitch the shape's loops at axial samples into a triangle mesh (open tubes)."""
    nzs = max(2, int(np.ceil((z_hi - z_lo)/dz)) + 1)
    zs = np.linspace(z_lo, z_hi, nzs)
    rings = [shape.loops(z, ds) for z in zs]
    V, F = [], []
    off = 0
    for li in range(len(rings[0])):
        n = rings[0][li].shape[0]
        for j, z in enumerate(zs):
            if rings[j][li].shape[0] != n:
                raise ValueError('loop vertex count must not vary with z')
            V.append(np.column_stack([rings[j][li], np.full(n, z)]))
        i = np.arange(n)
        i1 = (i + 1) % n
        for j in range(nzs - 1):
            a, b = off + j*n + i, off + j*n + i1
            c, d = off + (j + 1)*n + i, off + (j + 1)*n + i1
            F.append(np.stack([a, b, d], axis=1))
            F.append(np.stack([a, d, c], axis=1))
        off += n*nzs
    return np.concatenate(V), np.concatenate(F)


def port_sdf(shape, grid, chi, port_band_cells=3, mesh_refine=1):
    """Exact 3D signed distance to the port surface at every node that can matter.

    Exact on the whole fuel side inside the envelope (+3 cells, the front sweeps all of it) and
    within ``port_band_cells`` cells on the port side (cut cells at t = 0). Elsewhere only the sign
    is meaningful. Shapes flagged exact_3d_sdf (circles, straight prisms) are evaluated analytically.
    """
    if shape.exact_3d_sdf:
        return evaluate(shape.f, grid)
    hx, hy, hz = grid.h
    h = min(hx, hy)
    ds = h/mesh_refine
    dz = min(h, hz)/mesh_refine
    # Surface truncated at the end faces (inhibited ends): fuel near an end is reached only by the
    # port surface inside the grain, the same convention as CAD grains (hrap.burn3d.cad)
    V, F = surface_mesh(shape, grid.z[0], grid.z[-1], ds, dz)
    f = evaluate(shape.f, grid)
    chi = np.broadcast_to(chi, grid.shape)
    # In-slice |f| >= true 3D distance, so this port-side test is a superset of the needed band
    mask = (f > -2*port_band_cells*max(h, hz)) & (chi < 3*h)
    phi = signed_distance_on_grid(grid, V, F, f, mask=mask, far='sign')
    return phi


def cylindrical_grain(port, OD, L, n_xy=96, n_z=48, margin=4):
    """Grain of outer diameter OD and length L with an inert case; burning port(s) ``port`` (Shape)."""
    grid = Grid3D.cylinder(OD, L, n_xy, n_z, margin)
    X, Y = grid.xy_mesh()
    chi = (np.hypot(X, Y) - OD/2)[:, :, None]
    return Grain3D(grid, port_sdf(port, grid, chi), chi)
