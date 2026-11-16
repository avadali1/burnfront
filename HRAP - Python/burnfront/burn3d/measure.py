"""Sub-cell accurate geometric measures on slices of a level-set field.

The area of {psi < 0} in each cell is computed exactly for the piecewise-linear interpolant of psi
on triangles (both diagonal splits averaged for symmetry), giving O(h^2) accuracy for smooth fronts.
"""
import numpy as np


def tri_fraction_neg(a, b, c):
    """Exact fraction of a triangle where the linear interpolant of vertex values (a,b,c) is < 0."""
    v = np.sort(np.stack([a, b, c], axis=-1), axis=-1)
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
    f = np.where(v3 < 0.0, 1.0, 0.0)
    one = (v1 < 0.0) & (v2 >= 0.0)
    two = (v2 < 0.0) & (v3 >= 0.0)
    d1 = np.where(one, (v1 - v2)*(v1 - v3), 1.0)
    d2 = np.where(two, (v3 - v1)*(v3 - v2), 1.0)
    f = np.where(one, v1*v1/d1, f)
    f = np.where(two, 1.0 - v3*v3/d2, f)
    return f


def quad_fraction_neg(p00, p10, p01, p11):
    """Fraction of a rectangle cell (corner values) where psi < 0; average of both triangulations."""
    fa = tri_fraction_neg(p00, p10, p11) + tri_fraction_neg(p00, p11, p01)
    fb = tri_fraction_neg(p00, p10, p01) + tri_fraction_neg(p10, p11, p01)
    return 0.25*(fa + fb)


def _corners(a):
    return a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:]


def slice_areas(psi, hx, hy):
    """Area of {psi < 0} in every z-slice of psi (nx, ny, nz) -> (nz,)."""
    f = quad_fraction_neg(*_corners(psi))
    return f.sum(axis=(0, 1)) * hx * hy


def port_area_table(phi, chi, levels, hx, hy):
    """A[z, k] = area of {max(phi - levels[k], chi) < 0} in each slice: open port area after a
    normal offset of levels[k] (exact offset geometry when phi is a signed distance field).

    Only cells that can be cut for some level are evaluated (narrow band); others are summed in bulk.
    """
    levels = np.asarray(levels, dtype=float)
    nz = phi.shape[2]
    pc = _corners(phi)
    cc = _corners(np.broadcast_to(chi, phi.shape))
    pmax, pmin = np.maximum.reduce(pc), np.minimum.reduce(pc)
    cmax, cmin = np.maximum.reduce(cc), np.minimum.reduce(cc)

    full = (pmax < levels.min()) & (cmax < 0.0)         # inside port for every level
    empty = (pmin >= levels.max()) | (cmin >= 0.0)      # never inside
    band = ~(full | empty)
    zi = np.nonzero(band)[2]
    pb = [c[band] for c in pc]
    cb = [c[band] for c in cc]
    pmax_b, pmin_b = pmax[band], pmin[band]
    cmax_b, cmin_b = cmax[band], cmin[band]
    case_cut = (cmax_b >= 0.0) & (cmin_b < 0.0)   # cells cut by the envelope

    A = np.empty((nz, levels.size))
    A_full = full.sum(axis=(0, 1)).astype(float)
    for k, d in enumerate(levels):
        # Per level, only cells actually cut by psi = max(phi - d, chi) need the exact fraction
        inside = (pmax_b < d) & (cmax_b < 0.0)
        cut = ~inside & (pmin_b < d) & ((pmax_b >= d) | case_cut)
        frac = quad_fraction_neg(*[np.maximum(p[cut] - d, c[cut]) for p, c in zip(pb, cb)])
        A[:, k] = (A_full + np.bincount(zi[inside], minlength=nz)
                   + np.bincount(zi[cut], weights=frac, minlength=nz))
    return A * hx * hy


def envelope_areas(chi, nz, hx, hy):
    """Area inside the inert envelope per slice (same discretization as port_area_table)."""
    chi = np.broadcast_to(chi, chi.shape[:2] + (nz,)) if chi.shape[2] == 1 else chi
    return slice_areas(chi, hx, hy)


# --- Monotone cubic Hermite interpolation of A(d) along the level axis ------------------------------

def pchip_slopes(A, dd):
    """Fritsch-Carlson monotone slopes for uniformly spaced samples along the last axis."""
    delta = np.diff(A, axis=-1) / dd
    m = np.zeros_like(A)
    d0, d1 = delta[..., :-1], delta[..., 1:]
    same = (d0*d1) > 0.0
    m[..., 1:-1] = np.where(same, 2.0/(1.0/np.where(same, d0, 1.0) + 1.0/np.where(same, d1, 1.0)), 0.0)

    def end(dA, dB):
        e = (3.0*dA - dB)/2.0
        e = np.where(np.sign(e) != np.sign(dA), 0.0, e)
        e = np.where((np.sign(dA) != np.sign(dB)) & (np.abs(e) > 3.0*np.abs(dA)), 3.0*dA, e)
        return e

    if A.shape[-1] > 2:
        m[..., 0] = end(delta[..., 0], delta[..., 1])
        m[..., -1] = end(delta[..., -1], delta[..., -2])
    else:
        m[..., 0] = m[..., -1] = delta[..., 0]
    return m


def hermite_eval(A, M, dd, d, xp=np):
    """Evaluate the cubic Hermite interpolant (values A, slopes M, spacing dd, levels from 0) at
    per-row offsets d. Returns (value, derivative). d outside [0, (K-1) dd] is clamped.

    A, M: (nz, K); d: (nz,). Written for numpy or jax.numpy (xp).
    """
    K = A.shape[-1]
    s = xp.clip(d / dd, 0.0, K - 1.0)
    i = xp.clip(xp.floor(s).astype(int), 0, K - 2)
    t = s - i
    rows = xp.arange(A.shape[0])
    a0, a1 = A[rows, i], A[rows, i + 1]
    m0, m1 = M[rows, i]*dd, M[rows, i + 1]*dd
    t2, t3 = t*t, t*t*t
    val = (2*t3 - 3*t2 + 1)*a0 + (t3 - 2*t2 + t)*m0 + (-2*t3 + 3*t2)*a1 + (t3 - t2)*m1
    der = ((6*t2 - 6*t)*a0 + (3*t2 - 4*t + 1)*m0 + (-6*t2 + 6*t)*a1 + (3*t2 - 2*t)*m1) / dd
    return val, der


def trapz_weights(z):
    """Trapezoidal quadrature weights for nodes z."""
    z = np.asarray(z, dtype=float)
    w = np.zeros_like(z)
    dz = np.diff(z)
    w[:-1] += dz/2
    w[1:] += dz/2
    return w
