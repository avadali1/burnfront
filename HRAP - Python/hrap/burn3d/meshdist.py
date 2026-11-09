"""Exact distance from points to a triangle mesh, and signed distance fields on a Grid3D.

Candidate triangles come from a KD-tree over per-triangle sample points. With covering radius s
(every point of every triangle lies within s of one of that triangle's samples), the triangle that
holds the true closest point has a sample within d* + s of the query. A query is therefore accepted
only when its K-th nearest sample is at least d_found + s away; otherwise K is increased. The
result is exact (up to floating point), not a sampling approximation.
"""
import numpy as np
from scipy.spatial import cKDTree


def closest_point_triangles(p, a, b, c):
    """Squared distance from points p to triangles (a, b, c), all broadcastable (..., 3).

    Ericson, Real-Time Collision Detection, 5.1.5 (Voronoi-region classification).
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum('...i,...i', ab, ap)
    d2 = np.einsum('...i,...i', ac, ap)
    bp = p - b
    d3 = np.einsum('...i,...i', ab, bp)
    d4 = np.einsum('...i,...i', ac, bp)
    cp = p - c
    d5 = np.einsum('...i,...i', ab, cp)
    d6 = np.einsum('...i,...i', ac, cp)
    va = d3*d6 - d5*d4
    vb = d5*d2 - d1*d6
    vc = d1*d4 - d3*d2

    def safe(num, den):
        return num / np.where(den != 0.0, den, 1.0)

    # Interior (default), then override by edge/vertex regions in reverse priority
    den = va + vb + vc
    v = safe(vb, den); w = safe(vc, den)
    q = a + ab*v[..., None] + ac*w[..., None]
    # Edge BC
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = safe(d4 - d3, (d4 - d3) + (d5 - d6))
    q = np.where(m[..., None], b + (c - b)*t[..., None], q)
    # Edge AC
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = safe(d2, d2 - d6)
    q = np.where(m[..., None], a + ac*t[..., None], q)
    # Edge AB
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = safe(d1, d1 - d3)
    q = np.where(m[..., None], a + ab*t[..., None], q)
    # Vertices
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, q)
    diff = p - q
    return np.einsum('...i,...i', diff, diff)


class MeshDistance:
    def __init__(self, vertices, faces):
        V = np.asarray(vertices, dtype=float)
        F = np.asarray(faces, dtype=np.int64)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        self.a, self.b, self.c = a, b, c
        # One sample per triangle (its centroid). The farthest point of a triangle from its centroid
        # is a vertex, so s = max |vertex - centroid| is a rigorous covering radius.
        cen = (a + b + c)/3
        self.samples = cen
        self.sample_tri = np.arange(F.shape[0])
        rad = np.max(np.stack([np.linalg.norm(a - cen, axis=1), np.linalg.norm(b - cen, axis=1),
                               np.linalg.norm(c - cen, axis=1)]), axis=0)
        self.s = float(rad.max())
        self.tree = cKDTree(self.samples)
        self.abc = np.ascontiguousarray(np.concatenate([a, b, c], axis=1))  # (ntri, 9) single gather

    def _exact_over(self, points, k, radius=None, pair_budget=3_000_000):
        """Min distance over each point's k nearest candidate triangles (optionally only those whose
        centroid is within ``radius``)."""
        n = points.shape[0]
        out = np.empty(n)
        chunk = max(1, pair_budget // k)
        for s0 in range(0, n, chunk):
            p = points[s0:s0 + chunk]
            dk, ik = self.tree.query(p, k=k)
            if k == 1:
                dk, ik = dk[:, None], ik[:, None]
            T = self.abc[self.sample_tri[ik]]
            d2 = closest_point_triangles(p[:, None, :], T[..., 0:3], T[..., 3:6], T[..., 6:9])
            if radius is not None:
                d2 = np.where(dk <= radius[s0:s0 + chunk, None], d2, np.inf)
            out[s0:s0 + chunk] = np.sqrt(d2.min(axis=1))
        return out

    def unsigned(self, points, k1=8):
        """Exact unsigned distance from each point to the mesh.

        Stage 1: exact distance over the k1 nearest centroids -> upper bound d_up.
        Stage 2: any triangle closer than d_up has its centroid within d_up + s, so querying exactly
        that many neighbours (counted in C by the KD-tree) and taking the minimum is exact.
        """
        points = np.asarray(points, dtype=float)
        ns = self.samples.shape[0]
        k1 = min(k1, ns)
        d_up = self._exact_over(points, k1)
        r = d_up + self.s
        cnt = np.asarray(self.tree.query_ball_point(points, r=r, return_length=True))
        out = d_up.copy()
        need = cnt > k1
        K = k1
        while np.any(need):
            K = min(2*K, ns)
            sel = need & ((cnt <= K) | (K == ns))
            if np.any(sel):
                idx = np.nonzero(sel)[0]
                out[idx] = np.minimum(out[idx], self._exact_over(points[idx], K, radius=r[idx]))
                need &= ~sel
        return out


def signed_distance_on_grid(grid, vertices, faces, sign, mask=None, far='sign'):
    """phi = sign * exact distance to the mesh at grid nodes where ``mask`` is True (default: all).

    sign: (nx, ny, nz) array whose sign is negative inside the port.
    far: value used outside ``mask``: 'sign' keeps the input array (only its sign is meaningful),
         'fmm' uses fast-marching distances.
    """
    md = MeshDistance(vertices, faces)
    shape = grid.shape
    sign = np.asarray(sign, dtype=float)
    sgn = np.where(sign < 0, -1.0, 1.0)
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    ii, jj, kk = np.nonzero(mask)
    pts = np.stack([grid.x[ii], grid.y[jj], grid.z[kk]], axis=1)
    d = md.unsigned(pts)
    if mask.all():
        phi = np.empty(shape)
    elif far == 'fmm':
        from hrap.burn3d.levelset import reinitialize
        phi = reinitialize(sgn, grid.h)
    else:
        phi = sign.copy()
    phi[ii, jj, kk] = sgn[ii, jj, kk] * d
    return phi
