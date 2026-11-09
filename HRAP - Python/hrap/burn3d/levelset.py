"""Level-set evolution phi_t + F |grad phi| = 0 with F >= 0 (front moves toward +phi, i.e. into fuel).

Spatial: 5th-order HJ-WENO (Jiang & Peng 2000) one-sided derivatives with the Godunov
Hamiltonian for F >= 0 (Osher & Fedkiw, Level Set Methods, ch. 3, 6).
Temporal: 3rd-order TVD Runge-Kutta (Shu & Osher 1988).
"""
import numpy as np

try:
    import skfmm
except ImportError:  # pragma: no cover
    skfmm = None


def _pad(phi, axis, bc):
    pw = [(0, 0)] * phi.ndim
    pw[axis] = (3, 3)
    if bc == 'mirror':
        # Even reflection about the boundary node: d(phi)/dn = 0 (inhibited end face)
        return np.pad(phi, pw, mode='reflect')
    # Odd reflection about the boundary node == linear extrapolation (keeps |grad phi| ~ 1)
    return np.pad(phi, pw, mode='reflect', reflect_type='odd')


def _weno5(v1, v2, v3, v4, v5):
    p1 = v1/3 - 7*v2/6 + 11*v3/6
    p2 = -v2/6 + 5*v3/6 + v4/3
    p3 = v3/3 + 5*v4/6 - v5/6
    s1 = 13/12*(v1 - 2*v2 + v3)**2 + 0.25*(v1 - 4*v2 + 3*v3)**2
    s2 = 13/12*(v2 - 2*v3 + v4)**2 + 0.25*(v2 - v4)**2
    s3 = 13/12*(v3 - 2*v4 + v5)**2 + 0.25*(3*v3 - 4*v4 + v5)**2
    eps = 1e-6*np.maximum.reduce([v1*v1, v2*v2, v3*v3, v4*v4, v5*v5]) + 1e-99
    a1 = 0.1/(s1 + eps)**2
    a2 = 0.6/(s2 + eps)**2
    a3 = 0.3/(s3 + eps)**2
    return (a1*p1 + a2*p2 + a3*p3) / (a1 + a2 + a3)


def one_sided_derivs(phi, axis, h, bc='extrapolate', scheme='weno5'):
    """Return (D^- phi, D^+ phi) along an axis."""
    n = phi.shape[axis]
    d = np.diff(_pad(phi, axis, bc), axis=axis) / h   # d[j] = (p[j+1]-p[j])/h, length n+5
    idx = [slice(None)] * phi.ndim

    def take(k):
        idx[axis] = slice(k, k + n)
        return d[tuple(idx)]

    if scheme == 'weno5':
        dm = _weno5(take(0), take(1), take(2), take(3), take(4))
        dp = _weno5(take(5), take(4), take(3), take(2), take(1))
    elif scheme == 'eno1':
        dm, dp = take(2), take(3)
    else:
        raise ValueError(scheme)
    return dm, dp


def grad_norm_godunov(phi, h, bcs, scheme='weno5'):
    """Upwind |grad phi| for a front moving with F >= 0."""
    g2 = np.zeros_like(phi)
    for axis in range(phi.ndim):
        dm, dp = one_sided_derivs(phi, axis, h[axis], bcs[axis], scheme)
        g2 += np.maximum(np.maximum(dm, 0.0)**2, np.minimum(dp, 0.0)**2)
    return np.sqrt(g2)


def advance(phi, speed, T, h, bcs=('extrapolate', 'extrapolate', 'extrapolate'), cfl=0.5, scheme='weno5'):
    """Advance phi_t + speed |grad phi| = 0 over time T. ``speed`` broadcasts against phi, >= 0.

    Returns (phi, n_substeps).
    """
    speed = np.asarray(speed, dtype=float)
    if np.any(speed < 0):
        raise ValueError('speed must be non-negative')
    Fmax = float(np.max(speed)) if speed.size else 0.0
    if Fmax == 0.0 or T <= 0.0:
        return phi, 0
    n = int(np.ceil(T * Fmax * sum(1.0/hi for hi in h) / cfl))
    dt = T / n

    def L(p):
        return -speed * grad_norm_godunov(p, h, bcs, scheme)

    for _ in range(n):
        p1 = phi + dt*L(phi)
        p2 = 0.75*phi + 0.25*(p1 + dt*L(p1))
        phi = phi/3 + 2/3*(p2 + dt*L(p2))
    return phi, n


def hopf_lax_shift(phi, D, z, h):
    """Exact-solution update for phi_t + F |grad phi| = 0 over one grain step.

    Hopf-Lax: phi(x, t+dt) = min_{|y-x| <= D} phi(y, t), which equals phi(x, t) - D for a signed
    distance phi (fuel side) and uniform displacement D; this is exact at corners (arcs at convex port corners, sharp fuel wedges)
    where upwind schemes are only first order. For a displacement D(z) varying along the axis, the
    level surface through x is moved by D at its characteristic foot z - n_z*phi (n = grad phi/|grad phi|),
    which keeps |grad phi| = 1 + O(D'^2) instead of 1 - n_z D' for a plain shift.

    D: (nz,) displacement per station at nodes z.
    """
    if np.all(D == D[0]):
        return phi - D[0]
    gz = np.gradient(phi, h[2], axis=2)
    gx = np.gradient(phi, h[0], axis=0)
    gy = np.gradient(phi, h[1], axis=1)
    gn = np.sqrt(gx*gx + gy*gy + gz*gz)
    nz = np.where(gn > 0, gz/np.where(gn > 0, gn, 1.0), 0.0)
    zf = z[None, None, :] - nz*phi
    return phi - np.interp(zf.ravel(), z, D).reshape(phi.shape)


def reinitialize(phi, h, order=2):
    """Restore phi to a signed distance function with the same zero level set (fast marching)."""
    if skfmm is None:
        raise ImportError('scikit-fmm is required for reinitialization (pip install scikit-fmm)')
    if not (np.any(phi < 0) and np.any(phi > 0)):
        return phi
    return np.asarray(skfmm.distance(phi, dx=list(h), order=order), dtype=float)
