"""Station evaluation shared by the numpy drivers and the JAX engine part (pass xp=jax.numpy).

State per axial station: in-plane regression depth d(z) relative to the tabulated offset geometry.
Tables: A[z, k] port area at offset levels k*dd, M[z, k] = dA/dd (monotone Hermite slopes),
optional kappa[z, k] >= 1 = (tilted burning perimeter)/(in-plane perimeter) for the helical
2D approximation (1 for 3D tables, where tilt is already in A).

With kappa: P_eff = kappa P is the true burning area per unit length; fuel generation is
rho * rdot * P_eff; the section advances in-plane at dd/dt = kappa * rdot so that
d(A)/dt = P * kappa * rdot = P_eff * rdot (mass-consistent).
"""
import numpy as np

from burnfront.burn3d.measure import hermite_eval
from burnfront.burn3d.flow import station_rates


def interp_levels(T, dd, d, xp=np):
    """Linear interpolation of per-station level tables T (nz, K) at offsets d (nz,)."""
    K = T.shape[-1]
    s = xp.clip(d / dd, 0.0, K - 1.0)
    i = xp.clip(xp.floor(s).astype(int), 0, K - 2)
    t = s - i
    rows = xp.arange(T.shape[0])
    return (1 - t)*T[rows, i] + t*T[rows, i + 1]


def station_eval(A_tab, M_tab, dd, d, mdot_ox, z, w, k, n, rho, flux, kappa=None, xp=np):
    """Return dict with ddot, rdot, G, mdot_z, mdot_f, A, P_eff at the stations."""
    A, P = hermite_eval(A_tab, M_tab, dd, d, xp)
    if kappa is None:
        kap = 1.0
        P_eff = P
    else:
        kap = interp_levels(kappa, dd, d, xp)
        P_eff = kap * P
    rdot, G, mz, mf = station_rates(A, P_eff, mdot_ox, z, k, n, rho, w, flux, xp)
    return dict(ddot=kap*rdot, rdot=rdot, G=G, mdot_z=mz, mdot_f=mf, A=A, P=P_eff)
