"""Quasi-1D axial flow model along the port: local mass flux and regression rate at each station.

With rdot = k G^n and G(z) = mdot(z)/A(z), fuel addition along the port obeys
    dmdot/dz = rho * P(z) * rdot = c(z) mdot^n,     c = rho k P / A^n,
where P(z) = dA/dd is the burning perimeter (per unit axial length, includes surface tilt).
This is separable, so exactly
    mdot(z)^(1-n) = mdot_ox^(1-n) + (1-n) * int_0^z c dz'      (n != 1)
    mdot(z)       = mdot_ox * exp(int_0^z c dz')               (n == 1)
The only discretization is the quadrature of c (trapezoidal, O(dz^2)).

All functions accept ``xp`` = numpy or jax.numpy.
"""
import numpy as np

A_MIN = 1e-12  # m^2, floor to keep flux finite for a closed slice


def station_rates(A, P, mdot_ox, z, k, n, rho, w, flux='total', xp=np):
    """Return (rdot, G, mdot_z, mdot_f) at stations.

    A, P : (nz,) open port area and burning perimeter dA/dd at each station
    mdot_ox : oxidizer mass flow entering at z[0]
    z : (nz,) station positions; w : (nz,) quadrature weights for axial integrals
    k, n : rdot = k G^n;  rho : fuel density
    mdot_f = rho * sum(w * P * rdot)  is the fuel mass generation rate (consistent with d/dt of the
    trapezoidal fuel volume), used for all mass bookkeeping.
    """
    A = xp.maximum(A, A_MIN)
    mdot_ox = xp.maximum(mdot_ox, 0.0)
    if flux == 'oxidizer':
        mdot_z = mdot_ox * xp.ones_like(A)
    elif flux == 'total':
        c = rho * k * P / A**n
        seg = 0.5*(c[1:] + c[:-1]) * xp.diff(z)
        C = xp.concatenate([xp.zeros(1, dtype=seg.dtype), xp.cumsum(seg)])
        if n == 1.0:
            mdot_z = mdot_ox * xp.exp(C)
        else:
            base = mdot_ox**(1.0 - n) + (1.0 - n)*C
            # n > 1 can blow up (finite-length singularity); guard against nonpositive base
            mdot_z = xp.maximum(base, 0.0)**(1.0/(1.0 - n))
        # dm/dz = c m^n with m(0)=0 is non-unique for n<1; no oxidizer -> no combustion
        mdot_z = xp.where(mdot_ox > 0.0, mdot_z, 0.0)
    else:
        raise ValueError(flux)
    G = mdot_z / A
    rdot = xp.where(P > 0.0, k * G**n, 0.0)
    mdot_f = rho * xp.sum(w * P * rdot)
    return rdot, G, mdot_z, mdot_f
