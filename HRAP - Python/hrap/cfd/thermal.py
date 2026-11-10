"""Nozzle wall heat transfer: Bartz convective correlation and transient radial wall conduction.

Gas side (Bartz 1957, as given in Huzel & Huang):
    h_g = 0.026/Dt^0.2 (mu^0.2 cp/Pr^0.6)_0 (P0/c*)^0.8 (Dt/r_c)^0.1 (At/A)^0.9 sigma
    sigma = [0.5 (Tw/T0)(1 + (g-1)/2 M^2) + 0.5]^-0.68 [1 + (g-1)/2 M^2]^-0.12
    mu = 1.184e-7 M_w^0.5 T^0.6 [Pa s]  (the 46.6e-10 M^0.5 T^0.6 lb/(in s), T in degR, form in SI)
    Pr = 4g/(9g - 5),  q = h_g (T_aw - Tw),
    T_aw = T0 (1 + r (g-1)/2 M^2)/(1 + (g-1)/2 M^2),  recovery factor r = Pr^(1/3)
The local Mach number along the wall is the isentropic one for A(x)/At (subsonic upstream of the
throat, supersonic downstream) unless a CFD wall distribution is supplied.

Wall: 1D radial conduction in a cylindrical shell [R(x), R(x) + thickness] at every axial station,
finite volume with Crank-Nicolson time stepping (backward-Euler start-up steps); heated inner surface, outer surface convecting to
ambient (h_out, T_amb). Constant material properties; no ablation; no axial conduction.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

# k [W/m/K], rho [kg/m3], cp [J/kg/K], service temperature limit [K] (melting / decomposition)
MATERIALS = {
    'Graphite': dict(k=80.0, rho=1850.0, cp=1500.0, T_max=3300.0),
    'Copper': dict(k=390.0, rho=8900.0, cp=385.0, T_max=1356.0),
    'Steel (4130)': dict(k=42.0, rho=7850.0, cp=500.0, T_max=1700.0),
    'Aluminium (6061)': dict(k=167.0, rho=2700.0, cp=900.0, T_max=855.0),
    'Silica phenolic': dict(k=0.6, rho=1750.0, cp=1150.0, T_max=600.0),
}


def viscosity(M_w, T):
    return 1.184e-7*np.sqrt(M_w)*np.asarray(T)**0.6


def isentropic_mach(eps, g, supersonic):
    from scipy.optimize import brentq

    def ar(M):
        return (1/M)*((2/(g + 1))*(1 + 0.5*(g - 1)*M*M))**((g + 1)/(2*(g - 1)))
    eps = np.atleast_1d(eps)
    sup = np.broadcast_to(supersonic, eps.shape)
    out = np.empty(eps.size)
    for k, (e, s) in enumerate(zip(eps.ravel(), sup.ravel())):
        if e <= 1.0 + 1e-12:
            out[k] = 1.0
        elif s:
            out[k] = brentq(lambda M: ar(M) - e, 1.0, 80.0)
        else:
            out[k] = brentq(lambda M: ar(M) - e, 1e-8, 1.0)
    return out.reshape(eps.shape)


def bartz(P0, T0, g, Rgas, cstar, Dt, rc, A_ratio, M, Tw):
    """Gas-side heat transfer coefficient h_g [W/m2/K] and adiabatic wall temperature [K]."""
    M_w = 8314.46/Rgas
    cp = g*Rgas/(g - 1)
    Pr = 4*g/(9*g - 5)
    mu = viscosity(M_w, T0)
    f = 1 + 0.5*(g - 1)*M**2
    sigma = (0.5*(Tw/T0)*f + 0.5)**-0.68*f**-0.12
    h = 0.026/Dt**0.2*(mu**0.2*cp/Pr**0.6)*(P0/cstar)**0.8*(Dt/rc)**0.1*(1.0/A_ratio)**0.9*sigma
    Taw = T0*(1 + Pr**(1/3)*0.5*(g - 1)*M**2)/f
    return h, Taw


@dataclass
class WallResult:
    x: np.ndarray          # (nx,) axial stations (throat at 0)
    t: np.ndarray          # (nt,) times
    Tw: np.ndarray         # (nt, nx) inner wall temperature
    q: np.ndarray          # (nt, nx) heat flux into the wall [W/m2]
    h: np.ndarray          # (nt, nx) gas-side coefficient
    T_through: np.ndarray  # (nx, n_r) temperature through the wall at the final time
    r_frac: np.ndarray     # (n_r,) radial position through the wall, 0 = gas side
    material: str
    T_limit: float

    @property
    def peak(self):
        i = np.unravel_index(np.argmax(self.Tw), self.Tw.shape)
        return dict(T=float(self.Tw[i]), t=float(self.t[i[0]]), x=float(self.x[i[1]]))

    def time_to_limit(self):
        hit = np.where(self.Tw.max(axis=1) >= self.T_limit)[0]
        return float(self.t[hit[0]]) if hit.size else float('nan')


class ShellConduction:
    """Crank-Nicolson finite-volume conduction through cylindrical shells (one per axial station)."""

    def __init__(self, R_in, thickness, k, rho, cp, n_r=30, T_init=294.0):
        self.n = n_r
        s = np.linspace(0.0, 1.0, n_r)
        self.r = R_in[:, None] + thickness*s[None, :]
        dr = thickness/(n_r - 1)
        rf = 0.5*(self.r[:, 1:] + self.r[:, :-1])
        vol = np.zeros_like(self.r)
        vol[:, 1:-1] = 0.5*(rf[:, 1:]**2 - rf[:, :-1]**2)
        vol[:, 0] = 0.5*(rf[:, 0]**2 - self.r[:, 0]**2)
        vol[:, -1] = 0.5*(self.r[:, -1]**2 - rf[:, -1]**2)
        self.cond = k*rf/dr               # conductance between nodes (per radian, per unit length)
        self.C = rho*cp*vol
        self.T = np.full(self.r.shape, float(T_init))
        self.n_steps = 0

    def step(self, dt, h_in, T_in, h_out, T_out, theta=None):
        """Advance by dt with convective inner (h_in, T_in per station) and outer boundaries.
        theta-scheme: Crank-Nicolson (0.5) after 4 backward-Euler start steps (Rannacher start-up:
        CN alone is not L-stable and rings after a sudden surface heating)."""
        n = self.n
        if theta is None:
            theta = 1.0 if self.n_steps < 4 else 0.5
        self.n_steps += 1
        for i in range(self.r.shape[0]):
            kk = self.cond[i]
            diag = np.zeros(n)
            diag[:-1] += kk; diag[1:] += kk
            diag[0] += h_in[i]*self.r[i, 0]
            diag[-1] += h_out*self.r[i, -1]
            src = np.zeros(n)
            src[0] = h_in[i]*self.r[i, 0]*T_in[i]
            src[-1] = h_out*self.r[i, -1]*T_out
            Ti = self.T[i]
            LT = diag*Ti
            LT[:-1] -= kk*Ti[1:]
            LT[1:] -= kk*Ti[:-1]
            rhs = self.C[i]/dt*Ti - (1 - theta)*LT + src
            ab = np.zeros((3, n))
            ab[0, 1:] = -theta*kk
            ab[1] = self.C[i]/dt + theta*diag
            ab[2, :-1] = -theta*kk
            self.T[i] = solve_banded((1, 1), ab, rhs)


def wall_transient(geom, t, P0, T0, gamma, Rgas, cstar, material='Graphite', thickness=0.01, n_x=60,
                   n_r=30, T_init=294.0, h_out=10.0, T_amb=294.0, wall_mach=None, dt_max=0.01):
    """Inner-wall temperature history along the nozzle for chamber histories P0(t), T0(t), gamma(t),
    Rgas(t), cstar(t) (arrays on t). geom: hrap.cfd.nozzle_flow.NozzleGeometry.
    wall_mach: optional (x, M) wall Mach distribution (e.g. from the CFD) replacing isentropic M(x)."""
    mat = MATERIALS[material]
    x = np.linspace(geom.x_start, geom.x_exit, n_x)
    Rw = geom.radius(x)
    eps = (Rw/geom.Rt)**2
    rc = 0.5*(geom.rcu + geom.rcd)
    Dt = 2*geom.Rt
    wall = ShellConduction(Rw, thickness, mat['k'], mat['rho'], mat['cp'], n_r, T_init)
    t = np.asarray(t, float)
    out_T, out_q, out_h = [np.zeros((t.size, n_x)) for _ in range(3)]
    out_T[0] = wall.T[:, 0]
    M_cache = {}

    def at(arr, tt):
        return float(np.interp(tt, t, np.asarray(arr, float)))
    for n in range(1, t.size):
        t0, t1 = t[n - 1], t[n]
        nsub = max(1, int(np.ceil((t1 - t0)/dt_max)))
        dt = (t1 - t0)/nsub
        for m_ in range(nsub):
            ts = t0 + (m_ + 0.5)*dt
            g, Rg, P, Tc, cs = at(gamma, ts), at(Rgas, ts), at(P0, ts), at(T0, ts), at(cstar, ts)
            if wall_mach is not None:
                M = np.interp(x, wall_mach[0], wall_mach[1])
            else:
                key = round(g, 3)
                if key not in M_cache:
                    M_cache[key] = isentropic_mach(eps, g, x > 0)
                M = M_cache[key]
            if P > 0 and Tc > 0 and cs > 0:
                h, Taw = bartz(P, Tc, g, Rg, cs, Dt, rc, eps, M, wall.T[:, 0])
            else:
                h, Taw = np.zeros(n_x), wall.T[:, 0].copy()
            wall.step(dt, h, Taw, h_out, T_amb)
        out_T[n] = wall.T[:, 0]
        out_h[n] = h
        out_q[n] = h*(Taw - wall.T[:, 0])
    return WallResult(x=x, t=t, Tw=out_T, q=out_q, h=out_h, T_through=wall.T.copy(),
                      r_frac=np.linspace(0, 1, n_r), material=material, T_limit=mat['T_max'])
