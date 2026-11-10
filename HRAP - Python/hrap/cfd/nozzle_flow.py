"""Axisymmetric compressible-flow solver for the aft chamber, converging-diverging nozzle and plume.

Physics: 2D axisymmetric Euler equations (inviscid, calorically perfect gas with the chamber's
gamma and R; an effective gamma can reproduce shifting-equilibrium exit pressure), solved to steady
state. Ambient air is represented with the same gas constants (pressure and temperature of the
atmosphere, density p/(R T)).

Numerics: cell-centred finite volume on a body-fitted structured grid (per-radian form: face area
L r_mid, cell volume A_xr r_c, radial-momentum source p A_xr, which balances a uniform pressure
exactly), HLLC Riemann solver with Einfeldt wave speeds, MUSCL reconstruction of primitive
variables with the van Albada limiter, SSP-RK2 with local time stepping (pseudo-transient
convergence; intermediate fields are not time-accurate). JAX-compiled.

Boundaries: subsonic stagnation inlet (P0, T0, axial flow) at the chamber end, slip walls (acoustic
wall pressure), symmetry axis (zero-area faces), ambient far field at the top, pressure/extrapolation
outflow at the right. The plume block extends ``plume_len`` exit radii downstream.
"""
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


# --------------------------------------------------------------------------------------- geometry
@dataclass
class NozzleGeometry:
    Rt: float                 # throat radius
    Re: float                 # exit radius
    Rc: float                 # chamber radius
    Lc: float                 # straight chamber length upstream of the convergent
    conv_angle: float         # convergent half-angle (rad)
    div_angle: float          # divergent half-angle (rad), conical
    rcu: float                # upstream throat curvature radius
    rcd: float                # downstream throat curvature radius

    @classmethod
    def conical(cls, throat_d, ER, chamber_d, conv_angle_deg=45.0, div_angle_deg=15.0, rcu_ratio=1.5,
                rcd_ratio=0.382, Lc_ratio=0.6):
        Rt = throat_d/2
        Rc = max(chamber_d/2, 1.05*Rt)
        return cls(Rt=Rt, Re=Rt*np.sqrt(ER), Rc=Rc, Lc=Lc_ratio*Rc, conv_angle=np.radians(conv_angle_deg),
                   div_angle=np.radians(div_angle_deg), rcu=rcu_ratio*Rt, rcd=rcd_ratio*Rt)

    def _pts(self):
        tc, ta = self.conv_angle, self.div_angle
        x1 = -self.rcu*np.sin(tc); R1 = self.Rt + self.rcu*(1 - np.cos(tc))
        xc = x1 - (self.Rc - R1)/np.tan(tc)
        x2 = self.rcd*np.sin(ta); R2 = self.Rt + self.rcd*(1 - np.cos(ta))
        xe = x2 + (self.Re - R2)/np.tan(ta) if self.Re > R2 else x2
        return x1, R1, xc, x2, R2, xe

    @property
    def x_start(self):
        return self._pts()[2] - self.Lc

    @property
    def x_exit(self):
        return self._pts()[5]

    def radius(self, x):
        x = np.asarray(x, float)
        x1, R1, xc, x2, R2, xe = self._pts()
        tc, ta = self.conv_angle, self.div_angle
        R = np.where(x <= xc, self.Rc, 0.0)
        R = np.where((x > xc) & (x <= x1), R1 + (x1 - x)*np.tan(tc), R)
        R = np.where((x > x1) & (x <= 0), self.Rt + self.rcu - np.sqrt(np.maximum(self.rcu**2 - x**2, 0)), R)
        R = np.where((x > 0) & (x <= x2), self.Rt + self.rcd - np.sqrt(np.maximum(self.rcd**2 - x**2, 0)), R)
        R = np.where(x > x2, R2 + (x - x2)*np.tan(ta), R)
        return np.minimum(R, np.where(x > x2, self.Re, np.inf))


@dataclass
class Grid:
    X: np.ndarray            # (NI+1, NJ+1) node x
    R: np.ndarray            # (NI+1, NJ+1) node r
    active: np.ndarray       # (NI, NJ) bool
    ni_noz: int              # nozzle columns (cells)
    nj_noz: int              # nozzle rows (cells)
    geom: NozzleGeometry
    i_throat: int            # i-face index nearest the throat
    # face/cell metrics (filled by metrics())
    m: dict = field(default_factory=dict)


def _distribute(x0, x1, n, weight):
    """n+1 nodes on [x0, x1] with density proportional to weight(x)."""
    xs = np.linspace(x0, x1, 4001)
    w = weight(xs)
    c = np.concatenate([[0], np.cumsum(0.5*(w[1:] + w[:-1])*np.diff(xs))])
    return np.interp(np.linspace(0, c[-1], n + 1), c, xs)


def make_grid(geom, n_noz=120, n_rad=36, n_plume=140, n_far=40, plume_len=6.0, far_radius=3.0):
    """Nozzle block (n_noz x n_rad cells) + plume block (n_plume x (n_rad + n_far))."""
    xs, xe = geom.x_start, geom.x_exit
    # Cells clustered where the cross-section is small (throat)
    xn = _distribute(xs, xe, n_noz, lambda x: (geom.Rc/geom.radius(x))**0.8)
    dx_last = xn[-1] - xn[-2]
    L = plume_len*geom.Re
    # geometric growth from the exit spacing
    q = 1.0
    for _ in range(200):
        s = dx_last*(q**np.arange(1, n_plume + 1)).sum()
        q *= (L/s)**(1.0/n_plume)
    xp = xe + np.cumsum(dx_last*q**np.arange(1, n_plume + 1))
    xall = np.concatenate([xn, xp])
    eta = np.linspace(0, 1, n_rad + 1)
    dr0 = geom.Re/n_rad
    Rfar = far_radius*geom.Re
    g = 1.0
    for _ in range(200):
        s = dr0*(g**np.arange(1, n_far + 1)).sum()
        g *= ((Rfar - geom.Re)/s)**(1.0/n_far)
    r_ext = geom.Re + np.cumsum(dr0*g**np.arange(1, n_far + 1))
    NI, NJ = n_noz + n_plume, n_rad + n_far
    X = np.repeat(xall[:, None], NJ + 1, axis=1)
    R = np.zeros_like(X)
    Rw = geom.radius(xn)
    for i in range(NI + 1):
        if i <= n_noz:
            R[i, :n_rad + 1] = eta*Rw[i]
            R[i, n_rad + 1:] = Rw[i] + (r_ext - geom.Re)
        else:
            R[i, :n_rad + 1] = eta*geom.Re
            R[i, n_rad + 1:] = r_ext
    active = np.zeros((NI, NJ), bool)
    active[:n_noz, :n_rad] = True
    active[n_noz:, :] = True
    it = int(np.argmin(np.abs(xn)))
    grid = Grid(X=X, R=R, active=active, ni_noz=n_noz, nj_noz=n_rad, geom=geom, i_throat=it)
    grid.m = metrics(grid)
    return grid


def metrics(grid):
    X, R = grid.X, grid.R
    # i-faces: node (i,j) -> (i,j+1), normal +i
    dx = X[:, 1:] - X[:, :-1]; dr = R[:, 1:] - R[:, :-1]
    L = np.hypot(dx, dr)
    nxi, nri = dr/L, -dx/L
    Ai = L*0.5*(R[:, 1:] + R[:, :-1])
    # j-faces: node (i,j) -> (i+1,j), normal +j
    dx = X[1:, :] - X[:-1, :]; dr = R[1:, :] - R[:-1, :]
    L = np.hypot(dx, dr)
    nxj, nrj = -dr/L, dx/L
    Aj = L*0.5*(R[1:, :] + R[:-1, :])
    # cells (CCW: (i,j),(i+1,j),(i+1,j+1),(i,j+1))
    px = np.stack([X[:-1, :-1], X[1:, :-1], X[1:, 1:], X[:-1, 1:]], -1)
    pr = np.stack([R[:-1, :-1], R[1:, :-1], R[1:, 1:], R[:-1, 1:]], -1)
    cr = px*np.roll(pr, -1, -1) - np.roll(px, -1, -1)*pr
    Axr = 0.5*cr.sum(-1)
    xc = ((px + np.roll(px, -1, -1))*cr).sum(-1)/(6*Axr)
    rc = ((pr + np.roll(pr, -1, -1))*cr).sum(-1)/(6*Axr)
    V = Axr*rc
    act = grid.active
    NI, NJ = act.shape
    # wall faces: active on exactly one side
    pad_i = np.zeros((NI + 2, NJ), bool); pad_i[1:-1] = act
    left, right = pad_i[:-1], pad_i[1:]
    wall_i = left ^ right
    wall_i[0] = False; wall_i[-1] = False             # domain ends handled by inlet/outlet BCs
    pad_j = np.zeros((NI, NJ + 2), bool); pad_j[:, 1:-1] = act
    low, up = pad_j[:, :-1], pad_j[:, 1:]
    wall_j = low ^ up
    wall_j[:, 0] = False                               # axis (zero area)
    top_far = np.zeros_like(wall_j); top_far[:, -1] = act[:, -1]
    wall_j[:, -1] = False
    return dict(nxi=nxi, nri=nri, Ai=Ai, nxj=nxj, nrj=nrj, Aj=Aj, Axr=Axr, V=V, xc=xc, rc=rc,
                wall_i=wall_i, wall_i_left_active=left & wall_i, wall_j=wall_j, wall_j_low_active=low & wall_j,
                top_far=top_far, inlet=act[0])


# ------------------------------------------------------------------------------------------ solver
@dataclass
class FlowConditions:
    P0: float      # chamber stagnation pressure
    T0: float      # chamber stagnation temperature
    gamma: float
    Rgas: float    # J/kg/K
    Pa: float      # ambient pressure
    Ta: float = 294.0


def _prim(U, g):
    rho = U[..., 0]
    u = U[..., 1]/rho; v = U[..., 2]/rho
    p = (g - 1.0)*(U[..., 3] - 0.5*rho*(u*u + v*v))
    return jnp.stack([rho, u, v, p], -1)


def _cons(W, g):
    rho, u, v, p = W[..., 0], W[..., 1], W[..., 2], W[..., 3]
    return jnp.stack([rho, rho*u, rho*v, p/(g - 1.0) + 0.5*rho*(u*u + v*v)], -1)


def hllc(WL, WR, nx, ny, g):
    """HLLC flux per unit area through a face with unit normal (nx, ny) (left -> right)."""
    rL, uL, vL, pL = WL[..., 0], WL[..., 1], WL[..., 2], WL[..., 3]
    rR, uR, vR, pR = WR[..., 0], WR[..., 1], WR[..., 2], WR[..., 3]
    unL = uL*nx + vL*ny; utL = -uL*ny + vL*nx
    unR = uR*nx + vR*ny; utR = -uR*ny + vR*nx
    EL = pL/(g - 1) + 0.5*rL*(uL*uL + vL*vL)
    ER = pR/(g - 1) + 0.5*rR*(uR*uR + vR*vR)
    cL = jnp.sqrt(g*pL/rL); cR = jnp.sqrt(g*pR/rR)
    sL, sR = jnp.sqrt(rL), jnp.sqrt(rR)
    un_t = (sL*unL + sR*unR)/(sL + sR)
    ut_t = (sL*utL + sR*utR)/(sL + sR)
    H_t = (sL*(EL + pL)/rL + sR*(ER + pR)/rR)/(sL + sR)
    c_t = jnp.sqrt(jnp.maximum((g - 1)*(H_t - 0.5*(un_t**2 + ut_t**2)), 1e-12))
    SL = jnp.minimum(unL - cL, un_t - c_t)
    SR = jnp.maximum(unR + cR, un_t + c_t)
    SM = (pR - pL + rL*unL*(SL - unL) - rR*unR*(SR - unR))/(rL*(SL - unL) - rR*(SR - unR))

    def flux(r, un, ut, p, E):
        return jnp.stack([r*un, r*un*un + p, r*un*ut, un*(E + p)], -1)

    def ustar(r, un, ut, p, E, S):
        f = r*(S - un)/(S - SM)
        return jnp.stack([f, f*SM, f*ut, f*(E/r + (SM - un)*(SM + p/(r*(S - un))))], -1)
    FL = flux(rL, unL, utL, pL, EL); FR = flux(rR, unR, utR, pR, ER)
    UL = jnp.stack([rL, rL*unL, rL*utL, EL], -1); UR = jnp.stack([rR, rR*unR, rR*utR, ER], -1)
    FsL = FL + SL[..., None]*(ustar(rL, unL, utL, pL, EL, SL) - UL)
    FsR = FR + SR[..., None]*(ustar(rR, unR, utR, pR, ER, SR) - UR)
    F = jnp.where((SL >= 0)[..., None], FL,
                  jnp.where((SM >= 0)[..., None], FsL, jnp.where((SR > 0)[..., None], FsR, FR)))
    Fn, Ft = F[..., 1], F[..., 2]
    return jnp.stack([F[..., 0], Fn*nx - Ft*ny, Fn*ny + Ft*nx, F[..., 3]], -1)


def _limited_slope(Wm, W0, Wp, ok):
    a, b = W0 - Wm, Wp - W0
    s = (a*b*(a + b))/(a*a + b*b + 1e-30)          # van Albada
    s = jnp.where(a*b > 0, s, 0.0)
    return jnp.where(ok[..., None], s, 0.0)


def make_solver(grid, fc, cfl=0.5, order=2):
    """Returns (U0, step(U, n) -> (U, residual)) and helpers."""
    m = {k: jnp.asarray(v) for k, v in grid.m.items()}
    act = jnp.asarray(grid.active)
    NI, NJ = grid.active.shape
    g, Rg = fc.gamma, fc.Rgas
    cp = g*Rg/(g - 1)
    rho_a = fc.Pa/(Rg*fc.Ta)
    W_amb = jnp.array([rho_a, 0.0, 0.0, fc.Pa])
    # slopes allowed where both index neighbours exist and are active
    okI = np.zeros((NI, NJ), bool); okI[1:-1] = grid.active[1:-1] & grid.active[:-2] & grid.active[2:]
    okJ = np.zeros((NI, NJ), bool); okJ[:, 1:-1] = grid.active[:, 1:-1] & grid.active[:, :-2] & grid.active[:, 2:]
    okI, okJ = jnp.asarray(okI & (order > 1)), jnp.asarray(okJ & (order > 1))

    def inlet_state(Wc):
        u = jnp.maximum(Wc[..., 1], 0.0)
        T = jnp.maximum(fc.T0 - u*u/(2*cp), 0.3*fc.T0)
        p = fc.P0*(T/fc.T0)**(g/(g - 1))
        return jnp.stack([p/(Rg*T), u, jnp.zeros_like(u), p], -1)

    def face_fluxes(U):
        W = _prim(U, g)
        W = jnp.where(act[..., None], W, W_amb)
        # MUSCL face states, i direction
        Wm = jnp.concatenate([W[:1], W[:-1]], 0); Wp = jnp.concatenate([W[1:], W[-1:]], 0)
        dI = _limited_slope(Wm, W, Wp, okI)
        Wm = jnp.concatenate([W[:, :1], W[:, :-1]], 1); Wp = jnp.concatenate([W[:, 1:], W[:, -1:]], 1)
        dJ = _limited_slope(Wm, W, Wp, okJ)
        # interior i-faces 1..NI-1
        WLi = W[:-1] + 0.5*dI[:-1]; WRi = W[1:] - 0.5*dI[1:]
        bad = (WLi[..., 0] <= 0) | (WLi[..., 3] <= 0) | (WRi[..., 0] <= 0) | (WRi[..., 3] <= 0)
        WLi = jnp.where(bad[..., None], W[:-1], WLi); WRi = jnp.where(bad[..., None], W[1:], WRi)
        Fi_in = hllc(WLi, WRi, m['nxi'][1:-1], m['nri'][1:-1], g)
        # inlet (face 0) and outlet (face NI)
        Wg = inlet_state(W[0])
        F0 = hllc(Wg, W[0], m['nxi'][0], m['nri'][0], g)
        Wo = W[-1]
        c = jnp.sqrt(g*Wo[..., 3]/Wo[..., 0])
        un = Wo[..., 1]*m['nxi'][-1] + Wo[..., 2]*m['nri'][-1]
        Wg_out = jnp.where((un < c)[..., None], Wo.at[..., 3].set(fc.Pa), Wo)
        Wg_out = jnp.where((un < 0)[..., None], W_amb, Wg_out)
        FN = hllc(Wo, Wg_out, m['nxi'][-1], m['nri'][-1], g)
        Fi = jnp.concatenate([F0[None], Fi_in, FN[None]], 0)
        # interior j-faces 1..NJ-1
        WLj = W[:, :-1] + 0.5*dJ[:, :-1]; WRj = W[:, 1:] - 0.5*dJ[:, 1:]
        bad = (WLj[..., 0] <= 0) | (WLj[..., 3] <= 0) | (WRj[..., 0] <= 0) | (WRj[..., 3] <= 0)
        WLj = jnp.where(bad[..., None], W[:, :-1], WLj); WRj = jnp.where(bad[..., None], W[:, 1:], WRj)
        Fj_in = hllc(WLj, WRj, m['nxj'][:, 1:-1], m['nrj'][:, 1:-1], g)
        Fj0 = jnp.zeros((NI, 1, 4))                                  # axis: zero area
        Wt = W[:, -1]
        vn = Wt[..., 1]*m['nxj'][:, -1] + Wt[..., 2]*m['nrj'][:, -1]
        Wg_top = jnp.where((vn > 0)[..., None], Wt.at[..., 3].set(fc.Pa), W_amb)
        FjN = hllc(Wt, Wg_top, m['nxj'][:, -1], m['nrj'][:, -1], g)[:, None]
        Fj = jnp.concatenate([Fj0, Fj_in, FjN], 1)
        # slip walls: acoustic wall pressure from the active side
        Wpad = jnp.concatenate([W_amb[None, None].repeat(NJ, 1), W, W_amb[None, None].repeat(NJ, 1)], 0)
        WlI, WrI = Wpad[:-1], Wpad[1:]
        Ws = jnp.where(m['wall_i_left_active'][..., None], WlI, WrI)
        sgn = jnp.where(m['wall_i_left_active'], 1.0, -1.0)
        un = sgn*(Ws[..., 1]*m['nxi'] + Ws[..., 2]*m['nri'])
        pw = jnp.maximum(Ws[..., 3] + Ws[..., 0]*jnp.sqrt(g*Ws[..., 3]/Ws[..., 0])*un, 0.05*Ws[..., 3])
        Fw = jnp.stack([jnp.zeros_like(pw), pw*m['nxi'], pw*m['nri'], jnp.zeros_like(pw)], -1)
        Fi = jnp.where(m['wall_i'][..., None], Fw, Fi)
        Wpad = jnp.concatenate([W_amb[None, None].repeat(NI, 0), W, W_amb[None, None].repeat(NI, 0)], 1)
        WlJ, WrJ = Wpad[:, :-1], Wpad[:, 1:]
        Ws = jnp.where(m['wall_j_low_active'][..., None], WlJ, WrJ)
        sgn = jnp.where(m['wall_j_low_active'], 1.0, -1.0)
        un = sgn*(Ws[..., 1]*m['nxj'] + Ws[..., 2]*m['nrj'])
        pw = jnp.maximum(Ws[..., 3] + Ws[..., 0]*jnp.sqrt(g*Ws[..., 3]/Ws[..., 0])*un, 0.05*Ws[..., 3])
        Fw = jnp.stack([jnp.zeros_like(pw), pw*m['nxj'], pw*m['nrj'], jnp.zeros_like(pw)], -1)
        Fj = jnp.where(m['wall_j'][..., None], Fw, Fj)
        Fi = Fi*m['Ai'][..., None]; Fj = Fj*m['Aj'][..., None]
        return W, Fi, Fj

    def residual(U):
        W, Fi, Fj = face_fluxes(U)
        dU = -(Fi[1:] - Fi[:-1]) - (Fj[:, 1:] - Fj[:, :-1])
        dU = dU.at[..., 2].add(W[..., 3]*m['Axr'])
        # local time step from the face spectral radii
        c = jnp.sqrt(g*W[..., 3]/W[..., 0])
        lam_i = (jnp.abs(W[..., 1]*0.5*(m['nxi'][1:] + m['nxi'][:-1]) + W[..., 2]*0.5*(m['nri'][1:] + m['nri'][:-1])) + c) \
            * 0.5*(m['Ai'][1:] + m['Ai'][:-1])
        lam_j = (jnp.abs(W[..., 1]*0.5*(m['nxj'][:, 1:] + m['nxj'][:, :-1]) + W[..., 2]*0.5*(m['nrj'][:, 1:] + m['nrj'][:, :-1])) + c) \
            * 0.5*(m['Aj'][:, 1:] + m['Aj'][:, :-1])
        dt = cfl*m['V']/(lam_i + lam_j + 1e-300)
        return jnp.where(act[..., None], dU, 0.0), dt

    def rk2(U):
        R1, dt = residual(U)
        U1 = U + (dt/m['V'])[..., None]*R1
        R2, _ = residual(U1)
        Un = 0.5*(U + U1 + (dt/m['V'])[..., None]*R2)
        res = jnp.sqrt(jnp.sum(jnp.where(act, (Un[..., 0] - U[..., 0])/U[..., 0], 0.0)**2)/jnp.sum(act))
        return Un, res

    @jax.jit
    def step(U, n):
        def body(k, c):
            U, _ = c
            return rk2(U)
        return jax.lax.fori_loop(0, n, body, (U, 0.0))

    fluxes = jax.jit(lambda U: face_fluxes(U)[1])
    return step, fluxes


def initial_state(grid, fc):
    """Quasi-1D isentropic nozzle flow (supersonic branch downstream of the throat) inside the
    nozzle, ambient at rest in the plume block."""
    from scipy.optimize import brentq
    g, Rg = fc.gamma, fc.Rgas
    geom = grid.geom
    xc = grid.m['xc']
    NI, NJ = grid.active.shape
    W = np.zeros((NI, NJ, 4))
    W[..., 0] = fc.Pa/(Rg*fc.Ta); W[..., 3] = fc.Pa

    def area_ratio(M):
        return (1/M)*((2/(g + 1))*(1 + 0.5*(g - 1)*M*M))**((g + 1)/(2*(g - 1)))
    for i in range(grid.ni_noz):
        x = xc[i, 0]
        eps = (geom.radius(x)/geom.Rt)**2
        if eps <= 1.0 + 1e-9:
            M = 1.0
        elif x < 0:
            M = brentq(lambda M: area_ratio(M) - eps, 1e-6, 1.0)
        else:
            M = brentq(lambda M: area_ratio(M) - eps, 1.0, 50.0)
        T = fc.T0/(1 + 0.5*(g - 1)*M*M)
        p = fc.P0*(T/fc.T0)**(g/(g - 1))
        W[i, :grid.nj_noz] = [p/(Rg*T), M*np.sqrt(g*Rg*T), 0.0, p]
    return np.asarray(_cons(jnp.asarray(W), g))


# ------------------------------------------------------------------------------------ run & post
@dataclass
class CFDResult:
    grid: Grid
    fc: FlowConditions
    W: np.ndarray            # (NI, NJ, 4) primitives (rho, u, v, p)
    residuals: np.ndarray
    iterations: int
    summary: dict


def fields(grid, fc, U):
    W = np.asarray(_prim(jnp.asarray(U), fc.gamma))
    rho, u, v, p = W[..., 0], W[..., 1], W[..., 2], W[..., 3]
    T = p/(rho*fc.Rgas)
    c = np.sqrt(fc.gamma*p/rho)
    out = {'Mach': np.hypot(u, v)/c, 'Pressure [bar]': p/1e5, 'Temperature [K]': T,
           'Density [kg/m3]': rho, 'Velocity [m/s]': np.hypot(u, v)}
    for k in out:
        out[k] = np.where(grid.active, out[k], np.nan)
    return W, out


def _face_integrals(grid, fc, Fi, i_face):
    """Mass flow and thrust through i-face column i_face (nozzle rows) from the solver's numerical
    fluxes (exactly conservative), 2 pi included. Thrust = axial momentum + (p - Pa) on the face."""
    m = grid.m
    j = slice(0, grid.nj_noz)
    mdot = 2*np.pi*float(np.sum(Fi[i_face, j, 0]))
    F = 2*np.pi*float(np.sum(Fi[i_face, j, 1] - fc.Pa*m['nxi'][i_face, j]*m['Ai'][i_face, j]))
    return mdot, F


def summarize(grid, fc, W, ER, Fi):
    g, Rg = fc.gamma, fc.Rgas
    geom = grid.geom
    At = np.pi*geom.Rt**2
    Gam = np.sqrt(g)*(2/(g + 1))**((g + 1)/(2*(g - 1)))
    mdot_1d = fc.P0*At*Gam/np.sqrt(Rg*fc.T0)
    mdot_in, _ = _face_integrals(grid, fc, Fi, 0)
    mdot_t, _ = _face_integrals(grid, fc, Fi, grid.i_throat)
    mdot_e, F = _face_integrals(grid, fc, Fi, grid.ni_noz)
    # 1D ideal
    from scipy.optimize import brentq

    def area_ratio(M):
        return (1/M)*((2/(g + 1))*(1 + 0.5*(g - 1)*M*M))**((g + 1)/(2*(g - 1)))
    Me1 = brentq(lambda M: area_ratio(M) - ER, 1.0, 50.0)
    pe1 = fc.P0*(1 + 0.5*(g - 1)*Me1**2)**(-g/(g - 1))
    Cf1 = np.sqrt(2*g*g/(g - 1)*(2/(g + 1))**((g + 1)/(g - 1))*(1 - (pe1/fc.P0)**((g - 1)/g))) + (pe1 - fc.Pa)/fc.P0*ER
    lam = 0.5*(1 + np.cos(geom.div_angle))
    Cf1_div = (Cf1 - (pe1 - fc.Pa)/fc.P0*ER)*lam + (pe1 - fc.Pa)/fc.P0*ER
    # exit plane averages (mass-weighted Mach, area-weighted pressure)
    e = grid.ni_noz - 1
    rows = slice(0, grid.nj_noz)
    We = W[e, rows]
    c = np.sqrt(g*We[:, 3]/We[:, 0])
    Me = np.hypot(We[:, 1], We[:, 2])/c
    A = grid.m['Ai'][grid.ni_noz, rows]
    wts = We[:, 0]*We[:, 1]*A
    Me_avg = float(np.sum(Me*wts)/np.sum(wts)) if np.sum(wts) > 0 else float('nan')
    pe_avg = float(np.sum(We[:, 3]*A)/np.sum(A))
    # centreline Mach: shock inside the nozzle if the Mach number drops by > 0.3 downstream of the throat
    Mc = np.hypot(W[:grid.ni_noz, 0, 1], W[:grid.ni_noz, 0, 2])/np.sqrt(g*W[:grid.ni_noz, 0, 3]/W[:grid.ni_noz, 0, 0])
    # normal shock in the divergent section: axis flow subsonic again after reaching M > 1.2
    # (weak embedded compressions of conical nozzles do not count)
    down = Mc[grid.i_throat:]
    sup = np.maximum.accumulate(down) > 1.2
    shock_in_nozzle = bool(np.any(sup & (down < 1.0)))
    # wall distribution (cells adjacent to the wall)
    jw = grid.nj_noz - 1
    Ww = W[:grid.ni_noz, jw]
    cw = np.sqrt(g*Ww[:, 3]/Ww[:, 0])
    return {
        'mdot_cfd_kg_s': float(mdot_e), 'mdot_1d_kg_s': float(mdot_1d),
        'Cd_2d': float(mdot_e/mdot_1d), 'mass_imbalance': float((mdot_e - mdot_in)/mdot_in),
        'mass_imbalance_throat': float((mdot_t - mdot_in)/mdot_in),
        'thrust_cfd_N': float(F), 'Cf_cfd': float(F/(fc.P0*At)), 'Cf_1d': float(Cf1),
        'Cf_1d_divergence': float(Cf1_div), 'lambda_divergence': float(lam),
        'Me_avg': Me_avg, 'Me_1d': float(Me1), 'pe_avg_bar': pe_avg/1e5, 'pe_1d_bar': pe1/1e5,
        'shock_in_nozzle': shock_in_nozzle,
        'wall_x': grid.m['xc'][:grid.ni_noz, jw], 'wall_M': np.hypot(Ww[:, 1], Ww[:, 2])/cw,
        'wall_p': Ww[:, 3], 'wall_T': Ww[:, 3]/(Ww[:, 0]*Rg),
    }


def run(grid, fc, ER, max_iter=12000, tol=1e-7, chunk=100, avg_iter=2000, callback=None, cfl=0.5):
    """Iterate until the nozzle block is steady (mean |d rho/rho| per iteration < tol), then continue
    ``avg_iter`` iterations averaging the (inviscid, unsteady) plume.

    callback(it, history, U, phase) every ``chunk`` iterations; phase is 'converging' or 'averaging';
    return True to stop. history rows: (iteration, nozzle residual, whole-domain residual).
    """
    step, fluxes = make_solver(grid, fc, cfl=cfl)
    U = jnp.asarray(initial_state(grid, fc))
    noz = np.zeros(grid.active.shape, bool); noz[:grid.ni_noz, :grid.nj_noz] = True
    hist, it = [], 0
    phase, n_avg, U_sum = 'converging', 0, None
    prev = np.asarray(U)
    stopped = False
    while it < max_iter + avg_iter:
        U, res = step(U, chunk)
        it += chunk
        Un = np.asarray(U)
        if not np.all(np.isfinite(Un[grid.active])):
            raise FloatingPointError(f'CFD diverged at iteration {it}')
        d = np.abs(Un[..., 0] - prev[..., 0])/Un[..., 0]/chunk
        prev = Un
        r_noz = float(np.mean(d[noz]))
        hist.append((it, r_noz, float(res)))
        if phase == 'averaging':
            U_sum = Un.copy() if U_sum is None else U_sum + Un
            n_avg += 1
        if callback is not None and callback(it, hist, U, phase):
            stopped = True
            break
        if phase == 'converging' and (r_noz < tol or it >= max_iter):
            phase = 'averaging'
        elif phase == 'averaging' and n_avg*chunk >= avg_iter:
            break
    U_final = np.asarray(U)
    Fi = np.asarray(fluxes(U))
    if U_sum is not None and n_avg > 0:
        Um = U_sum/n_avg
        U_final = np.where(noz[..., None], U_final, Um)   # nozzle: converged; plume: averaged
    W, _ = fields(grid, fc, U_final)
    summ = summarize(grid, fc, W, ER, Fi)
    summ['converged'] = bool(hist[-1][1] < tol*10) if hist else False
    summ['stopped_early'] = stopped
    return CFDResult(grid=grid, fc=fc, W=W, residuals=np.array(hist), iterations=it, summary=summ)


def effective_gamma(ER, pe_over_p0, g_guess=1.2):
    """Constant gamma whose isentropic expansion to area ratio ER gives exit pressure ratio pe/p0
    (reproduces a shifting-equilibrium exit pressure in the single-gamma CFD)."""
    from scipy.optimize import brentq

    def pe(g):
        def ar(M):
            return (1/M)*((2/(g + 1))*(1 + 0.5*(g - 1)*M*M))**((g + 1)/(2*(g - 1)))
        Me = brentq(lambda M: ar(M) - ER, 1.0, 60.0)
        return (1 + 0.5*(g - 1)*Me*Me)**(-g/(g - 1))
    try:
        return brentq(lambda g: pe(g) - pe_over_p0, 1.05, 1.67)
    except ValueError:
        return g_guess
