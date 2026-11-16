"""Standalone 3D grain burn with a prescribed oxidizer mass flow (numpy/scipy only).

Algorithm (per grain step)
--------------------------
1. Tabulate A(z, d): open port area of every slice after a normal offset d in [0, d_max] (exact
   offset geometry because phi is a signed distance field), fitted with monotone cubic Hermite
   splines in d; P(z, d) = dA/dd is the burning perimeter (includes surface tilt).
2. Integrate the station ODE  dd(z)/dt = rdot(z) = k G(z)^n  with RK4 (G from flow.station_rates),
   together with the fuel mass ledger dm/dt = mdot_f = rho * sum_z w P rdot.
   Because P is the exact derivative of the interpolated A, the ledger equals rho * delta(volume)
   of the tabulated geometry up to the ODE truncation error.
3. Move the level set by the accumulated displacement d(z) along the local normal
   (phi_t + F|grad phi| = 0 with WENO5/TVD-RK3, F = d(z) over unit pseudo-time), then re-tabulate.
   The difference between the re-measured and predicted port volume is the level-set defect,
   reported for verification.
"""
from dataclasses import dataclass, field

import numpy as np

from burnfront.burn3d import measure, flow, levelset


@dataclass
class BurnResult:
    z: np.ndarray
    t: np.ndarray
    mdot_ox: np.ndarray
    mdot_f: np.ndarray
    m_burned: np.ndarray          # fuel mass consumed (ODE ledger), kg
    V_port: np.ndarray            # open port volume, m^3
    A_burn: np.ndarray            # burning surface area, m^2
    A_port_z: np.ndarray          # (Nt, nz) open port area per station
    P_z: np.ndarray               # (Nt, nz) burning perimeter per station
    G_z: np.ndarray               # (Nt, nz) local mass flux
    rdot_z: np.ndarray            # (Nt, nz) local regression rate
    mdot_z: np.ndarray            # (Nt, nz) axial mass flow
    depth_z: np.ndarray           # (Nt, nz) cumulative regression depth per station
    m_fuel0: float
    t_geo: np.ndarray             # times of level-set re-measurement
    m_burned_geo: np.ndarray      # fuel consumed from the re-measured level-set volume, kg
    t_burnthrough: float = np.nan # first time the front reaches the inert envelope
    m_sliver_burnthrough: float = np.nan
    snapshots: list = field(default_factory=list)  # [(t, phi float32)]
    n_steps: int = 0
    n_pde_substeps: int = 0

    @property
    def OF(self):
        return np.where(self.mdot_f > 0, self.mdot_ox / np.where(self.mdot_f > 0, self.mdot_f, 1.0), 0.0)

    @property
    def m_sliver(self):
        """Fuel remaining at the end of the run (kg), measured from the final level set."""
        return self.m_fuel0 - self.m_burned_geo[-1]

    @property
    def ledger_defect(self):
        """Relative mismatch between ODE-ledger and level-set-measured fuel consumption."""
        m_ode = np.interp(self.t_geo, self.t, self.m_burned)
        return (self.m_burned_geo - m_ode) / self.m_fuel0


def burn_prescribed(grain, law, mdot_ox, t_end, d_step=None, levels_per_step=8, margin=2.0,
                    n_ode=8, method='hopf-lax', scheme='weno5', cfl=0.5, snapshot_times=(),
                    max_steps=100000):
    """Burn ``grain`` (Grain3D) with oxidizer flow ``mdot_ox`` (float or callable of t) until t_end
    or until the fuel is fully consumed.

    d_step: target max regression per grain step (default: one in-plane cell).
    levels_per_step, margin: offset-table resolution (d_step/levels_per_step) and range (margin*d_step).
    n_ode: RK4 substeps per grain step.
    method: 'hopf-lax' (default, exact-solution shift; needs phi exact on the whole fuel side) or
            'pde' (WENO5/Godunov/TVD-RK3; first order at corners). scheme/cfl apply to 'pde'.
    """
    f_ox = mdot_ox if callable(mdot_ox) else (lambda t, c=float(mdot_ox): c)
    g = grain.grid
    hx, hy, hz = g.h
    z, w, nz = g.z, grain.w, g.z.size
    k, n, rho = law.k(grain.L), law.n, law.rho
    d_step = min(hx, hy) if d_step is None else d_step
    dd = d_step / levels_per_step
    levels = np.arange(int(np.ceil(margin*levels_per_step)) + 1) * dd
    A_env = grain.A_env
    m_fuel0 = rho * float(np.sum(w * (A_env - grain.port_areas())))

    rec = {key: [] for key in ['t', 'mox', 'mf', 'mb', 'V', 'Ab', 'A', 'P', 'G', 'r', 'mz', 'dep']}
    t_geo, m_geo = [], []
    snaps = list(sorted(snapshot_times))
    out_snaps = []
    t_bt, m_bt = np.nan, np.nan

    phi = grain.phi.copy()
    t, m_burn = 0.0, 0.0
    depth = np.zeros(nz)
    n_steps = n_sub_total = 0

    def evaluate(A_tab, M, d, t):
        A, P = measure.hermite_eval(A_tab, M, dd, d)
        rdot, G, mz, mf = flow.station_rates(A, P, f_ox(t), z, k, n, rho, w, law.flux)
        return A, P, rdot, G, mz, mf

    def record(t, ev, d, m_burn):
        A, P, rdot, G, mz, mf = ev
        rec['t'].append(t); rec['mox'].append(f_ox(t)); rec['mf'].append(mf); rec['mb'].append(m_burn)
        rec['V'].append(np.sum(w*A)); rec['Ab'].append(np.sum(w*P))
        rec['A'].append(A); rec['P'].append(P); rec['G'].append(G); rec['r'].append(rdot)
        rec['mz'].append(mz); rec['dep'].append(depth + d)

    while n_steps < max_steps:
        A_tab = grain.table(levels, phi)
        M = measure.pchip_slopes(A_tab, dd)
        A0 = A_tab[:, 0]
        # Port mass-equivalent rho*V_port; differences from the initial value = fuel consumed
        t_geo.append(t); m_geo.append(rho*float(np.sum(w*A0)))
        while snaps and snaps[0] <= t + 1e-12:
            out_snaps.append((t, phi.astype(np.float32))); snaps.pop(0)

        web = grain.web_remaining() if phi is grain.phi else _web(grain, phi)
        d = np.zeros(nz)
        ev = evaluate(A_tab, M, d, t)
        if t >= t_end - 1e-12 or not np.any(ev[1] > 0.0):
            record(t, ev, d, m_burn)
            break

        rmax = float(np.max(ev[2]))
        dt_g = (t_end - t) if rmax <= 0.0 else min(d_step / rmax, t_end - t)
        # RK4 on (d, m_burn); retry with a smaller step if the offset table range would be exceeded
        while True:
            dt = dt_g / n_ode
            ts, ds, ms, evs = t, d.copy(), m_burn, []
            ok = True
            for _ in range(n_ode):
                e1 = evaluate(A_tab, M, ds, ts)
                evs.append((ts, e1, ds.copy(), ms))
                e2 = evaluate(A_tab, M, ds + 0.5*dt*e1[2], ts + 0.5*dt)
                e3 = evaluate(A_tab, M, ds + 0.5*dt*e2[2], ts + 0.5*dt)
                e4 = evaluate(A_tab, M, ds + dt*e3[2], ts + dt)
                ds = ds + dt/6*(e1[2] + 2*e2[2] + 2*e3[2] + e4[2])
                ms = ms + dt/6*(e1[5] + 2*e2[5] + 2*e3[5] + e4[5])
                ts = ts + dt
                if np.any((ds > levels[-1]) & (e4[1] > 0.0)):
                    ok = False
                    break
            if ok:
                break
            dt_g *= 0.5

        for (ts_i, e_i, d_i, m_i) in evs:
            record(ts_i, e_i, d_i, m_i)
        # Burn-through: first time the front reaches the envelope at any station
        if np.isnan(t_bt):
            # A station stops regressing at the first table level at/after contact (P -> 0), never
            # before it, so a strict comparison suffices (roundoff guard only)
            target = web - 1e-9*dd
            if np.any(ds >= target):
                # Earliest crossing, linearly interpolated between RK4 substeps
                seq_t = np.array([e[0] for e in evs] + [ts])
                seq_d = np.stack([e[2] for e in evs] + [ds])          # (n_ode+1, nz)
                seq_m = np.array([e[3] for e in evs] + [ms])
                j = np.argmax(np.any(seq_d >= target, axis=1))       # first substep at/after crossing
                if j == 0:
                    t_bt, d_bt = seq_t[0], seq_d[0]
                else:
                    da, db = seq_d[j-1], seq_d[j]
                    hit = db >= target
                    frac = np.min(np.clip((target[hit] - da[hit]) / np.maximum(db[hit] - da[hit], 1e-300), 0, 1))
                    t_bt = seq_t[j-1] + frac*(seq_t[j] - seq_t[j-1])
                    d_bt = da + frac*(db - da)
                # Remaining fuel from the tabulated geometry (not the ODE ledger, which is least
                # accurate exactly at the burn-through kink)
                A_bt, _ = measure.hermite_eval(A_tab, M, dd, d_bt)
                m_bt = rho*float(np.sum(w*(A_env - A_bt)))

        # Advance the front by the accumulated per-station displacement. No reinitialization
        # (fast-marching reinitialization was measured to cost 5-7% in fuel flow).
        if method == 'hopf-lax':
            phi = levelset.hopf_lax_shift(phi, ds, z, g.h)
        elif method == 'pde':
            phi, nsub = levelset.advance(phi, ds[None, None, :], 1.0, g.h, cfl=cfl, scheme=scheme)
            n_sub_total += nsub
        else:
            raise ValueError(method)

        depth = depth + ds
        t, m_burn = ts, ms
        n_steps += 1

    # The first geometric sample is the initial state; convert volumes to consumed mass
    m_geo = np.array(m_geo)
    m_geo = m_geo - m_geo[0]

    arr = {key: np.array(v) for key, v in rec.items()}
    return BurnResult(
        z=z.copy(), t=arr['t'], mdot_ox=arr['mox'], mdot_f=arr['mf'], m_burned=arr['mb'],
        V_port=arr['V'], A_burn=arr['Ab'], A_port_z=arr['A'], P_z=arr['P'], G_z=arr['G'],
        rdot_z=arr['r'], mdot_z=arr['mz'], depth_z=arr['dep'], m_fuel0=m_fuel0,
        t_geo=np.array(t_geo), m_burned_geo=m_geo, t_burnthrough=t_bt, m_sliver_burnthrough=m_bt,
        snapshots=out_snaps, n_steps=n_steps, n_pde_substeps=n_sub_total,
    )


def _web(grain, phi):
    saved = grain.phi
    try:
        grain.phi = phi
        return grain.web_remaining()
    finally:
        grain.phi = saved
