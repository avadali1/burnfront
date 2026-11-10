"""Full-engine simulation with a 3D level-set grain (tank -> injector -> 3D grain -> chamber -> nozzle).

Multi-rate coupling
-------------------
The engine (tank, chamber, nozzle and the tabulated axial grain of hrap.grain_axial) is integrated
in JAX with the usual fixed time step. The 3D geometry enters through per-station offset tables
A(z, d), dA/dd measured from the level set phi:

  repeat
    1. tabulate A(z, d) for d in [0, margin*d_step] from the current phi (exact offsets of a signed
       distance field, monotone cubic Hermite in d), station depths d(z) := 0
    2. integrate the engine (one jitted while-loop) until the deepest station has regressed d_step
       or the end time is reached; the grain part integrates dd/dt = rdot(z) = k G(z)^n with
       G(z) = (mdot_ox + fuel added upstream)/A_port(z), coupled to the live injector flow and
       chamber pressure
    3. move the front by d(z) along the local normal (exact Hopf-Lax update, see
       hrap.burn3d.levelset.hopf_lax_shift) and re-tabulate

Within a segment the geometry is the exact offset geometry of phi, so the only coupling error is
the level-set update itself (verified ~1e-5 of the fuel mass per segment, recorded in
``EngineResult.geo_defect``). Twisted stars, helical ports and multi-port grains need nothing special.
"""
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

import hrap.core as core
from hrap.burn3d import measure, levelset
from hrap.burn3d.burn import _web
from hrap.grain_axial import make_axial_grain

AIR_M, T_AMB = 28.97, 294.0


@dataclass
class EngineResult:
    t: np.ndarray                  # (Nt,)
    x: np.ndarray                  # (Nt, Nx) engine state and derived quantities per record
    xmap: dict
    z: np.ndarray                  # (nz,) axial stations
    depth_z: np.ndarray            # (Nt, nz) cumulative normal regression per station
    mode: str = '3d'
    rho_f: float = 0.0
    m_fuel0: float = 0.0
    t_segments: np.ndarray = None  # start time of each grain step (3D)
    geo_defect: np.ndarray = None  # per segment: (re-measured - tabulated) fuel volume / initial fuel volume
    t_burnthrough: float = np.nan
    snapshots: list = field(default_factory=list)  # [(t, phi float32)] (3D) or [(t, depth_z)] (2D)
    grain: object = None           # Grain3D (3D) or SectionTable (2D)
    info: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.x[:, self.xmap[key]]

    def station(self, prefix):
        """(Nt, nz) per-station history, e.g. 'grn_G', 'grn_rdot', 'grn_d'."""
        return np.stack([self.x[:, self.xmap[f'{prefix}{i}']] for i in range(self.z.size)], axis=1)

    @property
    def m_fuel_burned(self):
        """Fuel consumed from the geometry (tabulated volume), kg, per record."""
        V = self['grn_V']
        return self.rho_f*(V[0] - V)

    def summary(self):
        t, F = self.t, self['noz_thrust']
        Pc = self['cmbr_P']
        mf, mo = self['grn_mdot'], self['tnk_mdot_inj']
        I = float(np.trapezoid(F, t))
        on = F > 0.05*F.max() if F.max() > 0 else np.zeros_like(F, bool)
        t_on = t[on]
        T_burn = float(t_on[-1] - t_on[0]) if t_on.size > 1 else 0.0
        m_ox = float(np.trapezoid(mo, t))
        m_f = float(np.trapezoid(mf, t))
        main = mf > 0
        return {
            'impulse_Ns': I,
            'burn_time_s': T_burn,
            'thrust_avg_N': I/T_burn if T_burn > 0 else 0.0,
            'thrust_peak_N': float(F.max()),
            'Pc_peak_Pa': float(Pc.max()),
            'Pc_avg_Pa': float(np.mean(Pc[on])) if on.any() else float(Pc[0]),
            'Isp_s': I/((m_ox + m_f)*9.80665) if m_ox + m_f > 0 else 0.0,
            'm_ox_used_kg': m_ox,
            'm_fuel_used_kg': m_f,
            'OF_mean': m_ox/m_f if m_f > 0 else np.inf,
            'OF_final': float(self['cmbr_OF'][main][-1]) if main.any() else 0.0,
            'G_head_peak': float(self['grn_G_head'].max()),
            'fuel_remaining_kg': float(self.m_fuel0 - self.m_fuel_burned[-1]),
            't_burnthrough_s': float(self.t_burnthrough),
            'tank_P_final_Pa': float(self['tnk_P'][-1]),
            # Time (s, while thrusting) the propellant table had to be clamped: O/F or Pc outside its range
            'chem_OF_clamped_s': self._time_flag('cmbr_chem_clamp', 1, on),
            'chem_Pc_clamped_s': self._time_flag('cmbr_chem_clamp', 2, on),
            'nozzle_separated_s': float(np.sum(self['noz_separated'][on] > 0.5)*self._dt) if 'noz_separated' in self.xmap else 0.0,
        }

    @property
    def _dt(self):
        return float(self.t[1] - self.t[0]) if self.t.size > 1 else 0.0

    def _time_flag(self, key, bit, on):
        if key not in self.xmap:
            return 0.0
        f = np.round(self[key]).astype(int)
        return float(np.sum(((f & bit) > 0) & on)*self._dt)

    def to_csv(self, path, keys=None, stations=('grn_G', 'grn_rdot')):
        import csv
        keys = keys or ['noz_thrust', 'cmbr_P', 'cmbr_OF', 'cmbr_cstar', 'tnk_P', 'tnk_T', 'tnk_m_ox',
                        'tnk_mdot_inj', 'grn_mdot', 'grn_V', 'grn_Ab', 'grn_A_port', 'grn_G_head',
                        'grn_G_aft', 'noz_mdot']
        cols = [self.t] + [self[k] for k in keys]
        head = ['t'] + keys
        for p in stations:
            S = self.station(p)
            for i in range(S.shape[1]):
                cols.append(S[:, i]); head.append(f'{p}{i}@z={self.z[i]:.4g}')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(np.stack(cols, 1).tolist())


def ambient_gas_mass(V_gas, Pa):
    return Pa*AIR_M/(8314.46*T_AMB)*V_gas


def _composite_fderiv(method):
    def fderiv(s, x, xmap):
        for f in method['fderivs']:
            x = f(s, x, xmap)
        return x
    return fderiv


def make_segment_runner(s, method, fstep, dt, n_buf):
    """Jitted engine segment: integrate until max station depth >= d_stop or n_lim steps.

    Returns (n, x_end, buf) with buf[0..n] the records (row j = state at t0 + j dt, after fderiv).
    """
    xmap = method['xmap']
    fderiv = _composite_fderiv(method)
    i_d = jnp.array([v for k, v in xmap.items() if k.startswith('grn_d') and k[5:].isdigit()])
    nx = len(xmap)

    @jax.jit
    def run(tab_A, tab_M, x, d_stop, n_lim):
        ss = {**s, 'grn_tab_A': tab_A, 'grn_tab_M': tab_M}
        buf = jnp.zeros((n_buf + 1, nx))

        def cond(c):
            i, x, _ = c
            return (i < n_lim) & (jnp.max(x[i_d]) < d_stop)

        def body(c):
            i, x, buf = c
            x = fderiv(ss, x, xmap)
            buf = buf.at[i].set(x)
            x = fstep(ss, x, xmap, method['diff_xmap'], method['diff_dmap'], dt, fderiv)
            for fu in method['fupdates']:
                x = fu(ss, x, xmap)
            return i + 1, x, buf

        i, x, buf = jax.lax.while_loop(cond, body, (0, x, buf))
        x = fderiv(ss, x, xmap)
        buf = buf.at[i].set(x)
        return i, x, buf

    return run, i_d


def run_engine_3d(grain, law, tank, cmbr, noz, T, dt=1e-3, stepper=core.step_rk4, V_free=0.0,
                  d_step=None, levels_per_step=8, margin=2.0, n_buf=None, snapshot_times=(),
                  verbose=False, **engine_kw):
    """Fire the engine with a 3D grain (hrap.burn3d.Grain3D).

    tank, cmbr, noz: parts from make_sat_tank / make_chamber / make_cd_nozzle.
    V_free: chamber gas volume outside the grain envelope (pre/post-combustion chambers), m^3; the
            chamber volume is set to V_free + envelope volume, so the gas volume is V_free + port volume.
    d_step: max regression per grain step (default one in-plane cell).
    engine_kw: chem_interp_k/M/T, Pa, ... as for hrap.core.make_engine.
    """
    g = grain.grid
    z, w = g.z, grain.w
    d_step = min(g.h[0], g.h[1]) if d_step is None else float(d_step)
    dd = d_step/levels_per_step
    levels = np.arange(int(np.ceil(margin*levels_per_step)) + 1)*dd
    phi = grain.phi.copy()

    def tables(phi):
        A = grain.table(levels, phi)
        return A, measure.pchip_slopes(A, dd)

    A_tab, M_tab = tables(phi)
    V_env = float(np.sum(w*grain.A_env))
    V_fuel0 = float(np.sum(w*(grain.A_env - A_tab[:, 0])))
    OD = 2*np.sqrt(np.max(grain.A_env)/np.pi)
    grn = make_axial_grain(A_tab, M_tab, dd, grain.A_env, grain.L, OD, law=law)
    s, x, method = core.make_engine(tank, grn, cmbr, noz, **engine_kw)
    xmap = method['xmap']
    # Chamber volume = free volume + grain envelope; gas fill from the actual gas volume (core.initialize)
    x = core.store_x(x, xmap, cmbr_V0=V_free + V_env)
    x = core.initialize(s, x, method)

    N_total = int(np.floor(T/dt + 1e-9))
    if n_buf is None:
        n_buf = min(N_total, 4000)
    run, i_d = make_segment_runner(s, method, stepper, dt, n_buf)
    i_d_np = np.asarray(i_d)

    rows, deps = [], []
    t_seg, defect, snaps_out = [], [], []
    snaps = sorted(snapshot_times)
    depth = np.zeros(z.size)
    t_bt = np.nan
    n_done = 0
    while True:
        web = _web(grain, phi)
        n_lim = min(n_buf, N_total - n_done)
        n, x_end, buf = run(jnp.asarray(A_tab), jnp.asarray(M_tab), x, d_step, n_lim)
        n = int(n)
        buf = np.asarray(buf[:n + 1])
        d_rec = buf[:, i_d_np]
        t_rec = (n_done + np.arange(n + 1))*dt
        t_seg.append(n_done*dt)
        last = (n_done + n >= N_total)
        keep = n + 1 if last else n          # the end row is the next segment's first row
        rows.append(buf[:keep]); deps.append(depth + d_rec[:keep])

        if np.isnan(t_bt):
            hit = np.any(d_rec >= web - 1e-9*dd, axis=1)
            if hit.any():
                j = int(np.argmax(hit))
                if j == 0:
                    t_bt = t_rec[0]
                else:
                    da, db = d_rec[j-1], d_rec[j]
                    m = db >= web - 1e-9*dd
                    frac = np.min(np.clip((web[m] - da[m])/np.maximum(db[m] - da[m], 1e-300), 0, 1))
                    t_bt = t_rec[j-1] + frac*dt
        d_end = np.asarray(x_end)[i_d_np]
        for ts in list(snaps):
            if ts <= t_rec[-1] + 1e-12 or last:
                j = int(np.clip(np.round((ts - t_rec[0])/dt), 0, n))
                snaps_out.append((float(t_rec[j]), levelset.hopf_lax_shift(phi, d_rec[j], z, g.h).astype(np.float32)))
                snaps.remove(ts)
        n_done += n
        if verbose:
            print(f'  grain step {len(t_seg):3d}: t = {n_done*dt:7.3f} s, max depth {d_end.max()*1e3:6.3f} mm, '
                  f'Pc = {float(x_end[xmap["cmbr_P"]])/1e5:6.2f} bar')
        if last:
            break
        # Advance the front, re-tabulate, reset station depths
        V_pred = float(np.sum(w*(grain.A_env - measure.hermite_eval(A_tab, M_tab, dd, d_end)[0])))
        phi = levelset.hopf_lax_shift(phi, d_end, z, g.h)
        A_tab, M_tab = tables(phi)
        V_new = float(np.sum(w*(grain.A_env - A_tab[:, 0])))
        defect.append((V_new - V_pred)/V_fuel0)
        depth = depth + d_end
        x = x_end.at[i_d].set(0.0)
        if n == 0:   # nothing could advance (should not happen); avoid an infinite loop
            break

    X = np.concatenate(rows)
    t = np.arange(X.shape[0])*dt
    return EngineResult(t=t, x=X, xmap=xmap, z=z.copy(), depth_z=np.concatenate(deps), mode='3d',
                        rho_f=law.rho, m_fuel0=law.rho*V_fuel0, t_segments=np.array(t_seg),
                        geo_defect=np.array(defect), t_burnthrough=float(t_bt), snapshots=snaps_out,
                        grain=grain, info=dict(d_step=d_step, dd=dd, n_segments=len(t_seg), dt=dt))
