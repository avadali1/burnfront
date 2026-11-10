"""Configuration-driven motor runs: arbitrary starting conditions, 2D/3D grain modes, parameter sweeps.

    from hrap.motor import MotorConfig, run_motor, sweep
    cfg = MotorConfig(port={'type': 'star', 'n_tips': 6, 'r_valley': 0.012, 'r_tip': 0.025,
                            'twist_turns': 1.0}, grain_OD=0.08, grain_L=0.3, mode='3d')
    res = run_motor(cfg)                  # hrap.engine3d.EngineResult
    print(res.summary())
    rows = sweep(cfg, {'inj_N': [4, 6, 8], 'fill': [0.7, 0.9]}, n_jobs=4, csv_path='sweep.csv')

All quantities are SI. Ports (``port`` dict, dimensions in m):
  {'type': 'circle', 'D': 0.05, 'cx': 0, 'cy': 0}
  {'type': 'star', 'n_tips': 6, 'r_valley': .012, 'r_tip': .025, 'rotation': 0}
  {'type': 'polygon', 'vertices': [[x, y], ...]}
  {'type': 'multi', 'ports': [port, port, ...]}          (non-overlapping ports)
  {'type': 'image', 'path': 'Grain.png'}                 (cross-section image, scaled by grain_OD)
  {'type': 'file', 'path': 'grain.step', 'units': 'mm', 'axis': 'z'}   (3D only; OD/L from CAD)
optional modifiers on any analytic/image port: 'twist_turns' (turns over the grain length) or
'twist_rate' (rad/m); 'helix_radius' + 'helix_pitch' (m, m/turn). In 2D mode these use the helical
2D approximation when ``helical_2d`` is True (otherwise they are ignored with a warning).
"""
import csv
import dataclasses
import itertools
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib.resources import files as _files
from pathlib import Path

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)

import hrap.core as core
from hrap.burn3d import shapes, RegressionLaw
from hrap.engine3d import EngineResult, run_engine_3d

_IN = 0.0254
PROP_DIR = Path(_files('hrap'))/'resources'/'propellant_configs'
STEPPERS = {'fe': core.step_fe, 'heun': core.step_heun, 'rk4': core.step_rk4}


@dataclass
class MotorConfig:
    # Propellant: name of a file in hrap/resources/propellant_configs or a path to an HRAP .mat.
    # a, n, m, rho override the file; phi_swirl multiplies the regression rate.
    propellant: str = 'ABS'
    a: float = None              # mm/s (HRAP units: rdot = a G^n L^m)
    n: float = None
    m: float = None
    rho: float = None            # fuel density, kg/m^3
    phi_swirl: float = 1.0
    flux: str = 'total'          # 'total' (ox + upstream fuel) or 'oxidizer'

    # Oxidizer tank (saturated two-phase). Give T_tank or P_tank (saturation); fill = initial liquid
    # volume fraction, or m_ox directly.
    fluid: str = 'NitrousOxide'
    tank_V: float = (np.pi/4*4.75**2*_IN**2)*(7.0*12*_IN)
    fill: float = 0.8
    m_ox: float = None
    T_tank: float = 294.0
    P_tank: float = None
    vent_CdA: float = 0.0
    vent_to_chamber: bool = False
    tank_model: str = 'equilibrium'   # 'equilibrium' (energy-conserving, default) or 'legacy'
    tank_wall_mass: float = 0.0       # kg of tank wall in thermal contact with the fluid (equilibrium model)
    tank_wall_cp: float = 900.0       # J/kg/K (aluminium ~900, steel ~490)

    # Injector
    inj_N: int = 1
    inj_d: float = 0.5*_IN       # hole diameter
    inj_Cd: float = 0.22
    inj_liq_model: str = 'Dyer'             # 'Dyer', 'HEM', 'Incompressible' (legacy tank: Incompressible only)
    inj_vap_model: str = 'Real Gas'         # compressible orifice (equilibrium tank) or 'Incompressible'

    # Grain
    mode: str = '3d'             # '3d' (level set) or '2d' (cross-section outline)
    port: dict = field(default_factory=lambda: {'type': 'circle', 'D': 2.5*_IN})
    grain_OD: float = 4.5*_IN
    grain_L: float = 30.0*_IN
    n_xy: int = 64               # 3D: cells across the grain OD
    n_z: int = 33                # 3D: axial stations
    d_step: float = None         # 3D: regression per grain step (default one cell)
    n_xy_2d: int = 256           # 2D: cells across OD for the section table
    n_z_2d: int = 21             # 2D: axial stations of the flux model
    helical_2d: bool = True

    # Chamber and nozzle
    V_free: float = 0.0          # chamber gas volume outside the grain envelope (pre/post chambers)
    cstar_eff: float = 1.0
    throat_d: float = 1.75*_IN
    ER: float = 4.99
    noz_Cd: float = 0.995
    noz_eff: float = 0.97
    noz_sep_ratio: float = 0.4   # Summerfield separation criterion p_sep/Pa (0 disables)
    Pa: float = 101325.0

    # Integration
    T: float = 10.0
    dt: float = None             # None: dt = chamber filling time constant / 10 (capped at 1 ms)
    stepper: str = 'rk4'

    def replace(self, **kw):
        """Copy with fields replaced; 'port.<key>' sets a key of the port dict."""
        port = dict(self.port)
        direct = {}
        for k, v in kw.items():
            if k.startswith('port.'):
                port[k[5:]] = v
            else:
                direct[k] = v
        if 'port' not in direct:
            direct['port'] = port
        return dataclasses.replace(self, **direct)


# --- cached resources ---------------------------------------------------------------------------
_CACHE = {}


def _cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def _prop_path(name):
    p = Path(name)
    if p.suffix == '.mat' and p.exists():
        return p
    q = PROP_DIR/(name if name.endswith('.mat') else name + '.mat')
    if not q.exists():
        raise FileNotFoundError(f'propellant {name!r} not found (available: '
                                f'{sorted(f.stem for f in PROP_DIR.glob("*.mat"))})')
    return q


def chemistry(propellant):
    """(k, M, T) interpolators over (O/F, Pc) from an HRAP propellant file."""
    def make():
        from hrap.chemtab import mat_interpolators
        return mat_interpolators(_prop_path(propellant))
    return _cached(('chem', str(propellant)), make)


def regression_law(cfg):
    kw = {k: getattr(cfg, k) for k in ('a', 'n', 'm', 'rho') if getattr(cfg, k) is not None}
    return RegressionLaw.from_mat(_prop_path(cfg.propellant), phi=cfg.phi_swirl, flux=cfg.flux, **kw)


def saturation(fluid):
    """Saturation property tables (JAX PCHIP) from the triple point to just below critical."""
    def make():
        import CoolProp.CoolProp as CP
        import hrap.fluid as hfluid
        return hfluid.bake_sat_coolprop(fluid, np.linspace(*_T_range(fluid), 256))
    return _cached(('sat', fluid), make)


def _T_range(fluid):
    import CoolProp.CoolProp as CP
    return max(CP.PropsSI('Ttriple', fluid), 150.0) + 0.5, CP.PropsSI('Tcrit', fluid) - 0.3


def hem_table(fluid):
    def make():
        import hrap.fluid as hfluid
        return hfluid.bake_hem_coolprop(fluid, np.linspace(*_T_range(fluid), 64))
    return _cached(('hem', fluid), make)


def initial_tank_state(cfg):
    """(T0, m_ox0, info) from T_tank or P_tank (saturation) and fill fraction or m_ox."""
    import CoolProp.CoolProp as CP
    sat = saturation(cfg.fluid)
    if cfg.P_tank is not None:
        T0 = float(CP.PropsSI('T', 'P', cfg.P_tank, 'Q', 0, cfg.fluid))
    else:
        T0 = float(cfg.T_tank)
    if not sat.T_min <= T0 <= sat.T_max:
        raise ValueError(f'tank temperature {T0:.2f} K outside the saturation range '
                         f'[{sat.T_min:.1f}, {sat.T_max:.1f}] K of {cfg.fluid}')
    p = sat(T0)
    rl, rv = float(p['rho_l']), float(p['rho_v'])
    if cfg.m_ox is not None:
        m = float(cfg.m_ox)
        if not cfg.tank_V*rv <= m <= cfg.tank_V*rl:
            raise ValueError(f'm_ox = {m} kg is not a two-phase state in {cfg.tank_V} m^3 at {T0:.2f} K '
                             f'(needs {cfg.tank_V*rv:.3f} .. {cfg.tank_V*rl:.3f} kg)')
    else:
        if not 0.0 <= cfg.fill <= 1.0:
            raise ValueError('fill must be in [0, 1]')
        m = cfg.tank_V*(cfg.fill*rl + (1 - cfg.fill)*rv)
    return T0, m, dict(T0=T0, P0=float(p['Pv']), m_ox0=m, rho_l=rl, rho_v=rv)


# --- grain construction -------------------------------------------------------------------------
def _base_shape(port, OD):
    t = port['type']
    if t == 'circle':
        return shapes.circle(port['D']/2, port.get('cx', 0.0), port.get('cy', 0.0))
    if t == 'star':
        return shapes.star(port['n_tips'], port['r_valley'], port['r_tip'], port.get('rotation', 0.0))
    if t == 'polygon':
        return shapes.polygon(np.asarray(port['vertices'], float))
    if t == 'multi':
        return shapes.union(*[_base_shape(p, OD) for p in port['ports']])
    if t == 'image':
        from hrap.burn3d.section import image_port_polygons
        polys, _ = image_port_polygons(port['path'], OD)
        ps = [shapes.polygon(p) for p in polys]
        return ps[0] if len(ps) == 1 else shapes.union(*ps)
    raise ValueError(f'unknown port type {t!r}')


def _twist_rate(port, L):
    if 'twist_rate' in port:
        return float(port['twist_rate'])
    return 2*np.pi*float(port.get('twist_turns', 0.0))/L


def port_shape_3d(port, OD, L):
    shp = _base_shape(port, OD)
    w = _twist_rate(port, L)
    if w:
        shp = shapes.twist(shp, w)
    if port.get('helix_radius'):
        shp = shapes.helix(shp, port['helix_radius'], port['helix_pitch'])
    return shp


def _grain_key(cfg):
    if cfg.mode == '3d':
        return ('3d', repr(cfg.port), cfg.grain_OD, cfg.grain_L, cfg.n_xy, cfg.n_z)
    return ('2d', repr(cfg.port), cfg.grain_OD, cfg.grain_L, cfg.n_xy_2d, cfg.helical_2d)


def build_grain(cfg, verbose=False):
    """3D: Grain3D. 2D: (SectionTable, kappa or None, Section2D). Cached per geometry."""
    def make():
        port = cfg.port
        if cfg.mode == '3d':
            if port['type'] == 'file':
                from hrap.burn3d import cad
                g, rep = cad.load_grain(port['path'], units=port.get('units'), axis=port.get('axis', 'z'),
                                        n_xy=cfg.n_xy, n_z=cfg.n_z, verbose=verbose)
                g.report = rep   # mesh validation report (shown by the GUI)
                return g
            return shapes.cylindrical_grain(port_shape_3d(port, cfg.grain_OD, cfg.grain_L),
                                            cfg.grain_OD, cfg.grain_L, n_xy=cfg.n_xy, n_z=cfg.n_z)
        if cfg.mode == '2d':
            from hrap.burn3d.section import Section2D
            if port['type'] == 'file':
                raise ValueError("2D mode needs a cross-section port (circle/star/polygon/multi/image), "
                                 "not a CAD file; use mode='3d'")
            sec = Section2D(_base_shape(port, cfg.grain_OD), cfg.grain_OD, n_xy=cfg.n_xy_2d)
            tab = sec.table()
            w = _twist_rate(port, cfg.grain_L)
            hr = port.get('helix_radius', 0.0)
            kap = None
            if w or hr:
                if cfg.helical_2d:
                    kap = sec.helical_kappa(tab, twist_rate=w, helix_radius=hr, helix_pitch=port.get('helix_pitch'))
                else:
                    warnings.warn('twist/helix ignored in 2D mode (helical_2d=False)')
            return tab, kap, sec
        raise ValueError(f"mode must be '3d' or '2d', got {cfg.mode!r}")
    return _cached(('grain',) + _grain_key(cfg), make)


# --- engine assembly and run ----------------------------------------------------------------------
def _parts(cfg):
    from hrap.tank import make_sat_tank, make_eq_tank
    from hrap.chamber import make_chamber
    from hrap.nozzle import make_cd_nozzle
    T0, m0, info = initial_tank_state(cfg)
    if cfg.tank_model == 'equilibrium':
        hem = hem_table(cfg.fluid) if cfg.inj_liq_model in ('HEM', 'Dyer') else None
        make_tank = lambda *a, **k: make_eq_tank(*a, get_hem=hem, wall_C=cfg.tank_wall_mass*cfg.tank_wall_cp, **k)
    elif cfg.tank_model == 'legacy':
        if cfg.inj_liq_model != 'Incompressible':
            raise ValueError("the legacy tank supports inj_liq_model='Incompressible' only")
        if cfg.tank_wall_mass:
            raise ValueError('tank wall heat capacity needs the equilibrium tank')
        make_tank = make_sat_tank
    else:
        raise ValueError("tank_model must be 'equilibrium' or 'legacy'")
    tnk = make_tank(
        saturation(cfg.fluid), V=cfg.tank_V, T=T0, m_ox=m0,
        inj_CdA=cfg.inj_Cd*np.pi/4*cfg.inj_d**2, inj_N=int(cfg.inj_N),
        inj_liq_model=core.StaticVar(cfg.inj_liq_model), inj_vap_model=core.StaticVar(cfg.inj_vap_model),
        vnt_S=(2 if cfg.vent_to_chamber else 1) if cfg.vent_CdA > 0 else 0, vnt_CdA=cfg.vent_CdA,
    )
    cmbr = make_chamber(cstar_eff=cfg.cstar_eff)
    noz = make_cd_nozzle(thrt=cfg.throat_d, ER=cfg.ER, eff=cfg.noz_eff, Cd=cfg.noz_Cd, sep_ratio=cfg.noz_sep_ratio)
    ik, iM, iT = chemistry(cfg.propellant)
    return tnk, cmbr, noz, dict(chem_interp_k=ik, chem_interp_M=iM, chem_interp_T=iT, Pa=cfg.Pa), info


def run_motor(cfg, grain=None, snapshot_times=(), verbose=False):
    """Run one motor configuration. ``grain`` optionally supplies a prebuilt grain (see build_grain)."""
    law = regression_law(cfg)
    tnk, cmbr, noz, kw, info = _parts(cfg)
    stepper = STEPPERS[cfg.stepper]
    grain = build_grain(cfg, verbose) if grain is None else grain
    if cfg.dt is None:
        cfg = dataclasses.replace(cfg, dt=auto_dt(cfg, grain))
    t0 = time.time()
    if cfg.mode == '3d':
        res = run_engine_3d(grain, law, tnk, cmbr, noz, cfg.T, dt=cfg.dt, stepper=stepper, V_free=cfg.V_free,
                            d_step=cfg.d_step, snapshot_times=snapshot_times, verbose=verbose, **kw)
    else:
        res = _run_2d(cfg, grain, law, tnk, cmbr, noz, stepper, kw)
    res.info.update(info, wall_time=time.time() - t0, config=cfg)
    return res


def initial_port_volume(cfg, grain):
    if cfg.mode == '3d':
        return float(np.sum(grain.w*grain.port_areas()))
    return cfg.grain_L*float(grain[0].A[0])


def auto_dt(cfg, grain):
    """Time step resolving the chamber filling/emptying time constant
    tau = m/mdot_out = V/(Cd At Gamma sqrt(RT)), with sqrt(RT) = Gamma c* and a high c* = 1800 m/s
    (shortest tau), at the initial (smallest) gas volume: dt = tau/10, at most 1 ms."""
    G = 0.6581  # Gamma(1.2)
    At = np.pi/4*cfg.throat_d**2
    V = cfg.V_free + initial_port_volume(cfg, grain)
    tau = V/(cfg.noz_Cd*At*G*G*1800.0*cfg.cstar_eff)
    return float(min(1e-3, tau/10))


def _run_2d(cfg, grain, law, tnk, cmbr, noz, stepper, kw):
    from hrap.grain_axial import make_axial_grain
    tab, kap, sec = grain
    A, M = tab.stations(cfg.n_z_2d)
    grn = make_axial_grain(A, M, tab.dd, tab.A_env, cfg.grain_L, cfg.grain_OD, law=law, kappa=kap)
    s, x, method = core.make_engine(tnk, grn, cmbr, noz, **kw)
    xm = method['xmap']
    V_env = cfg.grain_L*tab.A_env
    V_fuel0 = cfg.grain_L*(tab.A_env - float(tab.A[0]))
    x = core.store_x(x, xm, cmbr_V0=cfg.V_free + V_env)
    x = core.initialize(s, x, method)
    t, _, xs = core.make_integrator(stepper, method)(s, x, dt=cfg.dt, T=cfg.T, do_init=False)
    xs = np.asarray(xs)
    z = np.linspace(0.0, cfg.grain_L, cfg.n_z_2d)
    d = np.stack([xs[:, xm[f'grn_d{i}']] for i in range(z.size)], 1)
    # In-plane depth -> burn-through when any station reaches the section's contact offset
    hit = np.any(d >= tab.web_contact, axis=1)
    t_arr = np.asarray(t)
    t_bt = float(t_arr[np.argmax(hit)]) if hit.any() else np.nan
    if hit.any() and np.argmax(hit) > 0:
        j = int(np.argmax(hit))
        da, db = d[j-1].max(), d[j].max()
        t_bt = float(t_arr[j-1] + (tab.web_contact - da)/max(db - da, 1e-300)*(t_arr[j] - t_arr[j-1]))
    return EngineResult(t=t_arr, x=xs, xmap=xm, z=z, depth_z=d, mode='2d', rho_f=law.rho,
                        m_fuel0=law.rho*V_fuel0, t_burnthrough=t_bt, grain=grain,
                        info=dict(dt=cfg.dt, kappa=kap is not None))


# --- sweeps ---------------------------------------------------------------------------------------
def sweep_configs(base, params, mode='grid'):
    """List of (overrides dict, MotorConfig). mode 'grid' = Cartesian product, 'zip' = paired values."""
    keys = list(params)
    vals = [list(params[k]) for k in keys]
    if mode == 'grid':
        combos = itertools.product(*vals)
    elif mode == 'zip':
        if len({len(v) for v in vals}) > 1:
            raise ValueError("zip sweep needs equally long value lists")
        combos = zip(*vals)
    else:
        raise ValueError("mode must be 'grid' or 'zip'")
    out = []
    for c in combos:
        ov = dict(zip(keys, c))
        out.append((ov, base.replace(**ov)))
    return out


def _worker(i, ov, cfg, grain, history_dir):
    try:
        res = run_motor(cfg, grain=grain)
        row = res.summary()
        row['wall_time_s'] = res.info['wall_time']
        if history_dir is not None:
            p = Path(history_dir)/f'run_{i:04d}.csv'
            res.to_csv(p)
            row['history_csv'] = str(p)
        row['status'] = 'ok'
    except Exception as e:  # report and keep sweeping
        row = {'status': f'error: {type(e).__name__}: {e}'}
    return i, {**{k: v for k, v in ov.items()}, **row}


def sweep(base, params, mode='grid', n_jobs=1, csv_path=None, history_dir=None, verbose=True):
    """Run a parameter sweep over MotorConfig fields (or 'port.<key>').

    n_jobs > 1 runs configurations in parallel processes. Grains are built once per distinct geometry
    in the parent process and shipped to the workers. Writes one CSV row per run (swept values +
    summary metrics) when csv_path is given, and per-run time histories into history_dir.
    Returns the list of row dicts in sweep order.
    """
    runs = sweep_configs(base, params, mode)
    if history_dir is not None:
        Path(history_dir).mkdir(parents=True, exist_ok=True)
    grains = {}
    for _, cfg in runs:
        k = _grain_key(cfg)
        if k not in grains:
            g = build_grain(cfg)
            # Section2D holds closures (not picklable); the engine only needs the table and kappa
            grains[k] = (g[0], g[1], None) if cfg.mode == '2d' else g
    rows = [None]*len(runs)
    if n_jobs == 1:
        for i, (ov, cfg) in enumerate(runs):
            rows[i] = _worker(i, ov, cfg, grains[_grain_key(cfg)], history_dir)[1]
            if verbose:
                print(f'[{i+1}/{len(runs)}] {ov} -> {rows[i].get("status")}')
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn')) as ex:
            futs = [ex.submit(_worker, i, ov, cfg, grains[_grain_key(cfg)], history_dir)
                    for i, (ov, cfg) in enumerate(runs)]
            for n_done, f in enumerate(as_completed(futs), 1):
                i, row = f.result()
                rows[i] = row
                if verbose:
                    print(f'[{n_done}/{len(runs)}] {runs[i][0]} -> {row.get("status")}')
    if csv_path is not None:
        write_rows(csv_path, rows)
    return rows


def write_rows(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
