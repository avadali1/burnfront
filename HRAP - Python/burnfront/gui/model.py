"""GUI-independent logic for the Burnfront motor GUI (testable without a display).

- FIELDS: the editable inputs (section, label, MotorConfig field, unit, kind)
- values_to_config / config_to_values: widget values (GUI units) <-> MotorConfig (SI)
- grain views: regressing-surface triangle meshes from level-set snapshots, offscreen rendering
  to an RGBA image, cross-section contours, 2D outlines
- export helpers
"""
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from burnfront.motor import MotorConfig, PROP_DIR

MODE_2D, MODE_3D = '2D outline', '3D (STEP/STL)'
PORT_TYPES = ['Circle', 'Star', 'STEP/STL file', 'Cross-section image']


@dataclass
class Field:
    key: str              # widget tag / logical name
    label: str
    section: str
    kind: str = 'float'   # float, int, combo, bool, text, file
    unit: str = ''
    scale: float = 1.0    # SI = GUI * scale
    items: list = field(default_factory=list)
    default: object = None
    tip: str = ''
    show: str = ''        # visibility condition key (see visible())


def _props():
    return sorted(p.stem for p in PROP_DIR.glob('*.mat'))


_D = MotorConfig()
FIELDS = [
    # --- oxidizer tank
    Field('fluid', 'Oxidizer', 'Tank', 'combo', items=['NitrousOxide'], default='NitrousOxide',
          tip='Propellant chemistry tables are for N2O'),
    Field('tank_V', 'Tank volume', 'Tank', unit='L', scale=1e-3, default=3.0),
    Field('init_by', 'Initial state from', 'Tank', 'combo', items=['Temperature', 'Pressure'], default='Temperature'),
    Field('T_tank', 'Tank temperature', 'Tank', unit='K', default=294.0, show='init_T'),
    Field('P_tank', 'Tank pressure', 'Tank', unit='bar', scale=1e5, default=50.0, show='init_P'),
    Field('fill', 'Liquid fill (by volume)', 'Tank', unit='%', scale=1e-2, default=85.0),
    Field('tank_model', 'Tank model', 'Tank', 'combo', items=['equilibrium', 'legacy'], default='equilibrium'),
    Field('tank_wall_mass', 'Wall mass in contact', 'Tank', unit='kg', default=0.0, show='eq_tank'),
    Field('tank_wall_cp', 'Wall specific heat', 'Tank', unit='J/kg/K', default=900.0, show='eq_tank'),
    Field('vent_CdA', 'Vent CdA', 'Tank', unit='mm2', scale=1e-6, default=0.0),
    # --- injector
    Field('inj_N', 'Injector holes', 'Injector', 'int', default=6),
    Field('inj_d', 'Hole diameter', 'Injector', unit='mm', scale=1e-3, default=1.5),
    Field('inj_Cd', 'Discharge coefficient', 'Injector', default=0.65),
    Field('inj_liq_model', 'Liquid model', 'Injector', 'combo', items=['Dyer', 'HEM', 'Incompressible'], default='Dyer'),
    Field('inj_vap_model', 'Vapor model', 'Injector', 'combo', items=['Real Gas', 'Incompressible'], default='Real Gas'),
    # --- grain
    Field('mode', 'Grain model', 'Grain', 'combo', items=[MODE_3D, MODE_2D], default=MODE_3D),
    Field('port_type', 'Port', 'Grain', 'combo', items=PORT_TYPES, default='Star'),
    Field('port_D', 'Port diameter', 'Grain', unit='mm', scale=1e-3, default=25.0, show='circle'),
    Field('star_n', 'Star points', 'Grain', 'int', default=6, show='star'),
    Field('star_r_valley', 'Star valley radius', 'Grain', unit='mm', scale=1e-3, default=10.0, show='star'),
    Field('star_r_tip', 'Star tip radius', 'Grain', unit='mm', scale=1e-3, default=18.0, show='star'),
    Field('star_rot', 'Star rotation', 'Grain', unit='deg', scale=np.pi/180, default=0.0, show='star'),
    Field('twist_turns', 'Twist (turns over length)', 'Grain', default=0.0, show='analytic'),
    Field('helix_radius', 'Helix radius', 'Grain', unit='mm', scale=1e-3, default=0.0, show='analytic'),
    Field('helix_pitch', 'Helix pitch', 'Grain', unit='mm', scale=1e-3, default=100.0, show='analytic'),
    Field('file_path', 'Grain file', 'Grain', 'file', default='', show='file'),
    Field('file_units', 'File units', 'Grain', 'combo', items=['mm', 'm', 'cm', 'in', 'ft'], default='mm', show='file'),
    Field('file_axis', 'Port axis', 'Grain', 'combo', items=['z', 'x', 'y', '-z', '-x', '-y'], default='z', show='file'),
    Field('image_path', 'Section image', 'Grain', 'file', default='', show='image'),
    Field('grain_OD', 'Grain OD', 'Grain', unit='mm', scale=1e-3, default=70.0, show='not_file'),
    Field('grain_L', 'Grain length', 'Grain', unit='mm', scale=1e-3, default=300.0, show='not_file'),
    Field('propellant', 'Fuel', 'Grain', 'combo', items=_props(), default='ABS'),
    Field('a', 'Regression a (blank = file)', 'Grain', 'text', unit='mm/s', default=''),
    Field('n', 'Regression n (blank = file)', 'Grain', 'text', default=''),
    Field('phi_swirl', 'Swirl multiplier', 'Grain', default=1.0),
    Field('flux', 'Flux in rdot = aG^n', 'Grain', 'combo', items=['total', 'oxidizer'], default='total'),
    Field('n_xy', '3D cells across OD', 'Grain', 'int', default=48, show='3d'),
    Field('n_z', '3D axial stations', 'Grain', 'int', default=33, show='3d'),
    Field('n_xy_2d', '2D cells across OD', 'Grain', 'int', default=256, show='2d'),
    Field('n_z_2d', '2D axial stations', 'Grain', 'int', default=21, show='2d'),
    Field('helical_2d', 'Helical 2D correction', 'Grain', 'bool', default=True, show='2d'),
    # --- chamber and nozzle
    Field('V_free', 'Free chamber volume', 'Chamber & Nozzle', unit='cm3', scale=1e-6, default=200.0,
          tip='Pre/post-combustion volume outside the grain envelope'),
    Field('cstar_eff', 'c* efficiency', 'Chamber & Nozzle', default=0.95),
    Field('throat_d', 'Throat diameter', 'Chamber & Nozzle', unit='mm', scale=1e-3, default=15.0),
    Field('ER', 'Expansion ratio', 'Chamber & Nozzle', default=5.0),
    Field('noz_Cd', 'Nozzle Cd', 'Chamber & Nozzle', default=0.98),
    Field('noz_eff', 'Thrust efficiency', 'Chamber & Nozzle', default=0.97),
    Field('noz_sep_ratio', 'Separation p/Pa', 'Chamber & Nozzle', default=0.4),
    Field('noz_expansion', 'Nozzle expansion', 'Chamber & Nozzle', 'combo',
          items=['shifting equilibrium', 'frozen'], default='shifting equilibrium',
          tip='Shifting: products re-equilibrate as they expand (Cantera, validated against NASA CEA).\n'
              'The first run per fuel builds a cached table (a few minutes).'),
    Field('Pa', 'Ambient pressure', 'Chamber & Nozzle', unit='kPa', scale=1e3, default=101.325),
    # --- nozzle geometry and wall (CFD and thermal analysis)
    Field('conv_angle', 'Convergent half-angle', 'Nozzle geometry & wall', unit='deg', default=45.0),
    Field('div_angle', 'Divergent half-angle (cone)', 'Nozzle geometry & wall', unit='deg', default=15.0),
    Field('rcu_ratio', 'Throat upstream radius / Rt', 'Nozzle geometry & wall', default=1.5),
    Field('rcd_ratio', 'Throat downstream radius / Rt', 'Nozzle geometry & wall', default=0.382),
    Field('wall_material', 'Throat/nozzle material', 'Nozzle geometry & wall', 'combo',
          items=['Graphite', 'Copper', 'Steel (4130)', 'Aluminium (6061)', 'Silica phenolic'], default='Graphite'),
    Field('wall_thickness', 'Wall thickness', 'Nozzle geometry & wall', unit='mm', scale=1e-3, default=10.0),
    # --- numerics
    Field('T', 'Simulated time', 'Numerics', unit='s', default=10.0),
    Field('auto_dt', 'Automatic time step', 'Numerics', 'bool', default=True),
    Field('dt', 'Time step', 'Numerics', unit='ms', scale=1e-3, default=1.0, show='manual_dt'),
    Field('stepper', 'Integrator', 'Numerics', 'combo', items=['rk4', 'heun', 'fe'], default='rk4'),
    Field('snapshots', 'Surface snapshot times', 'Numerics', 'text', unit='s', default='0, 1, 2, 4, 6'),
]
FIELD = {f.key: f for f in FIELDS}
SECTIONS = list(dict.fromkeys(f.section for f in FIELDS))


def defaults():
    return {f.key: f.default for f in FIELDS}


def visible(f, v):
    """Whether field f is relevant for the current values v."""
    pt = v.get('port_type')
    is3d = v.get('mode') == MODE_3D
    cond = {
        '': True,
        'init_T': v.get('init_by') == 'Temperature',
        'init_P': v.get('init_by') == 'Pressure',
        'eq_tank': v.get('tank_model') == 'equilibrium',
        'circle': pt == 'Circle',
        'star': pt == 'Star',
        'analytic': pt in ('Circle', 'Star', 'Cross-section image'),
        'file': pt == 'STEP/STL file',
        'image': pt == 'Cross-section image',
        'not_file': pt != 'STEP/STL file',
        '3d': is3d,
        '2d': not is3d,
        'manual_dt': not v.get('auto_dt', True),
    }
    return cond[f.show]


def parse_times(text):
    out = []
    for tok in str(text).replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return sorted(set(out))


def _opt_float(s):
    s = str(s).strip()
    return None if s == '' else float(s)


def values_to_config(v):
    """Widget values (GUI units) -> (MotorConfig, snapshot_times). Raises ValueError on bad input."""
    si = {}
    for f in FIELDS:
        if f.kind in ('float', 'int'):
            si[f.key] = (int(v[f.key]) if f.kind == 'int' else float(v[f.key])*f.scale)
    pt = v['port_type']
    if pt == 'Circle':
        port = {'type': 'circle', 'D': si['port_D']}
    elif pt == 'Star':
        if not si['star_r_tip'] > si['star_r_valley'] > 0:
            raise ValueError('star needs tip radius > valley radius > 0')
        port = {'type': 'star', 'n_tips': si['star_n'], 'r_valley': si['star_r_valley'],
                'r_tip': si['star_r_tip'], 'rotation': si['star_rot']}
    elif pt == 'STEP/STL file':
        if not v['file_path']:
            raise ValueError('choose a STEP or STL grain file')
        if v['mode'] != MODE_3D:
            raise ValueError('CAD grains need the 3D mode (2D needs a cross-section)')
        port = {'type': 'file', 'path': v['file_path'], 'units': v['file_units'], 'axis': v['file_axis']}
    elif pt == 'Cross-section image':
        if not v['image_path']:
            raise ValueError('choose a cross-section image')
        port = {'type': 'image', 'path': v['image_path']}
    else:
        raise ValueError(pt)
    if pt != 'STEP/STL file':
        if si['twist_turns']:
            port['twist_turns'] = si['twist_turns']
        if si['helix_radius']:
            port['helix_radius'] = si['helix_radius']
            port['helix_pitch'] = si['helix_pitch']
    cfg = MotorConfig(
        propellant=v['propellant'], a=_opt_float(v['a']), n=_opt_float(v['n']), phi_swirl=si['phi_swirl'],
        flux=v['flux'], fluid=v['fluid'], tank_V=si['tank_V'], fill=si['fill'],
        T_tank=si['T_tank'] if v['init_by'] == 'Temperature' else None,
        P_tank=si['P_tank'] if v['init_by'] == 'Pressure' else None,
        vent_CdA=si['vent_CdA'], tank_model=v['tank_model'],
        tank_wall_mass=si['tank_wall_mass'] if v['tank_model'] == 'equilibrium' else 0.0,
        tank_wall_cp=si['tank_wall_cp'],
        inj_N=si['inj_N'], inj_d=si['inj_d'], inj_Cd=si['inj_Cd'],
        inj_liq_model=v['inj_liq_model'] if v['tank_model'] == 'equilibrium' else 'Incompressible',
        inj_vap_model=v['inj_vap_model'],
        mode='3d' if v['mode'] == MODE_3D else '2d', port=port,
        grain_OD=si['grain_OD'], grain_L=si['grain_L'], n_xy=si['n_xy'], n_z=si['n_z'],
        n_xy_2d=si['n_xy_2d'], n_z_2d=si['n_z_2d'], helical_2d=bool(v['helical_2d']),
        V_free=si['V_free'], cstar_eff=si['cstar_eff'], throat_d=si['throat_d'], ER=si['ER'],
        noz_Cd=si['noz_Cd'], noz_eff=si['noz_eff'], noz_sep_ratio=si['noz_sep_ratio'], Pa=si['Pa'],
        noz_expansion='frozen' if v.get('noz_expansion') == 'frozen' else 'equilibrium',
        T=si['T'], dt=None if v['auto_dt'] else si['dt'], stepper=v['stepper'],
    )
    for name, val in [('tank volume', cfg.tank_V), ('injector diameter', cfg.inj_d), ('throat', cfg.throat_d),
                      ('simulated time', cfg.T)]:
        if not val > 0:
            raise ValueError(f'{name} must be positive')
    if cfg.ER < 1:
        raise ValueError('expansion ratio must be >= 1')
    return cfg, parse_times(v['snapshots'])


def save_values(path, v):
    Path(path).write_text(json.dumps({'hrap_gui': 1, 'values': v}, indent=1))


def load_values(path):
    d = json.loads(Path(path).read_text())
    out = defaults()
    out.update({k: val for k, val in d.get('values', {}).items() if k in FIELD})
    return out


# --- grain views ------------------------------------------------------------------------------------
def port_surface(grain, phi, max_faces=40000):
    """Triangle mesh (verts in mm, faces, burning flag per face) of the open port boundary
    {max(phi, chi) = 0} of a Grain3D level set: burning front where phi = 0, exposed case where chi = 0."""
    from skimage.measure import marching_cubes
    g = grain.grid
    chi = np.broadcast_to(grain.chi, g.shape)
    psi = np.maximum(phi, chi).astype(np.float32)
    # Close the ends so the cavity renders as a closed surface
    pad = np.pad(psi, 1, mode='constant', constant_values=float(psi.max()) + 1.0)
    h = np.array(g.h)
    step = 1
    while True:
        try:
            V, F, _, _ = marching_cubes(pad, 0.0, spacing=tuple(h), step_size=step)
        except (ValueError, RuntimeError):
            return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros(0, bool)
        if len(F) <= max_faces or step >= 4:
            break
        step += 1
    V = V - h + np.array([g.x[0], g.y[0], g.z[0]])
    # Burning faces: front closer than the case
    from scipy.ndimage import map_coordinates
    c = V[F].mean(axis=1)
    idx = [(c[:, i] - [g.x[0], g.y[0], g.z[0]][i])/h[i] for i in range(3)]
    ph = map_coordinates(np.asarray(phi, float), idx, order=1, mode='nearest')
    ch = map_coordinates(np.asarray(chi, float), idx, order=1, mode='nearest')
    burning = ph >= ch
    return V*1e3, F, burning


def render_surface(V, F, burning, R_case_mm, L_mm, z0_mm=0.0, az=-60.0, el=20.0, size=(640, 480),
                   title='', dark=True):
    """Offscreen matplotlib render -> float32 RGBA array (h, w, 4) in [0, 1]."""
    import matplotlib
    matplotlib.use('Agg', force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import LightSource

    w, h = size
    fig = Figure(figsize=(w/100, h/100), dpi=100)
    FigureCanvasAgg(fig)
    bg = (0.12, 0.12, 0.13) if dark else (1, 1, 1)
    fg = (0.85, 0.85, 0.85) if dark else (0.1, 0.1, 0.1)
    fig.patch.set_facecolor(bg)
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.set_facecolor(bg)
    if len(F):
        tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        ls = LightSource(azdeg=315, altdeg=45)
        light = ls.direction if hasattr(ls, 'direction') else np.array([0.5, 0.5, 0.7])
        shade = 0.35 + 0.65*np.abs(n @ (light/np.linalg.norm(light)))
        base = np.where(burning[:, None], np.array([[0.95, 0.55, 0.15]]), np.array([[0.55, 0.6, 0.7]]))
        cols = np.clip(base*shade[:, None], 0, 1)
        # Map the port axis (z) to the plot's horizontal x axis for a natural side view
        pc = Poly3DCollection(tri[:, :, [2, 0, 1]], facecolors=cols, edgecolors='none', linewidths=0)
        ax.add_collection3d(pc)
    # Case outline
    th = np.linspace(0, 2*np.pi, 73)
    for zz in (z0_mm, z0_mm + L_mm):
        ax.plot(np.full_like(th, zz), R_case_mm*np.cos(th), R_case_mm*np.sin(th), color=fg, lw=0.8, alpha=0.7)
    for a in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ax.plot([z0_mm, z0_mm + L_mm], [R_case_mm*np.cos(a)]*2, [R_case_mm*np.sin(a)]*2, color=fg, lw=0.4, alpha=0.35)
    ax.set_xlim(z0_mm, z0_mm + L_mm)
    ax.set_ylim(-R_case_mm, R_case_mm)
    ax.set_zlim(-R_case_mm, R_case_mm)
    ax.set_box_aspect((L_mm, 2*R_case_mm, 2*R_case_mm))
    ax.view_init(elev=el, azim=az)
    ax.set_xlabel('z [mm]', color=fg)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((*bg, 0.0))
        axis.label.set_color(fg)
    ax.tick_params(colors=fg, labelsize=7)
    ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, color=fg, fontsize=10)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32)/255.0
    return np.ascontiguousarray(img)


def section_contours_3d(grain, phi, z_frac):
    """Front polylines (mm) in the slice nearest z = z_frac*L of a 3D level set, clipped to the case."""
    from skimage.measure import find_contours
    g = grain.grid
    k = int(round(np.clip(z_frac, 0, 1)*(g.z.size - 1)))
    chi = np.broadcast_to(grain.chi, g.shape)[:, :, k]
    out = []
    for c in find_contours(np.asarray(phi[:, :, k], float), 0.0):
        xy = np.stack([g.x[0] + c[:, 0]*g.h[0], g.y[0] + c[:, 1]*g.h[1]], 1)
        ij = np.clip(np.round(c).astype(int), 0, np.array(chi.shape) - 1)
        xy[chi[ij[:, 0], ij[:, 1]] > 0] = np.nan
        out.append(xy*1e3)
    return out, float(g.z[k])


def section_contours_2d(section, depth):
    return [c*1e3 for c in section.contours(depth)]


def case_circle_mm(R):
    th = np.linspace(0, 2*np.pi, 181)
    return R*1e3*np.cos(th), R*1e3*np.sin(th)


def split_nan(xy):
    """Split a polyline at NaN rows (for plotting libraries without NaN breaks)."""
    ok = ~np.isnan(xy).any(axis=1)
    out, cur = [], []
    for p, k in zip(xy, ok):
        if k:
            cur.append(p)
        elif cur:
            out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return [o for o in out if len(o) > 1]


def total_mass(res, dry_mass):
    """Motor mass history (kg): dry + oxidizer in the tank + fuel + chamber gas."""
    m_f = res.m_fuel0 - res.m_fuel_burned
    return dry_mass + res['tnk_m_ox'] + m_f + res['cmbr_m_g']


def export_eng(path, res, dry_mass, OD, L, mfg='Burnfront'):
    import burnfront.core as core
    core.export_eng(path, res.t, res['noz_thrust'], total_mass(res, dry_mass), OD=OD, L=L, mfg=mfg)


# --- CFD / thermal helpers --------------------------------------------------------------------------
CFD_GRIDS = {'Coarse (fast)': (60, 18, 70, 20), 'Medium': (90, 27, 105, 30), 'Fine (slow)': (135, 40, 160, 45)}


def nozzle_geometry(v, cfg):
    from burnfront.cfd.nozzle_flow import NozzleGeometry
    chamber_d = cfg.grain_OD if cfg.port.get('type') != 'file' else max(cfg.grain_OD, 2.5*cfg.throat_d)
    return NozzleGeometry.conical(cfg.throat_d, cfg.ER, chamber_d, conv_angle_deg=float(v['conv_angle']),
                                  div_angle_deg=float(v['div_angle']), rcu_ratio=float(v['rcu_ratio']),
                                  rcd_ratio=float(v['rcd_ratio']))


def cfd_conditions(res, cfg, t):
    """Chamber stagnation state at time t of a motor run -> FlowConditions (+ notes). With shifting-
    equilibrium expansion an effective gamma reproduces the engine's exit-pressure ratio."""
    from burnfront.cfd.nozzle_flow import FlowConditions, effective_gamma
    i = int(np.clip(np.searchsorted(res.t, t), 0, res.t.size - 1))
    P0, T0, g, R = (float(res[k][i]) for k in ('cmbr_P', 'cmbr_T', 'cmbr_k', 'cmbr_R'))
    note = f'frozen gamma = {g:.4f}'
    if res.info.get('expansion') == 'equilibrium' and 'noz_Pe' in res.xmap and res['noz_Pe'][i] > 0:
        g_eff = effective_gamma(cfg.ER, float(res['noz_Pe'][i])/P0, g)
        note = f'effective gamma = {g_eff:.4f} (matches the shifting-equilibrium exit pressure; frozen {g:.4f})'
        g = g_eff
    return FlowConditions(P0=P0, T0=T0, gamma=g, Rgas=R, Pa=cfg.Pa), i, note


def render_cfd(grid, F, label, title='', size=(860, 430), dark=True, vmin=None, vmax=None):
    """Ansys-style contour of a cell field F (NI, NJ) mirrored about the axis -> RGBA float32."""
    import matplotlib
    matplotlib.use('Agg', force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    w, h = size
    fig = Figure(figsize=(w/100, h/100), dpi=100)
    FigureCanvasAgg(fig)
    bg = (0.12, 0.12, 0.13) if dark else (1, 1, 1)
    fg = (0.88, 0.88, 0.88) if dark else (0.1, 0.1, 0.1)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.06, 0.12, 0.80, 0.78])
    ax.set_facecolor(bg)
    Fm = np.ma.masked_invalid(F)
    lo = np.nanmin(F) if vmin is None else vmin
    hi = np.nanmax(F) if vmax is None else vmax
    X, R = grid.X*1e3, grid.R*1e3
    for sgn in (1, -1):
        pc = ax.pcolormesh(X, sgn*R, Fm, cmap='turbo', shading='flat', vmin=lo, vmax=hi, rasterized=True)
    # nozzle wall and lip
    xw = X[:grid.ni_noz + 1, grid.nj_noz]
    rw = R[:grid.ni_noz + 1, grid.nj_noz]
    for sgn in (1, -1):
        ax.plot(xw, sgn*rw, color=fg, lw=1.2)
        ax.plot([xw[-1], xw[-1]], [sgn*rw[-1], sgn*R[grid.ni_noz, -1]], color=fg, lw=1.2)
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(-R.max(), R.max())
    ax.set_xlabel('x [mm] (throat at 0)', color=fg); ax.set_ylabel('r [mm]', color=fg)
    ax.tick_params(colors=fg, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(fg)
    cax = fig.add_axes([0.88, 0.12, 0.025, 0.78])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label(label, color=fg); cb.ax.tick_params(colors=fg, labelsize=8)
    if title:
        ax.set_title(title, color=fg, fontsize=10)
    fig.canvas.draw()
    return np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32)/255.0)


def thermal_inputs(res):
    """Chamber histories for the wall heat-transfer model, decimated to <= 400 samples."""
    idx = np.unique(np.linspace(0, res.t.size - 1, min(400, res.t.size)).astype(int))
    return dict(t=res.t[idx], P0=res['cmbr_P'][idx], T0=res['cmbr_T'][idx], gamma=res['cmbr_k'][idx],
                Rgas=res['cmbr_R'][idx], cstar=res['cmbr_cstar'][idx])
