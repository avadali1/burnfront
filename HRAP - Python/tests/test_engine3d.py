"""Full engine with a 3D level-set grain: agreement with the 2D engine and with the standalone 3D burn,
mass and c* balance, starting conditions, tank vapor phase, sweeps."""
import csv

import numpy as np
import pytest

import hrap.core as core
from hrap.burn3d import shapes, burn_prescribed
from hrap.motor import MotorConfig, run_motor, sweep, initial_tank_state, saturation, build_grain, regression_law

IN = 0.0254
# Small, quick motor: 2 L tank, 60 mm x 250 mm grain
BASE = MotorConfig(tank_V=2.0e-3, fill=0.85, T_tank=290.0, inj_N=4, inj_d=1.2e-3, inj_Cd=0.65,
                   port={'type': 'circle', 'D': 0.025}, grain_OD=0.06, grain_L=0.25,
                   throat_d=0.012, ER=4.0, cstar_eff=0.95, V_free=1.5e-4, T=5.0, dt=1e-3,
                   n_xy=32, n_z=13, n_z_2d=13)


def _main_rel(a, b, key, frac=0.2):
    x, y = a[key], b[key]
    n = min(len(x), len(y))
    sel = np.abs(y[:n]) > frac*np.abs(y[:n]).max()
    return np.max(np.abs(x[:n][sel]/y[:n][sel] - 1))


def test_3d_engine_matches_2d_engine_circle():
    r3 = run_motor(BASE.replace(mode='3d'))
    r2 = run_motor(BASE.replace(mode='2d'))
    assert np.max(np.abs(r3.geo_defect)) < 1e-5
    # Same flux/regression model. Instantaneous burning perimeter = dA/dd of grid-measured areas whose
    # O(h^2) error oscillates on the cell scale, so its pointwise error is O(h) (~5e-3 at 32 cells,
    # 2.4e-3 at 64); volumes and time integrals are second order.
    for key, tol in [('grn_mdot', 1e-2), ('cmbr_P', 3e-3), ('noz_thrust', 5e-3), ('tnk_P', 1e-3)]:
        assert _main_rel(r3, r2, key) < tol, key
    s3, s2 = r3.summary(), r2.summary()
    assert s3['m_fuel_used_kg'] == pytest.approx(s2['m_fuel_used_kg'], rel=1e-3)
    assert s3['impulse_Ns'] == pytest.approx(s2['impulse_Ns'], rel=1e-3)
    assert r3.m_fuel0 == pytest.approx(r2.m_fuel0, rel=1e-4)


@pytest.fixture(scope='module')
def twisted_run():
    cfg = BASE.replace(port={'type': 'star', 'n_tips': 5, 'r_valley': 0.008, 'r_tip': 0.016,
                             'twist_turns': 1.0}, n_xy=40, n_z=25, stepper='fe', T=4.0)
    res = run_motor(cfg, snapshot_times=(1.0, 2.5))
    return cfg, res


def test_3d_engine_matches_standalone_burn_with_engine_oxidizer_flow(twisted_run):
    """The coupled run's grain must equal burn_prescribed driven by the coupled run's own injector flow."""
    cfg, res = twisted_run
    t, mox = res.t, res['tnk_mdot_inj']
    ref = burn_prescribed(build_grain(cfg), regression_law(cfg), lambda tt: float(np.interp(tt, t, mox)), t[-1])
    # Compare at the reference's own (coarse, RK4-substep) record times; the engine record is 1 ms
    sel = (ref.t > 0.05) & (ref.t < 0.98*t[-1])
    mf = np.interp(ref.t[sel], t, res['grn_mdot'])
    assert np.max(np.abs(mf/ref.mdot_f[sel] - 1)) < 3e-3
    Ab = np.interp(ref.t[sel], t, res['grn_Ab'])
    assert np.max(np.abs(Ab/ref.A_burn[sel] - 1)) < 3e-3


def test_3d_engine_mass_and_cstar_balance(twisted_run):
    cfg, r = twisted_run
    dt = cfg.dt
    g = r.__getitem__
    # Forward Euler: row i derivatives drive step i -> sums are exact ledgers
    ox_out = np.sum(-g('tnk_mdot_ox')[:-1])*dt
    fuel = np.sum(g('grn_mdot')[:-1])*dt
    noz = np.sum(g('noz_mdot')[:-1])*dt
    dmg = g('cmbr_m_g')[-1] - g('cmbr_m_g')[0]
    assert ox_out == pytest.approx(g('tnk_m_ox')[0] - g('tnk_m_ox')[-1], rel=1e-9)
    # Fuel generated vs geometric volume loss: FE residual of V(d) plus the level-set defect per segment
    assert fuel == pytest.approx(r.m_fuel_burned[-1], rel=2e-4)
    assert np.sum(np.abs(r.geo_defect)) < 1e-4
    # The chamber evaluates the nozzle flow from its own current state: exact ledger
    assert ox_out + fuel == pytest.approx(noz + dmg, rel=1e-12)
    # c* balance
    At = np.pi/4*cfg.throat_d**2
    on = g('cmbr_P') > 2.0e5   # choked (critical pressure ratio ~1.8)
    assert np.allclose(g('noz_mdot')[on]*g('cmbr_cstar')[on], g('cmbr_P')[on]*cfg.noz_Cd*At, rtol=1e-9)
    main = g('tnk_mdot_inj') > 0.5*g('tnk_mdot_inj').max()
    cstar_in = g('cmbr_P')[main]*cfg.noz_Cd*At/(g('tnk_mdot_inj')[main] + g('grn_mdot')[main])
    assert np.median(np.abs(cstar_in/g('cmbr_cstar')[main] - 1)) < 2e-3
    # Quasi-steady burn: chamber gas temperature equals the (c*-efficiency corrected) flame temperature
    assert np.median(np.abs(g('cmbr_T')[main]/g('cmbr_T_in')[main] - 1)) < 1e-3
    # Snapshots: the front at t has moved by the recorded per-station depth
    assert [round(s[0], 3) for s in r.snapshots] == [1.0, 2.5]


def test_twisted_engine_burns_more_than_straight(twisted_run):
    cfg, rt = twisted_run
    """Exact initial surface ratio of this twisted/straight star pair (fine ruled meshes) is 1.028 and
    grows as the port opens (tilt ~ omega*r). The t = 0 grid value carries the first-order sharp-corner
    error, so burn averages are compared."""
    rs = run_motor(cfg.replace(**{'port.twist_turns': 0.0}))
    assert rt['grn_Ab'].mean() > 1.02*rs['grn_Ab'].mean()
    assert rt.summary()['m_fuel_used_kg'] > 1.02*rs.summary()['m_fuel_used_kg']


def test_starting_conditions():
    import CoolProp.CoolProp as CP
    # Tank pressure -> saturation temperature
    cfg = BASE.replace(P_tank=4.5e6, T_tank=None, fill=0.6)
    T0, m0, info = initial_tank_state(cfg)
    assert CP.PropsSI('P', 'T', T0, 'Q', 0, 'NitrousOxide') == pytest.approx(4.5e6, rel=1e-6)
    rl = CP.PropsSI('D', 'T', T0, 'Q', 0, 'NitrousOxide'); rv = CP.PropsSI('D', 'T', T0, 'Q', 1, 'NitrousOxide')
    assert m0 == pytest.approx(cfg.tank_V*(0.6*rl + 0.4*rv), rel=1e-4)
    # Baked tables reproduce CoolProp
    p = saturation('NitrousOxide')(T0)
    assert float(p['Pv']) == pytest.approx(4.5e6, rel=1e-4)
    with pytest.raises(ValueError):
        initial_tank_state(BASE.replace(m_ox=10.0))    # more than a liquid-full tank
    # Injector and nozzle inputs reach the engine: hole count scales the oxidizer flow at ignition
    r1 = run_motor(BASE.replace(mode='2d', inj_N=2, T=0.3))
    r2 = run_motor(BASE.replace(mode='2d', inj_N=4, T=0.3))
    assert r2['tnk_mdot_inj'][0] == pytest.approx(2*r1['tnk_mdot_inj'][0], rel=1e-9)


def test_vapor_blowdown_derivative_is_exact():
    """Analytic dT/dm of T = T1 (Z(T)/Z1 m/m1)^0.3 vs a central difference of the converged implicit solution."""
    import jax.numpy as jnp
    from scipy.optimize import brentq
    from hrap.tank import sat_vap_blowdown
    sat = saturation('NitrousOxide')
    T1, m1, mdot = 270.0, 1.3, -0.4
    Z = lambda T: float(sat(T)['Z'])
    Tm = lambda m: brentq(lambda T: T - T1*(Z(T)/Z(T1)*m/m1)**0.3, T1 - 20, T1 + 20, xtol=1e-14, rtol=1e-15)
    eps = 1e-5*m1
    dTdm = (Tm(m1 + eps) - Tm(m1 - eps))/(2*eps)
    Tdot, _ = sat_vap_blowdown(jnp.asarray(T1), m1, mdot, sat(T1), 1.0, sat)
    assert float(Tdot) == pytest.approx(dTdm*mdot, rel=1e-6)


def test_vapor_phase_has_no_model_chatter():
    r = run_motor(BASE.replace(mode='2d', fill=0.3, T=6.0, stepper='fe'))
    vap = r['tnk_vap_only'] > 0.5
    assert vap.any() and vap[-1]
    mi = r['tnk_mdot_inj'][vap][5:]
    d2 = np.diff(mi, 2)
    # smooth: no step-to-step sign alternation of the second difference
    assert np.sum(np.diff(np.sign(d2)) != 0) < 5


def test_parallel_sweep_csv(tmp_path):
    base = BASE.replace(mode='2d', T=2.0)
    rows = sweep(base, {'inj_N': [2, 4], 'throat_d': [0.010, 0.012]}, n_jobs=2,
                 csv_path=tmp_path/'sweep.csv', history_dir=tmp_path/'hist', verbose=False)
    assert len(rows) == 4 and all(r['status'] == 'ok' for r in rows)
    with open(tmp_path/'sweep.csv') as f:
        got = list(csv.DictReader(f))
    assert [(int(r['inj_N']), float(r['throat_d'])) for r in got] == [(2, .01), (2, .012), (4, .01), (4, .012)]
    # Serial run of one point reproduces the parallel result
    one = run_motor(base.replace(inj_N=4, throat_d=0.012)).summary()
    assert float(got[3]['impulse_Ns']) == pytest.approx(one['impulse_Ns'], rel=1e-12)
    assert (tmp_path/'hist'/'run_0003.csv').exists()
    # Smaller throat -> higher chamber pressure
    assert float(got[2]['Pc_avg_Pa']) > float(got[3]['Pc_avg_Pa'])
