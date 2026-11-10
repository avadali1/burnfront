"""2D (cross-section) mode: consistency with the analytic solution, the 3D core, and the JAX engine;
helical approximation; legacy baking functions; full-engine mass and c* balance."""
import contextlib
import io
from importlib.resources import files as imp_files
from pathlib import Path

import numpy as np
import pytest
import scipy.io

import hrap.core as core
from hrap.burn3d import shapes, RegressionLaw, burn_prescribed
from hrap.burn3d.section import Section2D, burn_section
from hrap.grain_axial import make_section_grain

OD, R0 = 0.1, 0.02
HRAP_ROOT = Path(imp_files('hrap'))


def _stub(typ, **x):
    return core.make_part(s={}, x=x, req_s=[], req_x=[], dx={}, typename=typ)


def test_circle_section_matches_analytic():
    L, mox = 0.3, 1.0
    law = RegressionLaw(a=3.0, flux='oxidizer')
    k, n = law.k(L), law.n
    c = (2*n + 1)*k*(mox/np.pi)**n
    t_bt = ((OD/2)**(2*n + 1) - R0**(2*n + 1))/c
    tab = Section2D(shapes.circle(R0), OD, n_xy=64).table()
    res = burn_section(tab, L, 3, law, mox, 1.1*t_bt, dt=2e-3)
    pre = res.t < 0.98*t_bt
    r = np.sqrt(res.A_port_z[:, 1]/np.pi)
    assert np.max(np.abs(r - (R0**(2*n + 1) + c*res.t)**(1/(2*n + 1)))[pre])/(OD/2 - R0) < 4e-4
    assert res.t_burnthrough == pytest.approx(t_bt, rel=2e-4)
    assert abs(res.m_sliver) < 1e-9*res.m_fuel0


@pytest.fixture(scope='module')
def stars():
    ODs, L, T = 0.08, 0.3, 0.6
    base = shapes.star(6, 0.012, 0.025)
    om = 2*np.pi/L
    law = RegressionLaw(a=3.0)
    r3s = burn_prescribed(shapes.cylindrical_grain(base, ODs, L, n_xy=48, n_z=25), law, 1.0, T)
    r3t = burn_prescribed(shapes.cylindrical_grain(shapes.twist(base, om), ODs, L, n_xy=48, n_z=37), law, 1.0, T)
    sec = Section2D(base, ODs, n_xy=256)
    tab = sec.table()
    kap = sec.helical_kappa(tab, twist_rate=om)
    r2 = burn_section(tab, L, 25, law, 1.0, T, dt=2e-3)
    r2h = burn_section(tab, L, 25, law, 1.0, T, dt=2e-3, kappa=kap)
    return dict(r3s=r3s, r3t=r3t, r2=r2, r2h=r2h, kap=kap, L=L, om=om)


def _maxrel(a, b, key, ts):
    return np.max(np.abs(np.interp(ts, a.t, getattr(a, key))/np.interp(ts, b.t, getattr(b, key)) - 1))


def test_straight_star_2d_equals_3d(stars):
    ts = np.linspace(0.05, 0.45, 30)
    for key in ('mdot_f', 'm_burned', 'A_burn'):
        assert _maxrel(stars['r2'], stars['r3s'], key, ts) < 1e-2, key
    assert stars['r2'].t_burnthrough == pytest.approx(stars['r3s'].t_burnthrough, rel=2e-3)


def test_helical_correction_approximates_twisted_3d(stars):
    ts = np.linspace(0.05, 0.42, 30)
    plain = _maxrel(stars['r2'], stars['r3t'], 'm_burned', ts)
    helical = _maxrel(stars['r2h'], stars['r3t'], 'm_burned', ts)
    assert plain > 0.04
    assert helical < 0.015
    assert helical < plain/4
    # kappa(0) is exact: ratio of twisted to straight surface area per unit length (fine meshes)
    def area(shp):
        V, F = shapes.surface_mesh(shp, 0, stars['L'], 2e-4, 2e-4)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return 0.5*np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    base = shapes.star(6, 0.012, 0.025)
    assert stars['kap'][0] == pytest.approx(area(shapes.twist(base, stars['om']))/area(base), rel=3e-3)
    assert np.all(stars['kap'] >= 1.0)


@pytest.mark.parametrize('twist', [0.0, 2*np.pi/0.3])
def test_jax_axial_grain_matches_numpy(twist):
    L, T, dt, mox = 0.3, 0.6, 1e-3, 1.0
    law = RegressionLaw(a=3.0)
    grn, sec, tab, kap = make_section_grain(shapes.star(6, 0.012, 0.025), 0.08, L, n_z=11, law=law, twist_rate=twist)
    s, x, method = core.make_engine(_stub('tnk', mdot_inj=mox), grn, _stub('cmbr', OF=0.0), _stub('noz'))
    t, _, xs = core.make_integrator(core.step_rk4, method)(s, x, dt=dt, T=T)
    xs, t = np.asarray(xs), np.asarray(t)
    g = lambda k: xs[:, method['xmap'][k]]
    ref = burn_section(tab, L, 11, law, mox, T, dt, kappa=kap)
    n = min(len(ref.t), len(t))
    assert np.max(np.abs(g('grn_mdot')[:n]/ref.mdot_f[:n] - 1)) < 1e-5
    assert np.max(np.abs(g('grn_Ab')[:n]/ref.A_burn[:n] - 1)) < 1e-5
    # Fuel ledger: integral of generation == rho * volume change (trapezoid in time, RK4 states)
    assert np.trapezoid(g('grn_mdot'), t) == pytest.approx(law.rho*(g('grn_V')[0] - g('grn_V')[-1]), rel=1e-5)


def test_bake_d2a_circle_consistent():
    from hrap.grain import bake_d2a
    t = np.linspace(0, 2*np.pi, 720, endpoint=False)
    poly = np.stack([R0*np.cos(t), R0*np.sin(t)], 1)
    A0, d, P, _ = bake_d2a(OD, poly, 256, 200)
    web = OD/2 - R0
    assert A0 == pytest.approx(np.pi/4*OD**2 - np.pi*R0**2, rel=1e-4)
    inner = d < 0.95*web
    assert np.max(np.abs(P[inner]/(2*np.pi*(R0 + d[inner])) - 1)) < 2e-3
    # A0 - integral of perimeter over the web = 0 (burnout). P jumps 2 pi R -> 0 at burnout, so the
    # trapezoid check needs fine sampling (its error is ~ P_max * dd / 2)
    A0f, df, Pf, _ = bake_d2a(OD, poly, 256, 4000)
    assert np.trapezoid(Pf, df) == pytest.approx(A0f, rel=5e-4)


def test_image_baking_subpixel_accuracy(tmp_path):
    import cv2
    from hrap.grain import bake_arbitrary_d2a
    S, N = 4, 640
    yy, xx = (np.mgrid[0:N*S, 0:N*S] + 0.5)/S
    r = np.hypot(xx - 320, yy - 320)
    img = np.where(r <= 300, 128.0, 255.0)
    img = np.where(r <= 150, 255.0, img).reshape(N, S, N, S).mean(axis=(1, 3))
    path = tmp_path/'circle.png'
    cv2.imwrite(str(path), np.round(img).astype(np.uint8))
    D = 0.06
    r0 = D/2*150/300
    with contextlib.redirect_stdout(io.StringIO()):
        d, P, contours, A0 = bake_arbitrary_d2a(path, D, n_step=60, n_visu=5)
    assert A0 == pytest.approx(np.pi/4*D**2 - np.pi*r0**2, rel=2e-3)
    assert P[0] == pytest.approx(2*np.pi*r0, rel=2e-3)
    assert d[-1] == pytest.approx(D/2 - r0, rel=5e-3)


def test_full_engine_2d_mode_mass_and_cstar_balance():
    """ABS 2D-mode grain (6-point star) with an 80%-liquid N2O tank, chamber and nozzle."""
    import hrap.fluid as fluid
    from hrap.tank import make_sat_tank
    from hrap.chamber import make_chamber
    from hrap.nozzle import make_cd_nozzle
    from hrap.units import _in, _ft
    from jax.scipy.interpolate import RegularGridInterpolator

    c = scipy.io.loadmat(HRAP_ROOT/'resources'/'propellant_configs'/'ABS.mat')['s'][0][0]
    grid = (c['prop_OF'].ravel(), c['prop_Pc'].ravel())
    from hrap.chemtab import mat_interpolators
    ik, iM, iT = mat_interpolators(HRAP_ROOT/'resources'/'propellant_configs'/'ABS.mat')
    law = RegressionLaw.from_mat(HRAP_ROOT/'resources'/'propellant_configs'/'ABS.mat')
    get_sat = fluid.bake_sat_coolprop('NitrousOxide', np.linspace(183.0, 309.0, 20))
    V_tnk, T0 = (np.pi/4*4.75**2*_in**2)*(7.0*_ft), 294.0
    sp = get_sat(T0)
    m_ox = 0.8*V_tnk*float(sp['rho_l']) + 0.2*V_tnk*float(sp['rho_v'])
    tnk = make_sat_tank(get_sat, V=V_tnk, inj_CdA=0.22*(np.pi/4*0.5**2*_in**2), m_ox=m_ox, T=T0)
    ODg, L = 4.5*_in, 30*_in
    grn, *_ = make_section_grain(shapes.star(5, 1.0*_in, 1.5*_in), ODg, L, n_z=11, law=law, n_xy=128)
    Cd, thrt = 0.995, 1.75*_in
    s, x, method = core.make_engine(tnk, grn, make_chamber(cstar_eff=0.95),
                                    make_cd_nozzle(thrt=thrt, ER=4.99, eff=0.97, Cd=Cd),
                                    chem_interp_k=ik, chem_interp_M=iM, chem_interp_T=iT, Pa=101e3)
    dt = 1e-3
    t, _, xs = core.make_integrator(core.step_fe, method)(s, x, dt=dt, T=10.0)
    xs = np.asarray(xs)
    g = lambda k: xs[:, method['xmap'][k]]
    assert not np.isnan(xs).any()

    # Forward Euler: row i derivatives drive step i -> sums are exact ledgers
    ox_out = np.sum(-g('tnk_mdot_ox')[:-1])*dt
    fuel = np.sum(g('grn_mdot')[:-1])*dt
    noz = np.sum(g('noz_mdot')[:-1])*dt
    dmg = g('cmbr_m_g')[-1] - g('cmbr_m_g')[0]
    assert ox_out == pytest.approx(g('tnk_m_ox')[0] - g('tnk_m_ox')[-1], rel=1e-9)
    # Fuel volume is a function of the integrated depths, V(d): forward Euler leaves an O(dt) residual
    assert fuel == pytest.approx(law.rho*(g('grn_V')[0] - g('grn_V')[-1]), rel=1e-4)
    # Chamber: injected oxidizer + fuel = nozzle outflow + stored gas
    # The chamber evaluates the nozzle flow from its own current state: exact ledger
    assert ox_out + fuel == pytest.approx(noz + dmg, rel=1e-12)

    # c* balance: nozzle flow obeys mdot = Pc Cd At / c*, and in the quasi-steady main burn the
    # incoming propellant flow matches it (chamber filling is a small correction)
    At = np.pi/4*thrt**2
    on = g('cmbr_P') > 2.0e5   # choked (critical pressure ratio ~1.8)
    assert np.allclose(g('noz_mdot')[on]*g('cmbr_cstar')[on], g('cmbr_P')[on]*Cd*At, rtol=1e-9)
    main = g('tnk_mdot_inj') > 0.5*g('tnk_mdot_inj').max()
    cstar_in = g('cmbr_P')[main]*Cd*At/(g('tnk_mdot_inj')[main] + g('grn_mdot')[main])
    assert np.median(np.abs(cstar_in/g('cmbr_cstar')[main] - 1)) < 2e-3
    # Quasi-steady burn: chamber gas temperature equals the (c*-efficiency corrected) flame temperature
    assert np.median(np.abs(g('cmbr_T')[main]/g('cmbr_T_in')[main] - 1)) < 1e-3
    assert 1200 < np.median(g('cmbr_cstar')[main]) < 1800
