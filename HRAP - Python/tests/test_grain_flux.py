"""Grain regression / oxidizer-flux tests for the JAX (0D) engine.

Key invariant: oxidizer mass flux G is computed against the OPEN port area
(pi/4*OD^2 - grn_A), never the solid fuel area grn_A (the old shiftOF bug).
"""
from importlib.resources import files as imp_files
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import jax.numpy as jnp

import hrap.core as core
from hrap.grain import (make_circle_shape, make_shiftOF_grain, make_constOF_grain,
                        port_area, ABS_REG)

HRAP_ROOT = Path(imp_files('hrap'))


# --- Minimal harness: constant oxidizer source, no chamber/nozzle dynamics ---------------------

def _stub(typename, **x):
    return core.make_part(s={}, x=x, req_s=[], req_x=[], dx={}, typename=typename)

def run_grain(grn, mdot_ox, T, dt, stepper=core.step_rk4):
    tnk  = _stub('tnk', mdot_inj=mdot_ox)
    cmbr = _stub('cmbr', OF=0.0)
    noz  = _stub('noz')
    s, x, method = core.make_engine(tnk, grn, cmbr, noz)
    t, _, xs = core.make_integrator(stepper, method)(s, x, dt=dt, T=T)
    xs = np.asarray(xs)
    return np.asarray(t), (lambda k: xs[:, method['xmap'][k]]), s, method


def circular_port_analytic(t, r0, mdot_ox, a_mm, n, L=1.0, m=0.0):
    """r^(2n+1) = r0^(2n+1) + (2n+1) a' (mdot/pi)^n t, for rdot = a' (mdot/(pi r^2))^n."""
    a = 1e-3 * a_mm * L**m
    return (r0**(2*n+1) + (2*n+1) * a * (mdot_ox/np.pi)**n * t)**(1/(2*n+1))


# --- Tests ---------------------------------------------------------------------------------------

def test_port_area_is_open_area_not_solid():
    OD, ID, L = 0.1, 0.04, 0.5
    grn = make_shiftOF_grain(make_circle_shape(ID=ID), OD=OD, L=L)
    tnk = _stub('tnk', mdot_inj=1.0)
    s, x, method = core.make_engine(tnk, grn, _stub('cmbr', OF=0.0), _stub('noz'))
    for f in method['fpreprs']:
        x = f(s, x, method['xmap'])
    xmap = method['xmap']
    A_solid = float(x[xmap['grn_A']])
    assert A_solid == pytest.approx(np.pi/4*(OD**2 - ID**2))
    assert float(port_area(s, x, xmap)) == pytest.approx(np.pi/4*ID**2)

    for f in method['fderivs']:  # stubs have none, so this is just the grain
        x = f(s, x, xmap)
    G = float(x[xmap['grn_G']])
    assert G == pytest.approx(1.0 / (np.pi/4*ID**2))
    assert G != pytest.approx(1.0 / A_solid)


@pytest.mark.parametrize("Reg", [ABS_REG, (0.12, 0.55, 0.0), (0.2, 0.5, 0.2)])
def test_circular_port_matches_analytic(Reg):
    OD, ID, L, rho, mdot_ox = 0.1143, 0.0635, 0.762, 1070.0, 1.2
    a_mm, n, m = Reg
    grn = make_shiftOF_grain(make_circle_shape(ID=ID), OD=OD, L=L, rho=rho, Reg=jnp.array(Reg))
    t, g, _, _ = run_grain(grn, mdot_ox, T=4.0, dt=2e-3)

    r_sim = ID/2 + g('grn_d')
    r_ref = circular_port_analytic(t, ID/2, mdot_ox, a_mm, n, L, m)
    assert np.max(np.abs(r_sim - r_ref)) / (OD/2 - ID/2) < 1e-6

    # Fuel flow = rho * rdot * perimeter * L, from analytic radius
    rdot_ref = 1e-3*a_mm*L**m * (mdot_ox/(np.pi*r_ref**2))**n
    mdot_f_ref = rho * rdot_ref * 2*np.pi*r_ref * L
    assert np.allclose(g('grn_mdot'), mdot_f_ref, rtol=1e-6)
    # O/F and flux are consistent
    assert np.allclose(g('cmbr_OF'), mdot_ox / mdot_f_ref, rtol=1e-6)
    assert np.allclose(g('grn_G'), mdot_ox / (np.pi*r_ref**2), rtol=1e-6)


def test_flux_decreases_as_port_opens():
    grn = make_shiftOF_grain(make_circle_shape(ID=0.03), OD=0.1, L=0.5)
    t, g, _, _ = run_grain(grn, 0.8, T=3.0, dt=5e-3)
    G = g('grn_G')
    assert np.all(np.diff(G) < 0.0)
    assert np.all(np.diff(g('grn_A_port')) > 0.0)


def test_burnout_no_nan_and_fuel_mass_balance():
    """Burn through the whole web: no NaN, regression stops, fuel mass is conserved (FE exact)."""
    OD, ID, L, rho = 0.05, 0.03, 0.3, 1070.0
    dt = 1e-3
    grn = make_shiftOF_grain(make_circle_shape(ID=ID), OD=OD, L=L, rho=rho, Reg=jnp.array([2.0, 0.5, 0.0]))
    t, g, _, _ = run_grain(grn, 0.5, T=6.0, dt=dt, stepper=core.step_fe)
    assert not np.any(np.isnan(np.c_[g('grn_A'), g('grn_mdot'), g('cmbr_OF'), g('grn_G')]))

    A = g('grn_A')
    assert A[-1] == 0.0                    # web fully consumed
    assert g('grn_mdot')[-1] == 0.0        # no fuel flow after burnout
    assert g('cmbr_OF')[-1] == 0.0         # undefined O/F reported as 0, not NaN/inf

    # Fuel mass: integral of mdot (FE, row i derivative drives step i) == initial solid mass
    burned = np.sum(g('grn_mdot')[:-1]) * dt
    m0 = rho * L * np.pi/4*(OD**2 - ID**2)
    # Last step overshoots A<0 then is clamped, so allow one step of slack
    assert burned == pytest.approx(m0, rel=0, abs=np.max(g('grn_mdot'))*dt)


def test_constOF_reports_open_port_flux():
    OD, ID = 0.1, 0.05
    grn = make_constOF_grain(make_circle_shape(ID=ID), OD=OD, L=0.5, OF=4.0)
    t, g, _, _ = run_grain(grn, 1.0, T=1.0, dt=1e-2)
    A_port = np.pi/4*OD**2 - g('grn_A')
    assert np.allclose(g('grn_A_port'), A_port)
    assert np.allclose(g('grn_G'), 1.0 / A_port)
    assert np.allclose(g('grn_mdot'), 1.0 / 4.0)


def test_full_engine_abs_shiftOF():
    """compare_legacy.py motor (ABS, 2.5" port, 4.5" OD, 30" long) with an 80%-liquid CoolProp tank:
    finite, physical, mass-consistent."""
    import hrap.fluid as fluid
    from hrap.tank import make_sat_tank
    from hrap.chamber import make_chamber
    from hrap.nozzle import make_cd_nozzle
    from hrap.units import _in, _ft
    from jax.scipy.interpolate import RegularGridInterpolator

    c = scipy.io.loadmat(HRAP_ROOT/'resources'/'propellant_configs'/'ABS.mat')['s'][0][0]
    assert tuple(c['prop_Reg'].ravel()) == pytest.approx(ABS_REG)
    grid = (c['prop_OF'].ravel(), c['prop_Pc'].ravel())
    from hrap.chemtab import mat_interpolators
    ik, iM, iT = mat_interpolators(HRAP_ROOT/'resources'/'propellant_configs'/'ABS.mat')

    OD, ID, L, rho = 4.5*_in, 2.5*_in, 30.0*_in, 1070.0
    get_sat = fluid.bake_sat_coolprop('NitrousOxide', np.linspace(183.0, 309.0, 20))
    V_tnk, T0, fill = (np.pi/4*4.75**2*_in**2)*(7.0*_ft), 294.0, 0.8
    sp = get_sat(T0)
    m_ox = fill*V_tnk*float(sp['rho_l']) + (1-fill)*V_tnk*float(sp['rho_v'])
    tnk = make_sat_tank(get_sat, V=V_tnk, inj_CdA=0.22*(np.pi/4*0.5**2*_in**2), m_ox=m_ox, T=T0)
    grn = make_shiftOF_grain(make_circle_shape(ID=ID), OD=OD, L=L, rho=rho)
    s, x, method = core.make_engine(tnk, grn, make_chamber(), make_cd_nozzle(thrt=1.75*_in, ER=4.99, eff=0.97),
                                    chem_interp_k=ik, chem_interp_M=iM, chem_interp_T=iT, Pa=101e3)
    dt = 1e-3
    t, _, xs = core.make_integrator(core.step_fe, method)(s, x, dt=dt, T=12.0)
    xs = np.asarray(xs); g = lambda k: xs[:, method['xmap'][k]]

    assert not np.isnan(xs).any()
    assert np.all(g('noz_thrust') >= 0.0)
    # Flux uses open port area
    on = g('tnk_mdot_inj') > 0
    assert np.allclose(g('grn_G')[on], g('tnk_mdot_inj')[on] / (np.pi/4*OD**2 - g('grn_A')[on]))
    # Fuel mass balance (FE: exact up to roundoff)
    burned = np.sum(g('grn_mdot')[:-1]) * dt
    assert burned == pytest.approx(rho*L*(g('grn_A')[0] - g('grn_A')[-1]), rel=1e-9)
    # Oxidizer mass balance
    assert np.sum(-g('tnk_mdot_ox')[:-1]) * dt == pytest.approx(g('tnk_m_ox')[0] - g('tnk_m_ox')[-1], rel=1e-9)
    # Main burn (injector flow >= half of peak): steady injection and a sensible O/F
    main = g('tnk_mdot_inj') >= 0.5*np.max(g('tnk_mdot_inj'))
    assert main.sum()*dt > 1.0
    assert np.all(g('tnk_mdot_inj')[np.argmax(main):len(main)-np.argmax(main[::-1])] > 0.0)
    assert 1.0 < np.median(g('cmbr_OF')[main]) < 10.0
