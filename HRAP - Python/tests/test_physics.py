"""Physics of the engine components: equilibrium tank (energy conservation), Dyer/HEM injector,
mixture chamber (initial fill, chemistry clamping), nozzle regimes, time-step convergence."""
import numpy as np
import pytest
import jax.numpy as jnp
import CoolProp.CoolProp as CP

from hrap.motor import MotorConfig, run_motor, saturation, hem_table
from hrap.nozzle import mass_flow, gamma_fn, area_ratio

FL = 'NitrousOxide'
BASE = MotorConfig(tank_V=2.0e-3, fill=0.85, T_tank=290.0, inj_N=4, inj_d=1.2e-3, inj_Cd=0.65,
                   port={'type': 'circle', 'D': 0.025}, grain_OD=0.06, grain_L=0.25,
                   throat_d=0.012, ER=4.0, cstar_eff=0.95, V_free=1.5e-4, T=4.0, dt=1e-3, mode='2d')


def test_equilibrium_tank_conserves_energy():
    """U(end) - U(0) = -int mdot_inj h_out dt (adiabatic rigid tank), through liquid run-out."""
    r = run_motor(BASE.replace(fill=0.25, T=4.0, dt=2.5e-4))
    U, t = r['tnk_U'], r.t
    assert r['tnk_vap_only'][-1] > 0.5 and r['tnk_vap_only'][0] < 0.5    # crossed liquid run-out
    out = np.trapezoid(r['tnk_mdot_inj']*r['tnk_h_out'], t)
    assert (U[-1] - U[0] + out)/abs(U[0] - U[-1]) == pytest.approx(0.0, abs=2e-3)
    # Mass: tank loss equals injected flow
    assert np.trapezoid(r['tnk_mdot_inj'], t) == pytest.approx(r['tnk_m_ox'][0] - r['tnk_m_ox'][-1], rel=1e-4)
    # State stays on the saturation line and inside the dome
    assert np.all(r['tnk_m_ox_liq'] >= 0.0)
    assert np.allclose(r['tnk_P'], [CP.PropsSI('P', 'T', T, 'Q', 0, FL) for T in r['tnk_T']], rtol=1e-4)


def test_equilibrium_tank_agrees_with_legacy_while_liquid():
    """With the same incompressible injector the legacy evaporative-cooling model is a close
    approximation of equilibrium in the liquid phase."""
    kw = dict(inj_liq_model='Incompressible', T=3.0)
    a = run_motor(BASE.replace(**kw))
    b = run_motor(BASE.replace(tank_model='legacy', **kw))
    assert a.summary()['impulse_Ns'] == pytest.approx(b.summary()['impulse_Ns'], rel=1e-2)
    # The legacy model cools ~11% less (it lumps the whole contents with the vapor heat capacity);
    # the equilibrium model is the energy-exact one (test above)
    dT_eq, dT_leg = a['tnk_T'][0] - a['tnk_T'][-1], b['tnk_T'][0] - b['tnk_T'][-1]
    assert dT_leg < dT_eq < 1.2*dT_leg


def test_hem_table_matches_coolprop():
    hem = hem_table(FL)
    for T, r in [(250.0, 0.9), (285.0, 0.85), (300.0, 0.95)]:
        Pv = CP.PropsSI('P', 'T', T, 'Q', 0, FL)
        s1, h1 = CP.PropsSI('S', 'T', T, 'Q', 0, FL), CP.PropsSI('H', 'T', T, 'Q', 0, FL)
        P2 = r*Pv
        G = CP.PropsSI('D', 'P', P2, 'S', s1, FL)*np.sqrt(2*(h1 - CP.PropsSI('H', 'P', P2, 'S', s1, FL)))
        assert float(hem(T, P2)) == pytest.approx(G, rel=2e-2)   # bilinear table
    # No flow without a pressure drop; choked plateau at low back pressure
    T = 290.0
    Pv = CP.PropsSI('P', 'T', T, 'Q', 0, FL)
    assert float(hem(T, Pv)) == 0.0
    assert float(hem(T, 0.2*Pv)) == pytest.approx(float(hem(T, 0.4*Pv)), rel=1e-9)
    # HEM is far below SPI for flashing N2O at large pressure drop
    spi = np.sqrt(2*CP.PropsSI('D', 'T', T, 'Q', 0, FL)*0.6*Pv)
    assert float(hem(T, 0.4*Pv)) < 0.6*spi


def test_dyer_is_mean_of_spi_and_hem():
    """Self-pressurized tank: k = sqrt((P1-P2)/(Pv-P2)) = 1, so Dyer = (SPI + HEM)/2 (at ignition Pc = Pa)."""
    m = {k: run_motor(BASE.replace(inj_liq_model=k, T=0.01))['tnk_mdot_inj'][0] for k in ('Incompressible', 'HEM', 'Dyer')}
    assert m['Dyer'] == pytest.approx(0.5*(m['Incompressible'] + m['HEM']), rel=1e-12)


def test_nozzle_regimes():
    g, R, T, Cd, D, ER, Pa = 1.22, 330.0, 3000.0, 0.98, 0.01, 4.0, 101325.0
    At = np.pi/4*D**2
    rc = (2/(g + 1))**(g/(g - 1))
    # Choked flow equals P Cd At / c*; continuity at the choking boundary of the venturi
    P = 3e6
    assert float(mass_flow(P, Pa, T, R, g, Cd, D, ER)) == pytest.approx(Cd*At*P*float(gamma_fn(g))/np.sqrt(R*T), rel=1e-12)
    from scipy.optimize import brentq
    # subsonic exit pressure ratio for area ratio ER
    M_sub = brentq(lambda M: float(area_ratio(M, g)) - ER, 1e-4, 1.0)
    P_b = Pa*(1 + 0.5*(g - 1)*M_sub**2)**(g/(g - 1))
    lo, hi = [float(mass_flow(P_b*f, Pa, T, R, g, Cd, D, ER)) for f in (1 - 1e-7, 1 + 1e-7)]
    assert lo == pytest.approx(hi, rel=1e-5)
    assert float(mass_flow(0.99*Pa, Pa, T, R, g, Cd, D, ER)) == 0.0


def _thrust(P, sep=0.4, ER=4.0, Pa=101325.0):
    import hrap.core as core
    from hrap.nozzle import make_cd_nozzle, d_cd_nozzle
    noz = make_cd_nozzle(thrt=0.01, ER=ER, eff=1.0, Cd=1.0, sep_ratio=sep)
    keys = ['cmbr_P', 'cmbr_k', 'cmbr_R', 'cmbr_T', 'noz_mdot', 'noz_Me', 'noz_Pe', 'noz_thrust', 'noz_separated']
    xmap = {k: i for i, k in enumerate(keys)}
    x = jnp.zeros(len(keys)).at[0].set(P).at[1].set(1.22).at[2].set(330.0).at[3].set(3000.0)
    s = {**noz['s'], 'Pa': Pa}
    x = d_cd_nozzle(s, x, xmap)
    return {k: float(x[xmap[k]]) for k in keys}


def test_nozzle_thrust_attached_equals_ideal_cf_and_separation_is_continuous():
    g, Pa, ER = 1.22, 101325.0, 4.0
    At = np.pi/4*0.01**2
    o = _thrust(3e6)
    pe = o['noz_Pe']/3e6
    Cf = np.sqrt(2*g**2/(g - 1)*(2/(g + 1))**((g + 1)/(g - 1))*(1 - pe**((g - 1)/g))) + (pe - Pa/3e6)*ER
    assert o['noz_thrust'] == pytest.approx(Cf*At*3e6, rel=1e-10) and o['noz_separated'] == 0.0
    # Chamber pressure at which the attached exit pressure equals 0.4 Pa: attached and separated thrust agree
    from scipy.optimize import brentq
    from hrap.nozzle import M_solve
    Me = float(M_solve(g, ER))
    P_sep = 0.4*Pa*(1 + 0.5*(g - 1)*Me**2)**(g/(g - 1))
    a, b = _thrust(P_sep*(1 + 1e-9)), _thrust(P_sep*(1 - 1e-9))
    assert a['noz_separated'] == 0.0 and b['noz_separated'] == 1.0
    assert a['noz_thrust'] == pytest.approx(b['noz_thrust'], rel=1e-6)
    # Separation raises thrust compared with an attached over-expanded nozzle (no sub-ambient wall pressure)
    assert _thrust(0.6*P_sep)['noz_thrust'] > _thrust(0.6*P_sep, sep=0.0)['noz_thrust']


def test_chamber_initial_fill_and_chemistry_clamp():
    r = run_motor(BASE.replace(T=0.05))
    i = 0
    V_gas = r['cmbr_V0'][i] - r['grn_V'][i]
    assert V_gas == pytest.approx(BASE.V_free + BASE.grain_L*np.pi*0.0125**2, rel=2e-3)   # free + port only
    assert r['cmbr_P'][i] == BASE.Pa and r['cmbr_m_air'][i] == r['cmbr_m_g'][i]
    assert r['cmbr_T'][i] == pytest.approx(294.0, rel=1e-12)
    # A very lean motor (tiny fuel port, big oxidizer flow) leaves the table's O/F range -> flagged
    lean = run_motor(BASE.replace(port={'type': 'circle', 'D': 0.004}, inj_N=12, T=0.3))
    s = lean.summary()
    assert np.max(lean['cmbr_OF']) > 10.0 and s['chem_OF_clamped_s'] > 0.1


def test_auto_dt_converged():
    a = run_motor(BASE.replace(dt=None))
    b = run_motor(BASE.replace(dt=a.info['dt']/2))
    assert a.info['dt'] < 1e-3
    for k in ('impulse_Ns', 'm_fuel_used_kg', 'Pc_peak_Pa'):
        assert a.summary()[k] == pytest.approx(b.summary()[k], rel=1e-4), k
