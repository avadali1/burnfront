"""Shifting-equilibrium nozzle expansion: agreement with NASA CEA (RocketCEA) and with HRAP's
CEA-generated propellant tables, nozzle/engine consistency."""
import numpy as np
import pytest

import hrap.expansion as ex
from hrap.chemtab import load_mat_tables
from hrap.motor import PROP_DIR, MotorConfig, run_motor, expansion_table

G0, PSI = 9.80665, 6894.757


@pytest.fixture(scope='module')
def eq():
    return ex.Equilibrium()


def _perf(c, ER, Pc, Pa):
    lnER = np.log(ER)
    re = np.interp(lnER, c['LE'], c['r_t']*ex.RHO_GRID)
    Ue = np.interp(lnER, c['LE'], c['U'])
    return c['cstar']*(Ue + re*ER)/G0, c['cstar']*(Ue + (re - Pa/Pc)*ER)/G0


@pytest.mark.parametrize('fuel,OF,frozen', [('Paraffin', 1.5, 0), ('ABS', 5.0, 0), ('HTPB', 6.5, 0),
                                            ('HDPE', 3.0, 0), ('ABS', 6.5, 1), ('Paraffin', 8.0, 1)])
def test_matches_nasa_cea(eq, fuel, OF, frozen):
    rc = pytest.importorskip('rocketcea.cea_obj')
    rc.add_new_oxidizer('N2OL_hrap', f'oxid N2O(L) N 2 O 1 wt%=100 h,cal={65100/4.184:.4f} t(k)=298.15')
    comp, hf = ex.REACTANTS[fuel]
    card = ' '.join(f'{e} {n}' for e, n in comp.items())
    rc.add_new_fuel(f'{fuel}_hrap', f'fuel {fuel} {card} wt%=100 h,cal={hf/4.184:.4f} t(k)=298.15')
    cea = rc.CEA_Obj(oxName='N2OL_hrap', fuelName=f'{fuel}_hrap')
    Pc, ER, Pa = 2.5e6, 5.0, 101325.0
    c = ex.nozzle_curves(eq, ex.FUELS[fuel], OF, Pc, shifting=not frozen)
    Iv, Ia = _perf(c, ER, Pc, Pa)
    assert c['T0'] == pytest.approx(cea.get_Tcomb(Pc/PSI, OF)/1.8, rel=1e-3)
    if not frozen:   # CEA reports the equilibrium c* only
        assert c['cstar'] == pytest.approx(cea.get_Cstar(Pc/PSI, OF)*0.3048, rel=1e-3)
    assert Iv == pytest.approx(cea.get_Isp(Pc/PSI, OF, ER, frozen=frozen), rel=1.5e-3)
    assert Ia == pytest.approx(cea.estimate_Ambient_Isp(Pc/PSI, OF, ER, Pa/PSI, frozen=frozen)[0], rel=1.5e-3)


def test_chamber_matches_hrap_cea_tables_with_condensed_carbon(eq):
    """Paraffin .mat (CEA): temperature and M = mixture mass per kmol gas, including fuel-rich points
    where graphite condenses."""
    OF, Pc, k, M, T = load_mat_tables(PROP_DIR/'Paraffin.mat')
    j = int(np.argmin(abs(Pc - 2e6)))
    for of in (1.0, 2.0, 4.0, 7.0):
        i = int(np.argmin(abs(OF - of)))
        elem, h = ex.reactant_mixture(ex.FUELS['Paraffin'], OF[i])
        st = eq.hp(h, Pc[j], elem)
        assert st['T'] == pytest.approx(T[i, j], rel=2e-3), of
        assert st['M'] == pytest.approx(M[i, j], rel=2e-3), of
        if of <= 2.0:
            assert st['nc'] > 0.0


def test_nozzle_thrust_at_table_node():
    """Equilibrium nozzle at a table node reproduces the table's thrust coefficient."""
    import jax.numpy as jnp
    from hrap.nozzle import make_cd_nozzle, d_cd_nozzle, expansion_kwargs
    tab = expansion_table('ABS')
    i, j = int(np.argmin(abs(tab.OF - 6.0))), int(np.argmin(abs(tab.Pc - 2.5e6)))
    OF, Pc = float(tab.OF[i]), float(tab.Pc[j])
    noz = make_cd_nozzle(thrt=0.02, ER=4.0, eff=1.0, Cd=1.0, **expansion_kwargs(tab))
    keys = ['cmbr_P', 'cmbr_k', 'cmbr_R', 'cmbr_T', 'cmbr_OF', 'cmbr_cstar', 'cmbr_cstar_fac', 'cmbr_y_burn',
            'noz_mdot', 'noz_Me', 'noz_Pe', 'noz_thrust', 'noz_separated']
    xm = {k: n for n, k in enumerate(keys)}
    g, R, T = float(tab.gamma_f[i, j]), ex.RU/float(tab.M0[i, j]), float(tab.T0[i, j])
    fc = float(tab.cstar[i, j]/tab.cstar_gamma[i, j])
    x = jnp.zeros(len(keys))
    for k, v in dict(cmbr_P=Pc, cmbr_k=g, cmbr_R=R, cmbr_T=T, cmbr_OF=OF, cmbr_cstar=float(tab.cstar[i, j]),
                     cmbr_cstar_fac=fc, cmbr_y_burn=1.0).items():
        x = x.at[xm[k]].set(v)
    x = d_cd_nozzle({**noz['s'], 'Pa': 101325.0}, x, xm)
    At = np.pi/4*0.02**2
    assert float(x[xm['noz_mdot']]) == pytest.approx(Pc*At/float(tab.cstar[i, j]), rel=1e-10)
    lnER = np.log(4.0)
    re = np.exp(np.interp(lnER, tab.LE[i, j], np.log(tab.r_t[i, j]*tab.rho)))   # log-log, as the engine
    Cf = np.interp(lnER, tab.LE[i, j], tab.U[i, j]) + (re - 101325.0/Pc)*4.0
    assert float(x[xm['noz_thrust']]) == pytest.approx(Cf*Pc*At, rel=1e-9)


def test_engine_equilibrium_vs_frozen():
    base = MotorConfig(tank_V=2.0e-3, fill=0.85, T_tank=290.0, inj_N=4, inj_d=1.2e-3, inj_Cd=0.65,
                       port={'type': 'circle', 'D': 0.025}, grain_OD=0.06, grain_L=0.25, throat_d=0.012,
                       ER=5.0, cstar_eff=0.95, V_free=1.5e-4, T=3.0, dt=5e-4, mode='2d', stepper='fe')
    re = run_motor(base)
    rf = run_motor(base.replace(noz_expansion='frozen'))
    assert re.info['expansion'] == 'equilibrium' and rf.info['expansion'] == 'frozen'
    se, sf = re.summary(), rf.summary()
    assert 1.0 < se['Isp_s']/sf['Isp_s'] < 1.06
    # Exact chamber ledger and choked-flow c* balance with the equilibrium c*
    g, dt = re.__getitem__, base.dt
    ox = np.sum(-g('tnk_mdot_ox')[:-1])*dt; fu = np.sum(g('grn_mdot')[:-1])*dt
    noz = np.sum(g('noz_mdot')[:-1])*dt; dmg = g('cmbr_m_g')[-1] - g('cmbr_m_g')[0]
    assert ox + fu == pytest.approx(noz + dmg, rel=1e-12)
    on = g('cmbr_P') > 2e5
    At = np.pi/4*base.throat_d**2
    assert np.allclose(g('noz_mdot')[on]*g('cmbr_cstar')[on], g('cmbr_P')[on]*base.noz_Cd*At, rtol=1e-9)
    main = g('cmbr_y_burn') > 0.99
    assert np.all(g('cmbr_cstar_fac')[main] > 1.0)       # shifting equilibrium raises c*
    # Unsupported propellant falls back to frozen with a note
    rm = run_motor(base.replace(propellant='Metalized_Plastisol', T=0.2))
    assert rm.info['expansion'] == 'frozen' and 'frozen' in rm.info['notes']
