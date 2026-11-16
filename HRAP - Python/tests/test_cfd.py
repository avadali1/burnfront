"""Nozzle/plume CFD (axisymmetric Euler) and nozzle wall heat transfer."""
import numpy as np
import pytest
from scipy.special import erf

from burnfront.cfd import nozzle_flow as nf
from burnfront.cfd.thermal import ShellConduction, bartz, isentropic_mach, wall_transient

GEOM = nf.NozzleGeometry.conical(0.02, 4.0, 0.07)       # rc_up/Rt = 1.5, 15 deg cone
COARSE = (60, 18, 70, 20)


def kliegel_levine_cd(g, R):
    """Kliegel & Levine (1969) discharge coefficient of a throat with wall curvature ratio R = rc/rt."""
    return 1 - (g + 1)/(R + 1)**2*(1/96 - (8*g - 27)/(2304*(R + 1)) + (754*g*g - 757*g + 3633)/(276480*(R + 1)**2))


@pytest.fixture(scope='module')
def attached():
    fc = nf.FlowConditions(P0=30e5, T0=3000.0, gamma=1.2, Rgas=330.0, Pa=101325.0)
    return nf.run(nf.make_grid(GEOM, *COARSE), fc, 4.0)


def test_geometry_is_smooth_and_matches_design():
    x = np.linspace(GEOM.x_start, GEOM.x_exit, 20001)
    R = GEOM.radius(x)
    assert R.min() == pytest.approx(GEOM.Rt, rel=1e-6) and abs(x[np.argmin(R)]) < 1e-5
    assert R[-1] == pytest.approx(GEOM.Re, rel=1e-6) and R[0] == pytest.approx(GEOM.Rc)
    assert np.max(np.abs(np.diff(R))) < 2e-5          # continuous contour


def test_attached_nozzle_conservation_and_discharge_coefficient(attached):
    s = attached.summary
    assert s['converged'] and not s['shock_in_nozzle']
    assert abs(s['mass_imbalance']) < 1e-6                     # exactly conservative fluxes
    # 2D throat discharge coefficient vs the analytic transonic solution (coarse grid)
    assert s['Cd_2d'] == pytest.approx(kliegel_levine_cd(1.2, 1.5), abs=2e-3)


def test_attached_nozzle_thrust_and_exit_state(attached):
    s = attached.summary
    # Inviscid conical nozzle: between the ideal 1D value and the divergence-factor estimate
    assert s['Cf_1d_divergence']*0.99 < s['Cf_cfd'] < s['Cf_1d']
    assert s['Me_avg'] == pytest.approx(s['Me_1d'], rel=3e-2)
    assert s['pe_avg_bar'] == pytest.approx(s['pe_1d_bar'], rel=6e-2)


def test_overexpanded_nozzle_has_internal_normal_shock():
    fc = nf.FlowConditions(P0=2e5, T0=3000.0, gamma=1.2, Rgas=330.0, Pa=101325.0)
    r = nf.run(nf.make_grid(GEOM, *COARSE), fc, 4.0, max_iter=4000, avg_iter=200)
    assert r.summary['shock_in_nozzle']
    assert r.summary['Me_avg'] < 1.0


def test_effective_gamma_reproduces_exit_pressure():
    from scipy.optimize import brentq
    g = nf.effective_gamma(5.0, 0.03)

    def ar(M):
        return (1/M)*((2/(g + 1))*(1 + 0.5*(g - 1)*M*M))**((g + 1)/(2*(g - 1)))
    Me = brentq(lambda M: ar(M) - 5.0, 1, 50)
    assert (1 + 0.5*(g - 1)*Me**2)**(-g/(g - 1)) == pytest.approx(0.03, rel=1e-8)


def test_conduction_matches_semi_infinite_solution():
    k, rho, cp = 40.0, 7800.0, 500.0
    a = k/(rho*cp)
    w = ShellConduction(np.array([50.0]), 0.05, k, rho, cp, n_r=201, T_init=300.0)
    t, dt = 0.0, 0.01
    while t < 5.0 - 1e-9:
        w.step(dt, np.array([1e9]), np.array([1300.0]), 0.0, 300.0)
        t += dt
    x = np.linspace(0, 0.05, 201)
    exact = 1300 + (300 - 1300)*erf(x/(2*np.sqrt(a*t)))
    assert np.max(np.abs(w.T[0] - exact)) < 0.2         # K, out of a 1000 K step


def test_conduction_energy_balance():
    """Heat entering through the inner surface equals the stored energy (adiabatic outside)."""
    k, rho, cp = 80.0, 1850.0, 1500.0
    w = ShellConduction(np.array([0.01]), 0.01, k, rho, cp, n_r=41, T_init=300.0)
    E0 = np.sum(w.C*w.T)
    Q, dt, h, Tg = 0.0, 0.01, 5e3, 3000.0
    for _ in range(300):
        Ts0 = w.T[0, 0]
        w.step(dt, np.array([h]), np.array([Tg]), 0.0, 300.0)
        Q += h*w.r[0, 0]*(Tg - 0.5*(Ts0 + w.T[0, 0]))*dt       # trapezoidal surface flux (per radian)
    assert np.sum(w.C*w.T) - E0 == pytest.approx(Q, rel=1e-3)


def test_bartz_scaling_and_throat_peak():
    g, R, T0 = 1.2, 330.0, 3000.0
    cs = np.sqrt(R*T0)/(np.sqrt(g)*(2/(g + 1))**((g + 1)/(2*(g - 1))))
    h1, _ = bartz(30e5, T0, g, R, cs, 0.02, 0.0188, 1.0, 1.0, 600.0)
    h2, _ = bartz(60e5, T0, g, R, cs, 0.02, 0.0188, 1.0, 1.0, 600.0)
    assert h2/h1 == pytest.approx(2**0.8, rel=1e-12)
    assert 5e3 < h1 < 3e4                                   # typical hybrid throat values at 30 bar
    x = np.linspace(GEOM.x_start, GEOM.x_exit, 400)
    eps = (GEOM.radius(x)/GEOM.Rt)**2
    M = isentropic_mach(eps, g, x > 0)
    h, Taw = bartz(30e5, T0, g, R, cs, 0.02, 0.5*(GEOM.rcu + GEOM.rcd), eps, M, 600.0)
    q = h*(Taw - 600.0)
    assert abs(x[np.argmax(q)]) < 0.15*GEOM.Rt              # peak heat flux at the throat


def test_wall_transient_heats_throat_first():
    t = np.linspace(0, 3.0, 61)
    w = wall_transient(GEOM, t, np.full_like(t, 30e5), np.full_like(t, 3000.0), np.full_like(t, 1.2),
                       np.full_like(t, 330.0), np.full_like(t, 1500.0), material='Copper', thickness=0.008,
                       n_x=30, n_r=16)
    j_hot = int(np.argmax(w.Tw[-1]))
    assert abs(w.x[j_hot]) < 0.2*GEOM.Rc
    assert np.all(np.diff(w.Tw[:, j_hot]) >= -1e-9)          # monotone heating at constant conditions
    assert np.isfinite(w.time_to_limit())                     # copper melts without cooling at 30 bar
