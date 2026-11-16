"""Verification of the numpy/scipy 3D level-set burn core (burnfront.burn3d)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from burnfront.burn3d import shapes, measure, flow, meshdist, RegressionLaw, burn_prescribed
from burnfront.burn3d.law import ABS_A, ABS_N, ABS_RHO


# --- measure ------------------------------------------------------------------------------------

def test_triangle_fraction_exact_for_linear_fields():
    rng = np.random.default_rng(0)
    # Linear psi = g.x + c on the unit right triangle; reference by dense sampling
    u = rng.uniform(size=(400_000, 2))
    u = u[u.sum(1) <= 1]
    for _ in range(20):
        g, c = rng.normal(size=2), rng.normal()*0.5
        vals = np.array([c, g[0] + c, g[1] + c])  # at (0,0), (1,0), (0,1)
        f = measure.tri_fraction_neg(*vals)
        ref = np.mean(u @ g + c < 0)
        assert f == pytest.approx(ref, abs=5e-3)


def test_slice_area_second_order():
    errs = []
    for n in (32, 64, 128):
        g = shapes.cylindrical_grain(shapes.circle(0.02), 0.1, 0.1, n_xy=n, n_z=2)
        errs.append(abs(g.port_areas()[0]/(np.pi*0.02**2) - 1))
        assert abs(g.A_env[0]/(np.pi*0.05**2) - 1) < 1e-3
    rates = np.log2(np.array(errs[:-1])/np.array(errs[1:]))
    assert np.all(rates > 1.8), (errs, rates)


# --- exact mesh distance ------------------------------------------------------------------------

def test_mesh_distance_is_exact():
    shp = shapes.twist(shapes.star(5, 0.01, 0.02), 10.0)
    V, F = shapes.surface_mesh(shp, 0.0, 0.1, 0.002, 0.002)
    md = meshdist.MeshDistance(V, F)
    rng = np.random.default_rng(2)
    pts = rng.uniform([-0.04, -0.04, -0.01], [0.04, 0.04, 0.11], (400, 3))
    d = md.unsigned(pts)
    bf = np.sqrt(np.min(meshdist.closest_point_triangles(pts[:, None, :], md.a[None], md.b[None], md.c[None]), axis=1))
    assert np.max(np.abs(d - bf)) < 1e-15


# --- axial flow model ---------------------------------------------------------------------------

def test_axial_flux_closed_form_matches_ode():
    L, k, n, rho, mox = 0.5, 2e-4, 0.62, 1070.0, 0.8
    A = lambda z: 1e-3*(1 + 0.5*np.sin(7*z) + z)
    P = lambda z: 0.15*(1 + 0.3*np.cos(5*z))
    ref = solve_ivp(lambda z, m: rho*k*P(z)*(m/A(z))**n, [0, L], [mox], rtol=1e-12, atol=1e-14, dense_output=True)
    errs = []
    for nz in (21, 41, 81):
        z = np.linspace(0, L, nz)
        _, _, mz, _ = flow.station_rates(A(z), P(z), mox, z, k, n, rho, measure.trapz_weights(z), 'total')
        errs.append(np.max(np.abs(mz/ref.sol(z)[0] - 1)))
    assert errs[1] < 1e-4
    assert np.all(np.log2(np.array(errs[:-1])/np.array(errs[1:])) > 1.9)


def test_no_oxidizer_no_burning():
    z = np.linspace(0, 1, 5)
    rdot, G, mz, mf = flow.station_rates(np.full(5, 1e-3), np.full(5, 0.1), 0.0, z, 1e-4, 0.5, 1000.0,
                                         measure.trapz_weights(z), 'total')
    assert np.all(rdot == 0) and mf == 0


# --- straight cylinder: analytic, existing 2D (JAX) model, total-flux reference ------------------

OD, R0 = 0.1, 0.02

def _r_analytic(t, L, law, mox):
    k, n = law.k(L), law.n
    return (R0**(2*n+1) + (2*n+1)*k*(mox/np.pi)**n*t)**(1/(2*n+1))


def test_cylinder_matches_analytic_second_order():
    L, mox = 0.3, 1.0
    law = RegressionLaw(flux='oxidizer')
    errs = []
    for nxy in (32, 64):
        g = shapes.cylindrical_grain(shapes.circle(R0), OD, L, n_xy=nxy, n_z=5)
        res = burn_prescribed(g, law, mox, 6.0)
        r = np.sqrt(res.A_port_z[:, 2]/np.pi)
        errs.append(np.max(np.abs(r - _r_analytic(res.t, L, law, mox)))/(OD/2 - R0))
        assert np.max(np.abs(res.ledger_defect)) < 1e-4
    assert errs[1] < 5e-4
    assert errs[0]/errs[1] > 3.5


def test_cylinder_burnthrough_time_and_sliver():
    L, mox = 0.3, 1.0
    law = RegressionLaw(a=3.0, flux='oxidizer')
    k, n = law.k(L), law.n
    t_bt = ((OD/2)**(2*n+1) - R0**(2*n+1)) / ((2*n+1)*k*(mox/np.pi)**n)
    g = shapes.cylindrical_grain(shapes.circle(R0), OD, L, n_xy=64, n_z=5)
    res = burn_prescribed(g, law, mox, 1.2*t_bt)
    assert res.t_burnthrough == pytest.approx(t_bt, rel=3e-4)
    assert abs(res.m_sliver) < 1e-9*res.m_fuel0      # concentric port: no sliver
    assert res.m_sliver_burnthrough < 1e-3*res.m_fuel0
    assert res.mdot_f[-1] == 0.0


def test_cylinder_3d_matches_existing_2d_jax_model():
    """Same circular port through the (fixed) JAX shiftOF engine with constant oxidizer flow."""
    import jax.numpy as jnp
    import burnfront.core as core
    from burnfront.grain import make_circle_shape, make_shiftOF_grain

    L, mox, T, dt = 0.3, 1.0, 4.0, 1e-3
    grn = make_shiftOF_grain(make_circle_shape(ID=2*R0), OD=OD, L=L, rho=ABS_RHO,
                             Reg=jnp.array([ABS_A, ABS_N, 0.0]))
    stub = lambda typ, **x: core.make_part(s={}, x=x, req_s=[], req_x=[], dx={}, typename=typ)
    s, x, method = core.make_engine(stub('tnk', mdot_inj=mox), grn, stub('cmbr', OF=0.0), stub('noz'))
    t2d, _, xs = core.make_integrator(core.step_rk4, method)(s, x, dt=dt, T=T)
    xs = np.asarray(xs); t2d = np.asarray(t2d)
    mf2d = xs[:, method['xmap']['grn_mdot']]
    r2d = R0 + xs[:, method['xmap']['grn_d']]

    g = shapes.cylindrical_grain(shapes.circle(R0), OD, L, n_xy=64, n_z=5)
    res = burn_prescribed(g, RegressionLaw(flux='oxidizer'), mox, T)
    r3d = np.sqrt(res.A_port_z[:, 2]/np.pi)
    assert np.max(np.abs(r3d - np.interp(res.t, t2d, r2d)))/(OD/2 - R0) < 5e-4
    assert np.max(np.abs(res.mdot_f/np.interp(res.t, t2d, mf2d) - 1)) < 2e-3


def test_cylinder_total_flux_matches_method_of_lines():
    L, mox, T = 0.6, 1.0, 0.8
    law = RegressionLaw(a=3.0, flux='total')
    k = law.k(L)
    zf = np.linspace(0, L, 2001)
    wf = measure.trapz_weights(zf)
    rhs = lambda t, r: flow.station_rates(np.pi*r**2, 2*np.pi*r, mox, zf, k, law.n, law.rho, wf, 'total')[0]
    mol = solve_ivp(rhs, [0, T], np.full(zf.size, R0), rtol=1e-11, atol=1e-13, dense_output=True)

    g = shapes.cylindrical_grain(shapes.circle(R0), OD, L, n_xy=48, n_z=17)
    res = burn_prescribed(g, law, mox, T)
    r3d = np.sqrt(res.A_port_z/np.pi)
    rref = np.stack([np.interp(res.z, zf, mol.sol(t)) for t in res.t])
    assert np.max(np.abs(r3d - rref))/(OD/2 - R0) < 1e-3
    mfref = np.array([flow.station_rates(np.pi*mol.sol(t)**2, 2*np.pi*mol.sol(t), mox, zf, k, law.n, law.rho, wf, 'total')[3]
                      for t in res.t])
    assert np.max(np.abs(res.mdot_f/mfref - 1)) < 3e-3
    # Head end regresses slower than the aft end (fuel added upstream raises the flux)
    assert res.depth_z[-1, 0] < res.depth_z[-1, -1]


# --- twisted star -------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def star_pair():
    ODs, L = 0.08, 0.3
    base = shapes.star(6, 0.012, 0.025)
    tw = shapes.twist(base, 2*np.pi/L)
    law = RegressionLaw(a=3.0)  # total flux; fast law keeps the test short
    out = {}
    for name, shp in (('straight', base), ('twisted', tw)):
        g = shapes.cylindrical_grain(shp, ODs, L, n_xy=48, n_z=36)
        out[name] = (g, burn_prescribed(g, law, 1.0, 0.6))
    return out


def test_twisted_star_conserves_mass(star_pair):
    g, res = star_pair['twisted']
    # Fuel: ODE ledger vs level-set volume
    assert np.max(np.abs(res.ledger_defect)) < 1e-4
    # Oxidizer enters at the head end unchanged; exit flow = oxidizer + generated fuel
    assert np.allclose(res.mdot_z[:, 0], res.mdot_ox)
    assert np.max(np.abs(res.mdot_z[:, -1]/(res.mdot_ox + res.mdot_f) - 1)) < 1e-3
    # Fuel consumed never exceeds the initial load; remaining fuel is non-negative
    assert res.m_burned[-1] <= res.m_fuel0 and res.m_sliver >= 0


def test_twisted_star_has_more_burning_surface(star_pair):
    gs, rs = star_pair['straight']
    gt, rt = star_pair['twisted']
    # Same cross-section -> same initial port volume
    assert rt.V_port[0] == pytest.approx(rs.V_port[0], rel=5e-3)
    # More surface at t=0 (reference: fine ruled-mesh area) and throughout the burn before burn-through
    V, F = shapes.surface_mesh(shapes.twist(shapes.star(6, 0.012, 0.025), 2*np.pi/0.3), 0, 0.3, 2e-4, 2e-4)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A_ref = 0.5*np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert rt.A_burn[0] == pytest.approx(A_ref, rel=5e-3)
    assert rt.A_burn[0] > 1.05*rs.A_burn[0]
    tb = min(rs.t_burnthrough, rt.t_burnthrough)
    ts = np.linspace(0, 0.95*tb, 20)
    assert np.all(np.interp(ts, rt.t, rt.A_burn) > np.interp(ts, rs.t, rs.A_burn))
    assert rt.m_burned[-1] > rs.m_burned[-1]


def test_pde_option_accurate_for_smooth_fronts():
    L, mox = 0.3, 1.0
    law = RegressionLaw(flux='oxidizer')
    g = shapes.cylindrical_grain(shapes.circle(R0), OD, L, n_xy=32, n_z=5)
    res = burn_prescribed(g, law, mox, 6.0, method='pde')
    r = np.sqrt(res.A_port_z[:, 2]/np.pi)
    assert np.max(np.abs(r - _r_analytic(res.t, L, law, mox)))/(OD/2 - R0) < 2e-3
