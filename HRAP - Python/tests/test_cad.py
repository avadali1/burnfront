"""STEP/STL import: validation, units/axis handling, agreement with analytic grains, grid convergence."""
import numpy as np
import pytest

from burnfront.burn3d import cad, shapes, RegressionLaw, burn_prescribed
from burnfront.burn3d.convergence import celik, grid_convergence

OD, L = 0.08, 0.3
STAR = (6, 0.012, 0.025)


@pytest.fixture(scope='module')
def step_files(tmp_path_factory):
    d = tmp_path_factory.mktemp('cad')
    straight = cad.write_star_grain_step(d/'star.step', OD, L, *STAR, units='m')
    twisted = cad.write_star_grain_step(d/'twist.step', OD, L, *STAR, twist_turns=1.0, units='m')
    return dict(dir=d, straight=straight, twisted=twisted)


def _fuel(g):
    return (g.phi > 0) & (np.broadcast_to(g.chi, g.phi.shape) < 0)


def test_step_straight_star_matches_analytic(step_files):
    g, rep = cad.load_grain(step_files['straight'], n_xy=32, n_z=8, verbose=False)
    assert rep.ok and rep.n_ports == 1
    assert rep.volume == pytest.approx(rep.cad_volume, rel=2e-4)          # tessellation error
    assert rep.OD == pytest.approx(OD, rel=1e-4) and rep.L == pytest.approx(L, rel=1e-9)
    ga = shapes.cylindrical_grain(shapes.star(*STAR), OD, L, n_xy=32, n_z=8)
    h = ga.grid.h[0]
    assert np.max(np.abs(g.phi - ga.phi)[_fuel(ga)]) < 2e-3*h
    assert g.fuel_volume() == pytest.approx(ga.fuel_volume(), rel=2e-4)
    assert np.allclose(g.port_areas(), ga.port_areas(), rtol=1e-5)


def test_step_twisted_star_matches_analytic(step_files):
    g, rep = cad.load_grain(step_files['twisted'], n_xy=32, n_z=24, verbose=False)
    assert rep.ok and rep.n_ports == 1
    ga = shapes.cylindrical_grain(shapes.twist(shapes.star(*STAR), 2*np.pi/L), OD, L, n_xy=32, n_z=24)
    assert np.max(np.abs(g.phi - ga.phi)[_fuel(ga)]) < 0.05*ga.grid.h[0]  # 64-section ruled loft
    law = RegressionLaw(a=3.0)
    rs, ra = burn_prescribed(g, law, 1.0, 0.4), burn_prescribed(ga, law, 1.0, 0.4)
    assert rs.m_burned[-1] == pytest.approx(ra.m_burned[-1], rel=5e-3)
    assert np.max(np.abs(rs.ledger_defect)) < 1e-4


def test_stl_units_and_axis(step_files):
    """Same tessellation exported as an STL in inches along -x must give the same grain."""
    V, F, vol = cad.tessellate_step(step_files['twisted'], 0.01, 0.1)
    import trimesh
    m = trimesh.Trimesh(V/25.4, F, process=True)            # mm -> inches
    R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)  # z -> -x
    m.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    stl = step_files['dir']/'twist_in_xneg.stl'
    m.export(stl)
    g_stl, rep_stl = cad.load_grain(stl, units='in', axis='-x', n_xy=24, n_z=12, verbose=False)
    mesh2, rep2, cls2 = cad.prepare_mesh(V, F, 1e-3, 'z', vol)
    g_ref = cad.grain_from_mesh(mesh2, rep2, cls2, n_xy=24, n_z=12)
    assert rep_stl.ok and rep_stl.OD == pytest.approx(rep2.OD, rel=1e-6)
    # STL stores float32 vertices: allow ~1e-7 relative coordinate noise
    assert np.max(np.abs(g_stl.phi - g_ref.phi)[_fuel(g_ref)]) < 1e-5*OD


def test_non_watertight_mesh_is_rejected(step_files):
    V, F, vol = cad.tessellate_step(step_files['straight'], 0.01, 0.1)
    import trimesh
    m = trimesh.Trimesh(V, F[:-10], process=True)  # remove a few faces -> holes
    stl = step_files['dir']/'holey.stl'
    m.export(stl)
    with pytest.raises(ValueError, match='watertight'):
        cad.load_grain(stl, units='mm', n_xy=16, n_z=4, verbose=False)


def test_multi_port_detection(tmp_path):
    from OCP.gp import gp_Pnt, gp_Ax2, gp_Dir
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    body = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 40.0, 200.0).Shape()
    for cx in (-18.0, 18.0):
        hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, 0, -5), gp_Dir(0, 0, 1)), 8.0, 210.0).Shape()
        body = BRepAlgoAPI_Cut(body, hole).Shape()
    w = STEPControl_Writer()
    w.Transfer(body, STEPControl_AsIs)
    w.Write(str(tmp_path/'two_port.step'))
    g, rep = cad.load_grain(tmp_path/'two_port.step', n_xy=96, n_z=4, verbose=False)
    assert rep.ok and rep.n_ports == 2
    A = g.port_areas()
    # O(h^2) area error at ~10 cells per port radius
    assert A[1] == pytest.approx(2*np.pi*0.008**2, rel=3e-3)


def test_celik_procedure():
    h = [1/np.sqrt(18000), 1/np.sqrt(8000), 1/np.sqrt(4500)]
    a = celik((6.063, 5.972, 5.863), h)   # Celik et al. (2008) worked example, phi_1 column
    assert a['p'] == pytest.approx(1.53, abs=0.01)
    assert a['phi_ext'] == pytest.approx(6.1685, abs=2e-4)
    hh = np.array([0.01, 0.015, 0.02])
    a = celik(1 + 3*hh**2, hh)
    assert a['p'] == pytest.approx(2.0, abs=1e-8) and a['phi_ext'] == pytest.approx(1.0, abs=1e-10)


def test_step_grid_convergence(step_files):
    """Grid-convergence report for a STEP grain. Sharp star corners make the error scatter with grid
    alignment, so the least-squares procedure (>= 4 grids) is the primary estimate; its uncertainty
    must contain the exact initial port volume (star area 900 mm^2 x 300 mm = 270 cm^3)."""
    law = RegressionLaw(a=3.0)
    rep = grid_convergence(lambda n: cad.load_grain(step_files['straight'], n_xy=n, n_z=4, verbose=False)[0],
                           [32, 48, 64, 96], law, 1.0, 0.3)
    print(rep)
    v = rep.ls['V_port0 [cm3]']
    assert abs(rep.values['V_port0 [cm3]'][-1] - 270.0) <= v['U']
    assert v['U_rel'] < 1e-2   # conservative: Fs = 3 when the free-order fit is ill-conditioned
    assert rep.ls['m_burned(T) [g]']['U_rel'] < 1e-2
    assert rep.ls['V_fuel0 [cm3]']['U_rel'] < 5e-3
