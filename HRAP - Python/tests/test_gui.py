"""GUI: input conversion/validation, surface extraction, and a headless end-to-end run of the app."""
import time

import numpy as np
import pytest

from burnfront.gui import model as gm
from burnfront.burn3d import shapes


def test_values_to_config_units_and_ports():
    v = gm.defaults()
    v.update(port_type='Star', star_n=5, star_r_valley=8.0, star_r_tip=16.0, twist_turns=1.0, tank_V=2.5,
             fill=70.0, init_by='Pressure', P_tank=45.0, inj_d=1.2, V_free=150.0, auto_dt=False, dt=0.5)
    cfg, snaps = gm.values_to_config(v)
    assert cfg.port == {'type': 'star', 'n_tips': 5, 'r_valley': 0.008, 'r_tip': 0.016, 'rotation': 0.0, 'twist_turns': 1.0}
    assert cfg.tank_V == pytest.approx(2.5e-3) and cfg.fill == pytest.approx(0.7)
    assert cfg.P_tank == pytest.approx(45e5) and cfg.T_tank is None
    assert cfg.inj_d == pytest.approx(1.2e-3) and cfg.V_free == pytest.approx(1.5e-4) and cfg.dt == pytest.approx(5e-4)
    assert snaps == [0, 1, 2, 4, 6]
    v2 = dict(v, port_type='STEP/STL file', file_path='', mode=gm.MODE_3D)
    with pytest.raises(ValueError, match='STEP'):
        gm.values_to_config(v2)
    with pytest.raises(ValueError, match='3D'):
        gm.values_to_config(dict(v2, file_path='g.step', mode=gm.MODE_2D))
    with pytest.raises(ValueError, match='tip radius'):
        gm.values_to_config(dict(v, star_r_tip=5.0))


def test_save_load_roundtrip(tmp_path):
    v = gm.defaults()
    v['star_n'] = 7
    gm.save_values(tmp_path/'c.json', v)
    assert gm.load_values(tmp_path/'c.json') == v


def test_port_surface_matches_port():
    """The extracted port surface of a cylindrical port has the right radius and area."""
    g = shapes.cylindrical_grain(shapes.circle(0.015), 0.06, 0.2, n_xy=48, n_z=21)
    V, F, burning = gm.port_surface(g, g.phi)
    r = np.hypot(V[:, 0], V[:, 1])
    lateral = (V[:, 2] > 1.0) & (V[:, 2] < 199.0)
    assert np.median(r[lateral]) == pytest.approx(15.0, rel=5e-3)
    assert burning.mean() > 0.95
    tri = V[F]
    area = 0.5*np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    exact = 2*np.pi*15*200 + 2*np.pi*15**2   # lateral + capped ends
    assert area == pytest.approx(exact, rel=3e-2)
    img = gm.render_surface(V, F, burning, 30.0, 200.0, size=(320, 240))
    assert img.shape == (240, 320, 4) and img[..., :3].std() > 0.01


def test_app_headless_end_to_end(tmp_path):
    dpg = pytest.importorskip('dearpygui.dearpygui')
    from burnfront.gui.app import App, _tag
    dpg.create_context()
    try:
        dpg.create_viewport(title='test', width=1200, height=800)
        app = App()
        app.build()
        dpg.setup_dearpygui()

        def run_and_wait():
            app.run()
            t0 = time.time()
            while app.busy and time.time() - t0 < 600:
                app.frame(); time.sleep(0.05)
            app.frame()

        small = dict(T=0.6, tank_V=1.0, grain_L=150.0, grain_OD=50.0, star_r_valley=7.0, star_r_tip=12.0,
                     n_xy=24, n_z=9, n_xy_2d=96, n_z_2d=7, snapshots='0, 0.5')
        for k, val in small.items():
            dpg.set_value(_tag(k), val)
        dpg.set_value(_tag('mode'), gm.MODE_2D)
        app.update_visibility()
        assert not dpg.is_item_shown('row_n_xy') and dpg.is_item_shown('row_n_xy_2d')
        run_and_wait()
        assert app.res is not None and app.res.mode == '2d', dpg.get_value('status')
        assert len(dpg.get_item_children('summary_table', 1)) == 18
        assert len(dpg.get_item_children('xs_y', 1)) >= 3

        dpg.set_value(_tag('mode'), gm.MODE_3D)
        dpg.set_value(_tag('twist_turns'), 0.5)
        app.update_visibility()
        run_and_wait()
        assert app.res.mode == '3d', dpg.get_value('status')
        assert dpg.get_item_configuration('snap_combo')['items'] == ['t = 0.00 s', 't = 0.50 s']
        app.render()
        app.export_csv(None, {'file_path_name': str(tmp_path/'h.csv')})
        app.export_eng(None, {'file_path_name': str(tmp_path/'m.eng')})
        assert (tmp_path/'h.csv').exists() and (tmp_path/'m.eng').exists()
        # Input errors are reported, not raised
        dpg.set_value(_tag('star_r_tip'), 1.0)
        app.run()
        assert 'Input error' in dpg.get_value('status') and not app.busy
    finally:
        dpg.destroy_context()


def test_app_cad_button_cfd_and_thermal(tmp_path, monkeypatch):
    dpg = pytest.importorskip('dearpygui.dearpygui')
    from burnfront.gui.app import App, _tag
    from burnfront.burn3d import cad
    monkeypatch.setitem(gm.CFD_GRIDS, 'Coarse (fast)', (30, 10, 30, 10))
    step = cad.write_star_grain_step(tmp_path/'g.step', 0.05, 0.15, 5, 0.007, 0.012, units='m')
    dpg.create_context()
    try:
        dpg.create_viewport(title='test', width=1200, height=800)
        app = App()
        app.build()
        dpg.setup_dearpygui()

        def wait(flag):
            t0 = time.time()
            while flag() and time.time() - t0 < 600:
                app.frame(); time.sleep(0.05)
            app.frame()

        # Add-file button path: switches to 3D/CAD and validates the mesh
        app.add_cad_file(None, {'file_path_name': str(step)})
        assert dpg.get_value(_tag('port_type')) == 'STEP/STL file' and dpg.get_value(_tag('mode')) == gm.MODE_3D
        wait(lambda: dpg.get_value('cad_info').startswith('Checking'))
        assert 'watertight' in dpg.get_value('cad_info') and 'OD 50.0 mm' in dpg.get_value('cad_info')
        # CFD and thermal need a motor run first
        app.run_cfd()
        assert 'Run the motor first' in dpg.get_value('cfd_status')
        for k, val in dict(T=0.6, tank_V=1.0, n_xy=20, n_z=7, snapshots='0', throat_d=8.0).items():
            dpg.set_value(_tag(k), val)
        app.run()
        wait(lambda: app.busy)
        assert app.res is not None and app.res.mode == '3d', dpg.get_value('status')
        dpg.set_value('cfd_t', 0.4)
        app.run_cfd()
        wait(lambda: app.cfd_busy)
        r = app.cfd['result']
        assert r is not None, dpg.get_value('cfd_status')
        assert abs(r.summary['mass_imbalance']) < 1e-5
        assert len(dpg.get_item_children('cfd_table', 1)) == 13
        app.render_cfd()
        app.run_thermal()
        wait(lambda: app.thermal_busy)
        w, _ = app.thermal
        assert w.Tw.max() > 400.0 and 'peak inner-wall temperature' in dpg.get_value('th_status')
    finally:
        dpg.destroy_context()
