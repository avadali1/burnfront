"""Burnfront motor GUI: 3D level-set grains (STEP/STL or analytic) or the 2D outline model, coupled to the
tank, injector, chamber and nozzle models of burnfront.motor.

Run with ``burnfront`` (or ``python -m burnfront.gui.app``). The legacy HRAP GUI is ``burnfront-legacy``.
"""
import os
import queue
import threading
import time
import traceback
from importlib.resources import files as imp_files
from pathlib import Path

import numpy as np
import dearpygui.dearpygui as dpg

from burnfront.gui import model as gm

RENDER_W, RENDER_H = 860, 560
CFD_W, CFD_H = 900, 420
CFD_FIELDS = ['Mach', 'Pressure [bar]', 'Temperature [K]', 'Density [kg/m3]', 'Velocity [m/s]']
LEFT_W = 450

# Sweepable fields: GUI key -> (MotorConfig name, GUI->SI scale)
SWEEP_MAP = {
    'tank_V': ('tank_V', 1e-3), 'fill': ('fill', 1e-2), 'T_tank': ('T_tank', 1.0), 'P_tank': ('P_tank', 1e5),
    'inj_N': ('inj_N', 1), 'inj_d': ('inj_d', 1e-3), 'inj_Cd': ('inj_Cd', 1.0),
    'grain_OD': ('grain_OD', 1e-3), 'grain_L': ('grain_L', 1e-3), 'port_D': ('port.D', 1e-3),
    'star_r_valley': ('port.r_valley', 1e-3), 'star_r_tip': ('port.r_tip', 1e-3),
    'twist_turns': ('port.twist_turns', 1.0), 'phi_swirl': ('phi_swirl', 1.0),
    'V_free': ('V_free', 1e-6), 'cstar_eff': ('cstar_eff', 1.0), 'throat_d': ('throat_d', 1e-3),
    'ER': ('ER', 1.0), 'noz_eff': ('noz_eff', 1.0), 'Pa': ('Pa', 1e3), 'tank_wall_mass': ('tank_wall_mass', 1.0),
}

SUMMARY_ROWS = [
    ('impulse_Ns', 'Total impulse', 'N s', 1.0),
    ('burn_time_s', 'Burn time (F > 5% peak)', 's', 1.0),
    ('thrust_avg_N', 'Average thrust', 'N', 1.0),
    ('thrust_peak_N', 'Peak thrust', 'N', 1.0),
    ('Isp_s', 'Delivered Isp', 's', 1.0),
    ('Pc_avg_Pa', 'Average chamber pressure', 'bar', 1e-5),
    ('Pc_peak_Pa', 'Peak chamber pressure', 'bar', 1e-5),
    ('m_ox_used_kg', 'Oxidizer used', 'kg', 1.0),
    ('m_fuel_used_kg', 'Fuel used', 'kg', 1.0),
    ('OF_mean', 'Mean O/F', '', 1.0),
    ('OF_final', 'Final O/F', '', 1.0),
    ('G_head_peak', 'Peak head-end flux', 'kg/m2/s', 1.0),
    ('fuel_remaining_kg', 'Fuel remaining (sliver)', 'kg', 1.0),
    ('t_burnthrough_s', 'Burn-through time', 's', 1.0),
    ('tank_P_final_Pa', 'Final tank pressure', 'bar', 1e-5),
    ('chem_OF_clamped_s', 'Time O/F outside chem table', 's', 1.0),
    ('chem_Pc_clamped_s', 'Time Pc outside chem table', 's', 1.0),
    ('nozzle_separated_s', 'Time with nozzle flow separation', 's', 1.0),
]


def _tag(key):
    return f'in_{key}'


class App:
    def __init__(self):
        self.res = None
        self.cfg = None
        self.q = queue.Queue()
        self.busy = False
        self.t_start = 0.0
        self.render_due = False
        self.last_change = 0.0
        self.sweep_rows = []
        # CFD / thermal / CAD-check background jobs (independent of motor runs)
        self.aux_q = queue.Queue()
        self.cfd = None
        self.cfd_busy = False
        self.cfd_stop = False
        self.cfd_render_t = 0.0
        self.cfd_dirty = False
        self.thermal = None
        self.thermal_busy = False

    # ---------------------------------------------------------------------------------- inputs
    def values(self):
        v = {}
        for f in gm.FIELDS:
            v[f.key] = dpg.get_value(_tag(f.key))
        return v

    def set_values(self, v):
        for f in gm.FIELDS:
            if f.key in v and v[f.key] is not None:
                val = v[f.key]
                if f.kind == 'float':
                    val = float(val)
                elif f.kind == 'int':
                    val = int(val)
                elif f.kind == 'bool':
                    val = bool(val)
                else:
                    val = str(val)
                dpg.set_value(_tag(f.key), val)
        self.update_visibility()

    def update_visibility(self, *_):
        v = self.values()
        for f in gm.FIELDS:
            dpg.configure_item(f'row_{f.key}', show=gm.visible(f, v))
        is3d = v['mode'] == gm.MODE_3D
        dpg.configure_item('view3d_group', show=is3d)
        dpg.configure_item('view3d_note', show=not is3d)
        dpg.configure_item('xs_z_slider', show=is3d)
        dpg.configure_item('xs_station', show=not is3d)

    def _file_dialog(self, key, exts):
        def cb(sender, app_data):
            dpg.set_value(_tag(key), app_data['file_path_name'])
        tag = f'dlg_{key}'
        with dpg.file_dialog(tag=tag, show=False, width=720, height=420, callback=cb, directory_selector=False):
            for e in exts:
                dpg.add_file_extension(e)
        return tag

    def build_inputs(self):
        for section in gm.SECTIONS:
            with dpg.collapsing_header(label=section, default_open=section in ('Tank', 'Injector', 'Grain')):
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp, borders_innerH=True):
                    dpg.add_table_column(init_width_or_weight=1.25)
                    dpg.add_table_column(init_width_or_weight=1.4)
                    dpg.add_table_column(init_width_or_weight=0.45)
                    for f in [f for f in gm.FIELDS if f.section == section]:
                        with dpg.table_row(tag=f'row_{f.key}'):
                            t = dpg.add_text(f.label)
                            if f.tip:
                                with dpg.tooltip(t):
                                    dpg.add_text(f.tip)
                            cb = self.on_change
                            if f.kind == 'float':
                                dpg.add_input_float(tag=_tag(f.key), default_value=float(f.default), width=-1,
                                                    format='%.6g', step=0, callback=cb)
                            elif f.kind == 'int':
                                dpg.add_input_int(tag=_tag(f.key), default_value=int(f.default), width=-1,
                                                  step=1, callback=cb)
                            elif f.kind == 'combo':
                                dpg.add_combo(f.items, tag=_tag(f.key), default_value=f.default, width=-1, callback=cb)
                            elif f.kind == 'bool':
                                dpg.add_checkbox(tag=_tag(f.key), default_value=bool(f.default), callback=cb)
                            elif f.kind == 'text':
                                dpg.add_input_text(tag=_tag(f.key), default_value=str(f.default), width=-1, callback=cb)
                            elif f.kind == 'file':
                                exts = ['.step', '.stp', '.stl', '.*'] if f.key == 'file_path' else ['.png', '.jpg', '.bmp', '.*']
                                dlg = self._file_dialog(f.key, exts)
                                with dpg.group(horizontal=True):
                                    dpg.add_input_text(tag=_tag(f.key), default_value='', width=-40, callback=cb)
                                    dpg.add_button(label='...', callback=lambda s, a, u: dpg.show_item(u), user_data=dlg)
                            dpg.add_text(f.unit)

    def on_change(self, *_):
        self.update_visibility()

    # ---------------------------------------------------------------------------------- CAD file
    def add_cad_file(self, sender, app_data):
        path = app_data['file_path_name']
        dpg.set_value(_tag('mode'), gm.MODE_3D)
        dpg.set_value(_tag('port_type'), 'STEP/STL file')
        dpg.set_value(_tag('file_path'), path)
        self.update_visibility()
        dpg.set_value('cad_info', f'Checking {Path(path).name} ...')
        units, axis = dpg.get_value(_tag('file_units')), dpg.get_value(_tag('file_axis'))

        def work():
            try:
                from burnfront.burn3d import cad
                _, r = cad.load_grain(path, units=units, axis=axis, n_xy=24, n_z=6, verbose=False)
                verr = (r.volume - r.cad_volume)/r.cad_volume if np.isfinite(r.cad_volume) else float('nan')
                txt = (f'{Path(path).name}: watertight mesh, OD {r.OD*1e3:.1f} mm, length {r.L*1e3:.1f} mm, '
                       f'{r.n_ports} port(s), initial burning area {r.port_area*1e4:.1f} cm2'
                       + (f', tessellation volume error {verr:+.1e}' if np.isfinite(verr) else '')
                       + '. Units/axis can be changed below; click Run.')
                self.aux_q.put(('cad', txt, False))
            except Exception as e:
                self.aux_q.put(('cad', f'{Path(path).name}: {e}', True))
        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------------------------------- running
    def run(self, *_):
        if self.busy:
            return
        try:
            cfg, snaps = gm.values_to_config(self.values())
        except Exception as e:
            self.status(f'Input error: {e}', error=True)
            return
        if cfg.mode == '3d' and not snaps:
            snaps = [0.0]
        self.busy = True
        self.t_start = time.time()
        dpg.configure_item('run_btn', enabled=False)

        def work():
            try:
                res = __import__('burnfront.motor', fromlist=['run_motor']).run_motor(cfg, snapshot_times=tuple(snaps))
                self.q.put(('done', (cfg, res)))
            except Exception as e:
                self.q.put(('error', f'{type(e).__name__}: {e}\n{traceback.format_exc()}'))
        threading.Thread(target=work, daemon=True).start()

    def status(self, msg, error=False):
        dpg.set_value('status', msg)
        dpg.configure_item('status', color=(255, 110, 110) if error else (190, 220, 190))

    def poll(self):
        if self.busy:
            dpg.set_value('status', f'Running... {time.time() - self.t_start:.0f} s')
        try:
            kind, payload = self.q.get_nowait()
        except queue.Empty:
            return
        self.busy = False
        dpg.configure_item('run_btn', enabled=True)
        dpg.configure_item('sweep_btn', enabled=True)
        if kind == 'done':
            self.cfg, self.res = payload
            self.status(f"Done in {time.time() - self.t_start:.1f} s  (dt = {self.res.info['dt']*1e3:.3g} ms, "
                        f"{self.cfg.mode.upper()})")
            self.show_result()
        elif kind == 'sweep':
            self.sweep_rows = payload
            self.status(f'Sweep finished: {len(payload)} runs in {time.time() - self.t_start:.0f} s')
            self.show_sweep()
        else:
            self.status('Run failed (details in the Summary tab)', error=True)
            dpg.set_value('summary_text', payload)

    # ---------------------------------------------------------------------------------- results
    def show_result(self):
        r = self.res
        t = r.t
        step = max(1, t.size // 4000)
        ts = np.ascontiguousarray(t[::step])

        def series(tag, y):
            dpg.set_value(tag, [ts, np.ascontiguousarray(np.asarray(y)[::step])])

        series('s_thrust', r['noz_thrust'])
        series('s_pc', r['cmbr_P']*1e-5)
        series('s_pt', r['tnk_P']*1e-5)
        of = np.where(r['grn_mdot'] > 0, r['cmbr_OF'], np.nan)
        series('s_of', np.nan_to_num(of, nan=0.0))
        G = r.station('grn_G')
        nz = G.shape[1]
        for name, j in [('head', 0), ('mid', nz//2), ('aft', nz - 1)]:
            series(f's_G_{name}', G[:, j])
            dpg.configure_item(f's_G_{name}', label=f'{name} (z = {r.z[j]*1e3:.0f} mm)')
        series('s_mdot_ox', r['tnk_mdot_inj'])
        series('s_mdot_f', r['grn_mdot'])
        for p in ['p_thrust', 'p_press', 'p_of', 'p_flux', 'p_mdot']:
            for ax in dpg.get_item_children(p, 1):
                if dpg.get_item_type(ax).endswith('mvPlotAxis'):
                    dpg.fit_axis_data(ax)
        # Flux map G(z, t)
        nt = min(400, t.size)
        idx = np.linspace(0, t.size - 1, nt).astype(int)
        Gm = G[idx].T[::-1]     # rows: z from aft (top) to head (bottom)
        gmax = float(np.max(Gm)) if Gm.size else 1.0
        dpg.delete_item('fluxmap_y', children_only=True)
        dpg.add_heat_series(np.ascontiguousarray(Gm.ravel()), rows=Gm.shape[0], cols=Gm.shape[1],
                            scale_min=0.0, scale_max=max(gmax, 1e-9), format='',
                            bounds_min=(float(t[0]), float(r.z[0]*1e3)), bounds_max=(float(t[-1]), float(r.z[-1]*1e3)),
                            parent='fluxmap_y')
        dpg.configure_item('fluxmap_scale', min_scale=0.0, max_scale=max(gmax, 1e-9))
        dpg.fit_axis_data('fluxmap_x'); dpg.fit_axis_data('fluxmap_y')

        # Summary
        s = r.summary()
        dpg.delete_item('summary_table', children_only=True, slot=1)
        for key, label, unit, sc in SUMMARY_ROWS:
            with dpg.table_row(parent='summary_table'):
                dpg.add_text(label)
                val = s.get(key, np.nan)
                dpg.add_text('-' if val is None or (isinstance(val, float) and np.isnan(val)) else f'{val*sc:.5g}')
                dpg.add_text(unit)
        info = r.info
        lines = [f"Nozzle expansion: {info.get('expansion', 'frozen')}" + (f"  ({info['notes']})" if info.get('notes') else ''),
                 f"Initial tank: T = {info['T0']:.2f} K, P = {info['P0']/1e5:.2f} bar, oxidizer {info['m_ox0']:.4g} kg",
                 f"Initial fuel: {r.m_fuel0:.4g} kg;  time step {info['dt']*1e3:.3g} ms;  wall time {info['wall_time']:.1f} s"]
        if r.mode == '3d':
            lines.append(f"3D grain: grid {r.grain.grid.shape}, {info['n_segments']} grain steps, "
                         f"level-set volume defect max {np.max(np.abs(r.geo_defect)) if len(r.geo_defect) else 0:.1e}")
            rep = getattr(r.grain, 'report', None)
            if rep is not None:
                lines.append('Mesh report:\n' + str(rep))
        dpg.set_value('summary_text', '\n'.join(lines))

        # Grain views
        if r.mode == '3d':
            items = [f't = {ts_:.2f} s' for ts_, _ in r.snapshots]
            dpg.configure_item('snap_combo', items=items)
            if items:
                dpg.set_value('snap_combo', items[-1])
            self.render_due = True
        self.draw_sections()

    def _snapshot(self):
        r = self.res
        if r is None or r.mode != '3d' or not r.snapshots:
            return None
        items = [f't = {ts_:.2f} s' for ts_, _ in r.snapshots]
        sel = dpg.get_value('snap_combo')
        i = items.index(sel) if sel in items else len(items) - 1
        return r.snapshots[i]

    def render(self):
        snap = self._snapshot()
        if snap is None:
            return
        ts, phi = snap
        g = self.res.grain
        V, F, burning = gm.port_surface(g, phi)
        R = float(np.sqrt(np.max(g.A_env)/np.pi))
        img = gm.render_surface(V, F, burning, R*1e3, g.L*1e3, z0_mm=g.grid.z[0]*1e3,
                                az=dpg.get_value('az'), el=dpg.get_value('el'), size=(RENDER_W, RENDER_H),
                                title=f'Port surface at t = {ts:.2f} s  (orange: burning front, grey: exposed case)')
        dpg.set_value('surf_tex', img.ravel())

    def draw_sections(self, *_):
        r = self.res
        if r is None:
            return
        dpg.delete_item('xs_y', children_only=True)
        R = None
        if r.mode == '3d':
            g = r.grain
            R = float(np.sqrt(np.max(g.A_env)/np.pi))
            zf = dpg.get_value('xs_z_slider')
            for ts, phi in r.snapshots:
                cs, z = gm.section_contours_3d(g, phi, zf)
                first = True
                for c in cs:
                    for seg in gm.split_nan(c):
                        dpg.add_line_series(np.ascontiguousarray(seg[:, 0]), np.ascontiguousarray(seg[:, 1]),
                                            label=f't = {ts:.2f} s' if first else '', parent='xs_y')
                        first = False
            dpg.configure_item('xs_plot', label=f'Cross-section at z = {z*1e3:.1f} mm')
        else:
            tab, kap, sec = r.grain
            if sec is None:
                return
            R = sec.R
            j = int(np.clip(dpg.get_value('xs_station'), 0, r.z.size - 1))
            try:
                times = gm.parse_times(dpg.get_value(_tag('snapshots'))) or [0.0]
            except ValueError:
                times = [0.0]
            for ts in times:
                d = float(np.interp(ts, r.t, r.depth_z[:, j]))
                first = True
                for c in gm.section_contours_2d(sec, d):
                    for seg in gm.split_nan(c):
                        dpg.add_line_series(np.ascontiguousarray(seg[:, 0]), np.ascontiguousarray(seg[:, 1]),
                                            label=f't = {ts:.2f} s' if first else '', parent='xs_y')
                        first = False
            dpg.configure_item('xs_plot', label=f'Cross-section, station {j} (z = {r.z[j]*1e3:.1f} mm)')
        if R is not None:
            cx, cy = gm.case_circle_mm(R)
            dpg.add_line_series(cx, cy, label='case', parent='xs_y')
        dpg.fit_axis_data('xs_x'); dpg.fit_axis_data('xs_y')

    # ---------------------------------------------------------------------------------- sweeps
    def run_sweep(self, *_):
        if self.busy:
            return
        try:
            cfg, _ = gm.values_to_config(self.values())
            params = {}
            for i in (1, 2):
                key = dpg.get_value(f'sw_param{i}')
                txt = dpg.get_value(f'sw_values{i}').strip()
                if key == '(none)' or not txt:
                    continue
                name, sc = SWEEP_MAP[key]
                vals = [float(x) for x in txt.replace(';', ',').split(',') if x.strip()]
                params[name] = [int(x) if sc == 1 and key == 'inj_N' else x*sc for x in vals]
            if not params:
                raise ValueError('choose at least one parameter and its values')
            n_jobs = max(1, int(dpg.get_value('sw_jobs')))
            csv_path = dpg.get_value('sw_csv').strip() or None
            mode = 'zip' if dpg.get_value('sw_mode') == 'paired' else 'grid'
        except Exception as e:
            self.status(f'Sweep input error: {e}', error=True)
            return
        self.busy = True
        self.t_start = time.time()
        dpg.configure_item('sweep_btn', enabled=False)

        def work():
            try:
                from burnfront.motor import sweep
                rows = sweep(cfg, params, mode=mode, n_jobs=n_jobs, csv_path=csv_path, verbose=False)
                self.q.put(('sweep', rows))
            except Exception as e:
                self.q.put(('error', f'{type(e).__name__}: {e}\n{traceback.format_exc()}'))
        threading.Thread(target=work, daemon=True).start()

    def show_sweep(self):
        rows = self.sweep_rows
        dpg.delete_item('sweep_table', children_only=True)
        if not rows:
            return
        keys = [k for k in rows[0] if k not in ('history_csv',)]
        show = [k for k in keys if k in rows[0] and not k.startswith('chem_')][:12]
        for k in show:
            dpg.add_table_column(label=k, parent='sweep_table')
        for r in rows:
            with dpg.table_row(parent='sweep_table'):
                for k in show:
                    v = r.get(k, '')
                    dpg.add_text(f'{v:.5g}' if isinstance(v, float) else str(v))

    # ---------------------------------------------------------------------------------- files
    def save_cfg(self, sender, app_data):
        gm.save_values(app_data['file_path_name'], self.values())
        self.status(f"Saved {app_data['file_path_name']}")

    def load_cfg(self, sender, app_data):
        try:
            self.set_values(gm.load_values(app_data['file_path_name']))
            self.status(f"Loaded {app_data['file_path_name']}")
        except Exception as e:
            self.status(f'Load failed: {e}', error=True)

    def export_csv(self, sender, app_data):
        if self.res is not None:
            self.res.to_csv(app_data['file_path_name'])
            self.status(f"Wrote {app_data['file_path_name']}")

    def export_eng(self, sender, app_data):
        if self.res is None:
            return
        try:
            OD = self.cfg.grain_OD if self.cfg.port['type'] != 'file' else 2*np.sqrt(np.max(self.res.grain.A_env)/np.pi)
            gm.export_eng(app_data['file_path_name'], self.res, dpg.get_value('dry_mass'), OD,
                          dpg.get_value('motor_L')*1e-3, mfg='Burnfront')
            self.status(f"Wrote {app_data['file_path_name']}")
        except Exception as e:
            self.status(f'ENG export failed: {e}', error=True)

    # ---------------------------------------------------------------------------------- CFD
    def run_cfd(self, *_):
        if self.cfd_busy:
            return
        if self.res is None:
            dpg.set_value('cfd_status', 'Run the motor first: the CFD uses its chamber state at the chosen time.')
            return
        from burnfront.cfd import nozzle_flow as nf
        v = self.values()
        cfg = self.cfg
        try:
            geom = gm.nozzle_geometry(v, cfg)
            fc, i, note = gm.cfd_conditions(self.res, cfg, dpg.get_value('cfd_t'))
            grid = nf.make_grid(geom, *gm.CFD_GRIDS[dpg.get_value('cfd_grid')])
        except Exception as e:
            dpg.set_value('cfd_status', f'CFD setup failed: {e}')
            return
        self.cfd = dict(grid=grid, fc=fc, U=None, hist=[], phase='converging', it=0, result=None, note=note,
                        t=float(self.res.t[i]), ER=cfg.ER)
        self.cfd_busy, self.cfd_stop, self.cfd_dirty = True, False, False
        dpg.configure_item('cfd_btn', enabled=False)
        dpg.set_value('cfd_status', f'Compiling and starting the solver ({int(grid.active.sum())} cells) ...')
        dpg.delete_item('cfd_table', children_only=True, slot=1)

        def cb(it, hist, U, phase):
            self.aux_q.put(('cfd_snap', it, list(hist), np.asarray(U), phase))
            return self.cfd_stop

        def work():
            try:
                r = nf.run(grid, fc, cfg.ER, callback=cb)
                self.aux_q.put(('cfd_done', r))
            except Exception as e:
                self.aux_q.put(('cfd_err', f'{type(e).__name__}: {e}'))
        threading.Thread(target=work, daemon=True).start()

    def stop_cfd(self, *_):
        self.cfd_stop = True

    def render_cfd(self):
        from burnfront.cfd import nozzle_flow as nf
        c = self.cfd
        if c is None or (c['U'] is None and c['result'] is None):
            return
        field = dpg.get_value('cfd_field')
        if c['result'] is not None:
            U = np.asarray(nf._cons(c['result'].W, c['fc'].gamma))
        else:
            U = c['U']
        _, F = nf.fields(c['grid'], c['fc'], U)
        phase = {'converging': 'converging', 'averaging': 'plume averaging', 'done': 'final'}[c['phase']]
        title = (f"{field} | t = {c['t']:.2f} s, P0 = {c['fc'].P0/1e5:.1f} bar, T0 = {c['fc'].T0:.0f} K | "
                 f"iteration {c['it']} ({phase})")
        img = gm.render_cfd(c['grid'], F[field], field, title, size=(CFD_W, CFD_H))
        dpg.set_value('cfd_tex', img.ravel())

    def show_cfd_result(self, r):
        s = r.summary
        F1 = s['Cf_1d']*self.cfd['fc'].P0*np.pi*self.cfd['grid'].geom.Rt**2
        rows = [
            ('Nozzle mass flow (CFD)', f"{s['mdot_cfd_kg_s']:.5g}", 'kg/s'),
            ('Ideal 1D choked mass flow', f"{s['mdot_1d_kg_s']:.5g}", 'kg/s'),
            ('Discharge coefficient (2D throat)', f"{s['Cd_2d']:.5f}", ''),
            ('Thrust (CFD, inviscid)', f"{s['thrust_cfd_N']:.5g}", 'N'),
            ('Thrust (ideal 1D, same chamber)', f'{F1:.5g}', 'N'),
            ('Thrust coefficient CFD / 1D', f"{s['Cf_cfd']:.4f} / {s['Cf_1d']:.4f}", ''),
            ('Nozzle efficiency Cf_CFD / Cf_1D', f"{s['Cf_cfd']/s['Cf_1d']:.4f}", ''),
            ('Conical divergence factor (1+cos a)/2', f"{s['lambda_divergence']:.4f}", ''),
            ('Exit Mach (mass-avg) CFD / 1D', f"{s['Me_avg']:.3f} / {s['Me_1d']:.3f}", ''),
            ('Exit pressure CFD / 1D', f"{s['pe_avg_bar']:.4f} / {s['pe_1d_bar']:.4f}", 'bar'),
            ('Normal shock inside nozzle', 'YES (flow would separate)' if s['shock_in_nozzle'] else 'no', ''),
            ('Mass conservation (exit vs inlet)', f"{s['mass_imbalance']:.1e}", ''),
            ('Converged (nozzle)', 'yes' if s['converged'] else 'no (unsteady or stopped)', ''),
        ]
        dpg.delete_item('cfd_table', children_only=True, slot=1)
        for a, b, c in rows:
            with dpg.table_row(parent='cfd_table'):
                dpg.add_text(a); dpg.add_text(b); dpg.add_text(c)
        x = np.ascontiguousarray(s['wall_x']*1e3)
        dpg.set_value('cfd_wall_M', [x, np.ascontiguousarray(s['wall_M'])])
        dpg.set_value('cfd_wall_p', [x, np.ascontiguousarray(s['wall_p']/self.cfd['fc'].P0)])
        for ax in ('cfd_wall_x', 'cfd_wall_y'):
            dpg.fit_axis_data(ax)

    # ---------------------------------------------------------------------------------- thermal
    def run_thermal(self, *_):
        if self.thermal_busy:
            return
        if self.res is None:
            dpg.set_value('th_status', 'Run the motor first: the wall heating uses its chamber history.')
            return
        from burnfront.cfd.thermal import wall_transient
        v = self.values()
        try:
            geom = gm.nozzle_geometry(v, self.cfg)
            inp = gm.thermal_inputs(self.res)
        except Exception as e:
            dpg.set_value('th_status', f'Setup failed: {e}')
            return
        mat, thick = v['wall_material'], float(v['wall_thickness'])*1e-3
        self.thermal_busy = True
        dpg.configure_item('th_btn', enabled=False)
        dpg.set_value('th_status', 'Computing wall heating over the burn ...')

        def work():
            try:
                w = wall_transient(geom, inp['t'], inp['P0'], inp['T0'], inp['gamma'], inp['Rgas'], inp['cstar'],
                                   material=mat, thickness=thick, n_x=50, n_r=24, T_amb=294.0, dt_max=0.02)
                self.aux_q.put(('th_done', (w, geom)))
            except Exception as e:
                self.aux_q.put(('th_err', f'{type(e).__name__}: {e}'))
        threading.Thread(target=work, daemon=True).start()

    def show_thermal(self, w, geom):
        x = np.ascontiguousarray(w.x*1e3)
        it = int(np.argmax(np.max(w.q, axis=1)))
        dpg.set_value('th_q', [x, np.ascontiguousarray(w.q[it]/1e6)])
        dpg.configure_item('th_q', label=f'heat flux at t = {w.t[it]:.2f} s')
        dpg.set_value('th_Tend', [x, np.ascontiguousarray(w.Tw[-1])])
        dpg.set_value('th_Tmax', [x, np.ascontiguousarray(w.Tw.max(axis=0))])
        dpg.set_value('th_lim_x', [np.array([x[0], x[-1]]), np.array([w.T_limit, w.T_limit])])
        jt = int(np.argmin(np.abs(w.x)))
        jm = int(np.argmax(w.Tw.max(axis=0)))
        t = np.ascontiguousarray(w.t)
        dpg.set_value('th_Tt_throat', [t, np.ascontiguousarray(w.Tw[:, jt])])
        dpg.set_value('th_Tt_max', [t, np.ascontiguousarray(w.Tw[:, jm])])
        dpg.configure_item('th_Tt_max', label=f'hottest station (x = {w.x[jm]*1e3:.1f} mm)')
        dpg.set_value('th_lim_t', [np.array([t[0], t[-1]]), np.array([w.T_limit, w.T_limit])])
        for ax in ('th_q_x', 'th_q_y', 'th_T_x', 'th_t_x'):
            dpg.fit_axis_data(ax)
        # temperature axes always include the material limit line
        top = max(w.T_limit, float(w.Tw.max()))*1.05
        bot = min(float(w.Tw.min()), 294.0) - 50.0
        for ax in ('th_T_y', 'th_t_y'):
            dpg.set_axis_limits(ax, bot, top)
        pk = w.peak
        ttl = w.time_to_limit()
        msg = (f"{w.material}: peak inner-wall temperature {pk['T']:.0f} K at x = {pk['x']*1e3:.1f} mm, "
               f"t = {pk['t']:.2f} s; peak heat flux {w.q.max()/1e6:.2f} MW/m2 (throat h = {w.h[it, jt]:.0f} W/m2/K). ")
        msg += (f'EXCEEDS the {w.T_limit:.0f} K limit at t = {ttl:.2f} s.' if np.isfinite(ttl)
                else f'Stays below the {w.T_limit:.0f} K limit (margin {w.T_limit - pk["T"]:.0f} K).')
        dpg.set_value('th_status', msg)
        dpg.configure_item('th_status', color=(255, 120, 120) if np.isfinite(ttl) else (190, 220, 190))

    def poll_aux(self):
        latest = None
        while True:
            try:
                msg = self.aux_q.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == 'cfd_snap':
                latest = msg
            elif kind == 'cfd_done':
                r = msg[1]
                self.cfd.update(result=r, phase='done', it=r.iterations, hist=[tuple(h) for h in r.residuals])
                self.cfd_busy = False
                latest = None
                dpg.configure_item('cfd_btn', enabled=True)
                dpg.set_value('cfd_status', f"Finished after {r.iterations} iterations; {self.cfd['note']}. "
                              'Inviscid flow: no boundary layers or turbulent mixing; the plume is a pseudo-time average.')
                self.show_cfd_result(r)
                self._cfd_residuals()
                self.cfd_dirty = True
            elif kind == 'cfd_err':
                self.cfd_busy = False
                dpg.configure_item('cfd_btn', enabled=True)
                dpg.set_value('cfd_status', 'CFD failed: ' + msg[1])
            elif kind == 'th_done':
                self.thermal_busy = False
                dpg.configure_item('th_btn', enabled=True)
                self.thermal = msg[1]
                self.show_thermal(*msg[1])
            elif kind == 'th_err':
                self.thermal_busy = False
                dpg.configure_item('th_btn', enabled=True)
                dpg.set_value('th_status', 'Thermal analysis failed: ' + msg[1])
            elif kind == 'cad':
                dpg.set_value('cad_info', msg[1])
                dpg.configure_item('cad_info', color=(255, 110, 110) if msg[2] else (190, 220, 190))
        if latest is not None and self.cfd is not None and self.cfd['result'] is None:
            _, it, hist, U, phase = latest
            self.cfd.update(it=it, hist=hist, U=U, phase=phase)
            dpg.set_value('cfd_status', f'Iteration {it}: nozzle residual {hist[-1][1]:.2e} ({phase}) ...')
            self._cfd_residuals()
            self.cfd_dirty = True
        if self.cfd_dirty and time.time() - self.cfd_render_t > 0.4:
            self.cfd_dirty = False
            self.cfd_render_t = time.time()
            try:
                self.render_cfd()
            except Exception as e:
                dpg.set_value('cfd_status', f'Render failed: {e}')

    def _cfd_residuals(self):
        h = np.asarray(self.cfd['hist'], float)
        if h.size:
            dpg.set_value('cfd_res_noz', [np.ascontiguousarray(h[:, 0]), np.ascontiguousarray(np.maximum(h[:, 1], 1e-16))])
            dpg.set_value('cfd_res_all', [np.ascontiguousarray(h[:, 0]), np.ascontiguousarray(np.maximum(h[:, 2], 1e-16))])
            dpg.fit_axis_data('cfd_res_x'); dpg.fit_axis_data('cfd_res_y')

    def _cfd_field_changed(self, *_):
        self.cfd_dirty = True

    # ---------------------------------------------------------------------------------- layout
    def build(self):
        root = Path(imp_files('burnfront'))
        with dpg.font_registry():
            font = dpg.add_font(str(root/'resources'/'fonts'/'Roboto-Regular.ttf'), 15)
        dpg.bind_font(font)
        with dpg.texture_registry():
            dpg.add_raw_texture(RENDER_W, RENDER_H, np.zeros(RENDER_W*RENDER_H*4, np.float32),
                                format=dpg.mvFormat_Float_rgba, tag='surf_tex')
            dpg.add_raw_texture(CFD_W, CFD_H, np.zeros(CFD_W*CFD_H*4, np.float32),
                                format=dpg.mvFormat_Float_rgba, tag='cfd_tex')
        with dpg.file_dialog(tag='dlg_cad', show=False, width=760, height=440, callback=self.add_cad_file,
                             directory_selector=False, label='Add a STEP or STL grain file'):
            dpg.add_file_extension('Grain CAD (*.step *.stp *.stl){.step,.stp,.stl,.STEP,.STP,.STL}')
            dpg.add_file_extension('.*')
        for tag, cb, ext in [('dlg_save', self.save_cfg, '.json'), ('dlg_load', self.load_cfg, '.json'),
                             ('dlg_csv', self.export_csv, '.csv'), ('dlg_eng', self.export_eng, '.eng')]:
            with dpg.file_dialog(tag=tag, show=False, width=720, height=420, callback=cb, directory_selector=False):
                dpg.add_file_extension(ext)

        with dpg.window(tag='main'):
            with dpg.menu_bar():
                with dpg.menu(label='File'):
                    dpg.add_menu_item(label='Load configuration...', callback=lambda: dpg.show_item('dlg_load'))
                    dpg.add_menu_item(label='Save configuration...', callback=lambda: dpg.show_item('dlg_save'))
                    dpg.add_separator()
                    dpg.add_menu_item(label='Export history CSV...', callback=lambda: dpg.show_item('dlg_csv'))
                    dpg.add_menu_item(label='Export ENG...', callback=lambda: dpg.show_item('dlg_eng'))
                with dpg.menu(label='Export settings'):
                    dpg.add_input_float(label='Dry mass [kg]', tag='dry_mass', default_value=5.0, width=120, step=0)
                    dpg.add_input_float(label='Motor length [mm]', tag='motor_L', default_value=1200.0, width=120, step=0)
            with dpg.group(horizontal=True):
                with dpg.child_window(width=LEFT_W, tag='left'):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='  Run  ', tag='run_btn', callback=self.run)
                        dpg.add_text('', tag='status', wrap=LEFT_W - 90)
                    dpg.add_separator()
                    dpg.add_button(label='+  Add STEP / STL grain file...', tag='cad_btn', width=-1, height=34,
                                   callback=lambda: dpg.show_item('dlg_cad'))
                    with dpg.theme() as accent:
                        with dpg.theme_component(dpg.mvButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 110, 200))
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 135, 230))
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (30, 90, 170))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                    dpg.bind_item_theme('cad_btn', accent)
                    with dpg.tooltip('cad_btn'):
                        dpg.add_text('Load the grain geometry from CAD (STEP: exact; STL: set its units below).\n'
                                     'The mesh is validated immediately; the 3D model is used for the run.')
                    dpg.add_text('No CAD file loaded: the analytic port below is used.', tag='cad_info',
                                 wrap=LEFT_W - 30, color=(170, 170, 170))
                    dpg.add_separator()
                    self.build_inputs()
                with dpg.child_window(tag='right'):
                    with dpg.tab_bar(tag='tabs'):
                        with dpg.tab(label='Grain view', tag='tab_view'):
                            with dpg.group(tag='view3d_group'):
                                with dpg.group(horizontal=True):
                                    dpg.add_text('Snapshot')
                                    dpg.add_combo([], tag='snap_combo', width=140, callback=self._req_render)
                                    dpg.add_slider_float(label='azimuth', tag='az', default_value=-60, min_value=-180,
                                                         max_value=180, width=180, callback=self._req_render)
                                    dpg.add_slider_float(label='elevation', tag='el', default_value=20, min_value=-90,
                                                         max_value=90, width=140, callback=self._req_render)
                                dpg.add_image('surf_tex')
                            dpg.add_text('2D outline mode: the grain is a single cross-section extruded along the '
                                         'port; see the Cross-section tab.', tag='view3d_note', show=False)
                        with dpg.tab(label='Cross-section', tag='tab_xs'):
                            dpg.add_slider_float(label='axial position (0 = head, 1 = aft)', tag='xs_z_slider',
                                                 default_value=0.5, min_value=0.0, max_value=1.0, width=300,
                                                 callback=self.draw_sections)
                            dpg.add_slider_int(label='axial station', tag='xs_station', default_value=0, min_value=0,
                                               max_value=64, width=300, callback=self.draw_sections, show=False)
                            with dpg.plot(tag='xs_plot', label='Cross-section', height=-1, width=-1, equal_aspects=True):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label='x [mm]', tag='xs_x')
                                dpg.add_plot_axis(dpg.mvYAxis, label='y [mm]', tag='xs_y')
                        with dpg.tab(label='Histories', tag='tab_hist'):
                            with dpg.subplots(3, 2, height=-1, width=-1, tag='hist_sub'):
                                self._plot('p_thrust', 'Thrust', 'N', [('s_thrust', 'thrust')])
                                self._plot('p_press', 'Pressures', 'bar', [('s_pc', 'chamber'), ('s_pt', 'tank')])
                                self._plot('p_of', 'O/F', '', [('s_of', 'O/F')])
                                self._plot('p_flux', 'Local mass flux G', 'kg/m2/s',
                                           [('s_G_head', 'head'), ('s_G_mid', 'mid'), ('s_G_aft', 'aft')])
                                self._plot('p_mdot', 'Mass flow', 'kg/s', [('s_mdot_ox', 'oxidizer'), ('s_mdot_f', 'fuel')])
                        with dpg.tab(label='Flux map', tag='tab_flux'):
                            with dpg.group(horizontal=True):
                                dpg.add_colormap_scale(tag='fluxmap_scale', min_scale=0, max_scale=1, height=-1,
                                                       colormap=dpg.mvPlotColormap_Viridis)
                                with dpg.plot(tag='fluxmap', label='Local mass flux G(z, t) [kg/m2/s]', height=-1, width=-1):
                                    dpg.add_plot_axis(dpg.mvXAxis, label='t [s]', tag='fluxmap_x')
                                    dpg.add_plot_axis(dpg.mvYAxis, label='z [mm]', tag='fluxmap_y')
                                dpg.bind_colormap('fluxmap', dpg.mvPlotColormap_Viridis)
                        with dpg.tab(label='CFD: nozzle & plume', tag='tab_cfd'):
                            with dpg.group(horizontal=True):
                                dpg.add_text('Time in burn [s]')
                                dpg.add_input_float(tag='cfd_t', default_value=1.0, width=80, step=0, format='%.2f')
                                dpg.add_combo(list(gm.CFD_GRIDS), tag='cfd_grid', default_value='Coarse (fast)', width=130)
                                dpg.add_combo(CFD_FIELDS, tag='cfd_field', default_value='Mach', width=150,
                                              callback=self._cfd_field_changed)
                                dpg.add_button(label=' Run CFD ', tag='cfd_btn', callback=self.run_cfd)
                                dpg.add_button(label=' Stop ', callback=self.stop_cfd)
                            dpg.add_text('Axisymmetric compressible (Euler) flow through the aft chamber, nozzle and '
                                         'exhaust plume at the motor state of the chosen time; updates live while it converges.',
                                         tag='cfd_status', wrap=CFD_W)
                            dpg.add_image('cfd_tex')
                            with dpg.group(horizontal=True):
                                with dpg.plot(label='Residuals', height=230, width=300):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label='iteration', tag='cfd_res_x')
                                    with dpg.plot_axis(dpg.mvYAxis, label='mean |drho/rho| per iteration', tag='cfd_res_y',
                                                       scale=dpg.mvPlotScale_Log10):
                                        dpg.add_line_series([], [], label='nozzle', tag='cfd_res_noz')
                                        dpg.add_line_series([], [], label='whole domain', tag='cfd_res_all')
                                with dpg.plot(label='Nozzle wall', height=230, width=300):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label='x [mm]', tag='cfd_wall_x')
                                    with dpg.plot_axis(dpg.mvYAxis, label='', tag='cfd_wall_y'):
                                        dpg.add_line_series([], [], label='Mach', tag='cfd_wall_M')
                                        dpg.add_line_series([], [], label='p / P0', tag='cfd_wall_p')
                                with dpg.table(tag='cfd_table', header_row=True, borders_innerH=True, width=-1,
                                               policy=dpg.mvTable_SizingStretchProp):
                                    dpg.add_table_column(label='Result')
                                    dpg.add_table_column(label='Value')
                                    dpg.add_table_column(label='')
                        with dpg.tab(label='Nozzle thermal', tag='tab_thermal'):
                            with dpg.group(horizontal=True):
                                dpg.add_button(label=' Compute wall heating over the burn ', tag='th_btn',
                                               callback=self.run_thermal)
                                dpg.add_text('(material and wall thickness: "Nozzle geometry & wall" on the left)')
                            dpg.add_text('Bartz gas-side heat transfer and transient conduction through the nozzle wall, '
                                         'driven by the chamber history of the last motor run.', tag='th_status', wrap=1000)
                            with dpg.subplots(2, 2, height=-1, width=-1):
                                with dpg.plot(label='Gas-side heat flux (peak time)'):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label='x [mm] (throat at 0)', tag='th_q_x')
                                    with dpg.plot_axis(dpg.mvYAxis, label='MW/m2', tag='th_q_y'):
                                        dpg.add_line_series([], [], label='heat flux', tag='th_q')
                                with dpg.plot(label='Inner wall temperature along the nozzle'):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label='x [mm]', tag='th_T_x')
                                    with dpg.plot_axis(dpg.mvYAxis, label='K', tag='th_T_y'):
                                        dpg.add_line_series([], [], label='end of run', tag='th_Tend')
                                        dpg.add_line_series([], [], label='maximum over run', tag='th_Tmax')
                                        dpg.add_line_series([], [], label='material limit', tag='th_lim_x')
                                with dpg.plot(label='Inner wall temperature vs time'):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label='t [s]', tag='th_t_x')
                                    with dpg.plot_axis(dpg.mvYAxis, label='K', tag='th_t_y'):
                                        dpg.add_line_series([], [], label='throat', tag='th_Tt_throat')
                                        dpg.add_line_series([], [], label='hottest station', tag='th_Tt_max')
                                        dpg.add_line_series([], [], label='material limit', tag='th_lim_t')
                        with dpg.tab(label='Summary', tag='tab_sum'):
                            with dpg.table(tag='summary_table', header_row=True, borders_innerH=True,
                                           policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column(label='Quantity')
                                dpg.add_table_column(label='Value')
                                dpg.add_table_column(label='Unit')
                            dpg.add_spacer(height=8)
                            dpg.add_text('', tag='summary_text', wrap=900)
                        with dpg.tab(label='Sweep', tag='tab_sweep'):
                            keys = ['(none)'] + list(SWEEP_MAP)
                            for i in (1, 2):
                                with dpg.group(horizontal=True):
                                    dpg.add_combo(keys, tag=f'sw_param{i}', default_value=keys[1] if i == 1 else '(none)', width=160)
                                    dpg.add_input_text(tag=f'sw_values{i}', hint='values in GUI units, e.g. 4, 6, 8', width=320,
                                                       default_value='4, 6, 8' if i == 1 else '')
                            with dpg.group(horizontal=True):
                                dpg.add_combo(['grid', 'paired'], tag='sw_mode', default_value='grid', width=90)
                                dpg.add_input_int(label='parallel jobs', tag='sw_jobs', default_value=max(1, (os.cpu_count() or 2)//2), width=100)
                                dpg.add_input_text(label='CSV', tag='sw_csv', default_value='sweep.csv', width=220)
                                dpg.add_button(label='Run sweep', tag='sweep_btn', callback=self.run_sweep)
                            with dpg.table(tag='sweep_table', header_row=True, borders_innerH=True, resizable=True,
                                           policy=dpg.mvTable_SizingStretchProp, scrollX=True):
                                pass
        self.update_visibility()

    def _plot(self, tag, title, ylabel, series):
        with dpg.plot(tag=tag, label=title):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label='t [s]')
            with dpg.plot_axis(dpg.mvYAxis, label=ylabel):
                for stag, lab in series:
                    dpg.add_line_series([], [], label=lab, tag=stag)

    def _req_render(self, *_):
        self.render_due = True
        self.last_change = time.time()

    def frame(self):
        self.poll()
        self.poll_aux()
        if self.render_due and not self.busy and time.time() - self.last_change > 0.12:
            self.render_due = False
            try:
                self.render()
            except Exception as e:
                self.status(f'Render failed: {e}', error=True)


def main():
    import jax
    jax.config.update('jax_enable_x64', True)
    dpg.create_context()
    root = Path(imp_files('burnfront'))
    dpg.create_viewport(title='Burnfront', width=1480, height=920, small_icon=str(root/'resources'/'icon.ico'),
                        large_icon=str(root/'resources'/'icon.ico'))
    app = App()
    app.build()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window('main', True)
    while dpg.is_dearpygui_running():
        app.frame()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
