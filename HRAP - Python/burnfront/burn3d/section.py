"""2D mode: a z-invariant grain described by one cross-section, with axial stations.

For a single cross-section with exact 2D signed distance phi, the uniform in-plane offset
{phi < d} is the exact level-set (Hopf-Lax) solution, so A(d) and P(d) = dA/dd are tabulated once
over the whole web with the same sub-cell measure as the 3D core. Each axial station carries its own
depth d(z) and uses the same axial flux model as 3D (burnfront.burn3d.flow). This is the 3D algorithm
restricted to z-invariant geometry, minus re-tabulation.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure as skmeasure

from burnfront.burn3d import measure
from burnfront.burn3d.axial import station_eval
from burnfront.burn3d.burn import BurnResult
from burnfront.burn3d.grid import Grid3D
from burnfront.burn3d import shapes as _shapes


@dataclass
class SectionTable:
    levels: np.ndarray   # (K,) offsets, uniform spacing dd from 0
    A: np.ndarray        # (K,) open port area
    M: np.ndarray        # (K,) dA/dd (monotone Hermite slopes)
    dd: float
    A_env: float         # area inside the case
    web_contact: float   # offset at which the front first reaches the case (burn-through)

    def stations(self, n_z):
        """Broadcast to (n_z, K) arrays for the station ODE."""
        return np.tile(self.A, (n_z, 1)), np.tile(self.M, (n_z, 1))


class Section2D:
    """Cross-section of a grain: burning port(s) ``port`` (a shapes.Shape, evaluated at z=0) inside a
    circular inert case of diameter OD, sampled with n_xy cells across OD."""

    def __init__(self, port, OD, n_xy=256, margin=4):
        g = Grid3D.cylinder(OD, 1.0, n_xy, 2, margin)
        X, Y = g.xy_mesh()
        self.port, self.OD, self.R = port, OD, OD/2
        self.x = g.x
        self.h = g.h[0]
        self.X, self.Y = X, Y
        self.phi = np.asarray(port.f(X, Y, np.zeros_like(X)), dtype=float)
        self.chi = np.hypot(X, Y) - self.R

    @classmethod
    def from_polygons(cls, polygons, OD, n_xy=256, margin=4):
        polys = [_shapes.polygon(p) for p in polygons]
        port = polys[0] if len(polys) == 1 else _shapes.union(*polys)
        return cls(port, OD, n_xy, margin)

    def _phi_at(self, pts):
        ij = (pts - self.x[0]) / self.h
        return ndimage.map_coordinates(self.phi, [ij[:, 0], ij[:, 1]], order=3, mode='nearest')

    def table(self, dd=None):
        dd = self.h/8 if dd is None else dd
        fuel = (self.chi < 0) & (self.phi > 0)
        dmax = (float(self.phi[fuel].max()) if fuel.any() else 0.0) + 2*self.h
        levels = np.arange(int(np.ceil(dmax/dd)) + 1) * dd
        A = measure.port_area_table(self.phi[:, :, None], self.chi[:, :, None], levels, self.h, self.h)[0]
        M = measure.pchip_slopes(A, dd)
        A_env = float(measure.slice_areas(self.chi[:, :, None], self.h, self.h)[0])
        # Burn-through offset: minimum of phi on the case circle (cubic interpolation, fine sampling)
        th = np.linspace(0, 2*np.pi, max(720, int(8*np.pi*self.OD/self.h)), endpoint=False)
        web = float(np.min(self._phi_at(self.R*np.stack([np.cos(th), np.sin(th)], 1))))
        return SectionTable(levels, A, M, dd, A_env, web)

    def contours(self, d):
        """Front polylines (metres) at offset d, clipped to the case (NaN-separated outside)."""
        out = []
        for c in skmeasure.find_contours(self.phi, d):
            xy = self.x[0] + c*self.h
            xy[np.hypot(xy[:, 0], xy[:, 1]) > self.R] = np.nan
            out.append(xy)
        return out

    def helical_kappa(self, table, twist_rate=0.0, helix_radius=0.0, helix_pitch=None,
                      n_eval=96, n_phase=32):
        """Optional 2D approximation for twisted / helical ports: kappa(d) = P_eff/P >= 1 where
        P_eff = contour integral of sqrt(1 + (d phi/dz)^2) dl on the offset front.

        twist:  d phi/dz = omega (p x n)               (cross-section rotating at omega rad/m)
        helix:  d phi/dz = n . c'(z), |c'| = 2 pi e / pitch, averaged over the helix phase
                (cross-section translating around a helix of radius e).
        Exact at d = 0 (surface area of the twisted/helical port); for d > 0 it uses the in-plane
        offset of the section, which is the approximation.
        """
        if twist_rate and helix_radius:
            raise ValueError('use either twist_rate or helix_radius')
        lv = table.levels
        sample = np.unique(np.round(np.linspace(0, lv.size - 1, min(n_eval, lv.size))).astype(int))
        kap_s = np.ones(sample.size)
        alpha = np.linspace(0, 2*np.pi, n_phase, endpoint=False)
        cs = (2*np.pi*helix_radius/helix_pitch) if helix_radius else 0.0
        for j, i in enumerate(sample):
            Ltot = Leff = 0.0
            for c in skmeasure.find_contours(self.phi, lv[i]):
                xy = self.x[0] + c*self.h
                seg = np.diff(xy, axis=0)
                ln = np.hypot(seg[:, 0], seg[:, 1])
                mid = 0.5*(xy[1:] + xy[:-1])
                keep = (np.hypot(mid[:, 0], mid[:, 1]) < self.R) & (ln > 0)
                if not keep.any():
                    continue
                seg, ln, mid = seg[keep], ln[keep], mid[keep]
                nx, ny = seg[:, 1]/ln, -seg[:, 0]/ln
                if twist_rate:
                    s = twist_rate*(mid[:, 0]*ny - mid[:, 1]*nx)
                    wgt = np.sqrt(1 + s*s)
                else:
                    s = cs*(-nx[:, None]*np.sin(alpha) + ny[:, None]*np.cos(alpha))
                    wgt = np.mean(np.sqrt(1 + s*s), axis=1)
                Ltot += ln.sum()
                Leff += (ln*wgt).sum()
            kap_s[j] = Leff/Ltot if Ltot > 0 else (kap_s[j-1] if j else 1.0)
        return np.interp(np.arange(lv.size), sample, kap_s)


def image_port_polygons(path, OD, simplify_px=0.1):
    """Port outlines from a cross-section image: dark/gray grain on a light background, light port(s).

    The grain disk is found from the largest dark region (minimum enclosing circle -> centre and
    scale OD / (2 r_px)). Port boundaries are sub-pixel iso-contours of the grayscale image halfway
    between the grain gray level and white (no blurring), simplified to ``simplify_px`` pixels.
    Returns (list of (N, 2) polygons in metres, centred on the grain axis, y up; info dict).
    """
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    dark = (gray < 240).astype(np.uint8)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(cnts, key=cv2.contourArea)
    (cx, cy), r_px = cv2.minEnclosingCircle(outer)
    scale = OD / (2*r_px)
    grain_level = float(np.median(gray[dark.astype(bool)]))
    level = 0.5*(grain_level + 255.0)
    polys = []
    for c in skmeasure.find_contours(gray, level):
        if not np.allclose(c[0], c[-1]):
            continue                                  # open contour: touches the image border
        if np.max(np.hypot(c[:, 0] - cy, c[:, 1] - cx)) > r_px - 1.5:
            continue                                  # the grain's outer boundary, not a port
        c = skmeasure.approximate_polygon(c, tolerance=simplify_px)[:-1]
        if len(c) < 3:
            continue
        polys.append(np.stack([(c[:, 1] - cx)*scale, (cy - c[:, 0])*scale], axis=1))
    if not polys:
        raise ValueError('no closed port outline found inside the grain')
    return polys, dict(cx=cx, cy=cy, r_px=r_px, scale=scale, level=level)


def burn_section(table, L, n_z, law, mdot_ox, t_end, dt, kappa=None):
    """2D-mode burn with prescribed oxidizer flow (numpy RK4, fixed step). Returns a BurnResult."""
    f_ox = mdot_ox if callable(mdot_ox) else (lambda t, c=float(mdot_ox): c)
    z = np.linspace(0.0, L, n_z)
    w = measure.trapz_weights(z)
    A_tab, M_tab = table.stations(n_z)
    kap = None if kappa is None else np.tile(kappa, (n_z, 1))
    k, n, rho = law.k(L), law.n, law.rho
    A_env = np.full(n_z, table.A_env)
    m_fuel0 = rho*float(np.sum(w*(A_env - A_tab[:, 0])))

    def ev(d, t):
        return station_eval(A_tab, M_tab, table.dd, d, f_ox(t), z, w, k, n, rho, law.flux, kap)

    keys = ['t', 'mox', 'mf', 'mb', 'V', 'Ab', 'A', 'P', 'G', 'r', 'mz', 'dep', 'mgeo']
    rec = {key: [] for key in keys}
    d, m, t = np.zeros(n_z), 0.0, 0.0
    t_bt = m_bt = np.nan
    nsteps = int(np.ceil(t_end/dt - 1e-9))
    for i in range(nsteps + 1):
        e1 = ev(d, t)
        for key, val in (('t', t), ('mox', f_ox(t)), ('mf', e1['mdot_f']), ('mb', m),
                         ('V', np.sum(w*e1['A'])), ('Ab', np.sum(w*e1['P'])), ('A', e1['A']),
                         ('P', e1['P']), ('G', e1['G']), ('r', e1['rdot']), ('mz', e1['mdot_z']),
                         ('dep', d.copy()), ('mgeo', rho*np.sum(w*(e1['A'] - A_tab[:, 0])))):
            rec[key].append(val)
        if i == nsteps or not np.any(e1['P'] > 0):
            break
        h = min(dt, t_end - t)
        e2 = ev(d + 0.5*h*e1['ddot'], t + 0.5*h)
        e3 = ev(d + 0.5*h*e2['ddot'], t + 0.5*h)
        e4 = ev(d + h*e3['ddot'], t + h)
        dn = d + h/6*(e1['ddot'] + 2*e2['ddot'] + 2*e3['ddot'] + e4['ddot'])
        mn = m + h/6*(e1['mdot_f'] + 2*e2['mdot_f'] + 2*e3['mdot_f'] + e4['mdot_f'])
        if np.isnan(t_bt) and np.any(dn >= table.web_contact):
            hit = dn >= table.web_contact
            frac = np.min(np.clip((table.web_contact - d[hit])/np.maximum(dn[hit] - d[hit], 1e-300), 0, 1))
            t_bt = t + frac*h
            from burnfront.burn3d.measure import hermite_eval
            A_bt, _ = hermite_eval(A_tab, M_tab, table.dd, d + frac*(dn - d))
            m_bt = rho*float(np.sum(w*(A_env - A_bt)))
        d, m, t = dn, mn, t + h

    a = {key: np.array(v) for key, v in rec.items()}
    return BurnResult(
        z=z, t=a['t'], mdot_ox=a['mox'], mdot_f=a['mf'], m_burned=a['mb'], V_port=a['V'],
        A_burn=a['Ab'], A_port_z=a['A'], P_z=a['P'], G_z=a['G'], rdot_z=a['r'], mdot_z=a['mz'],
        depth_z=a['dep'], m_fuel0=m_fuel0, t_geo=a['t'], m_burned_geo=a['mgeo'],
        t_burnthrough=t_bt, m_sliver_burnthrough=m_bt, n_steps=len(a['t']) - 1,
    )
