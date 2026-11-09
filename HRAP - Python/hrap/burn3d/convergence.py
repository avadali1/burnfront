"""Grid-convergence study for 3D burns (Celik et al. 2008, ASME J. Fluids Eng. 130(7) 078001).

For three grids h1 < h2 < h3 (1 = finest) and a quantity phi_i, the observed order p solves
    p = | ln|e32/e21| + ln((r21^p - s)/(r32^p - s)) | / ln r21,   e_ij = phi_i - phi_j, s = sign(e32/e21)
then phi_ext = (r21^p phi1 - phi2)/(r21^p - 1) and GCI_fine = 1.25 |(phi1 - phi2)/phi1| / (r21^p - 1).
"""
import csv
import time
from dataclasses import dataclass, field

import numpy as np

from hrap.burn3d.burn import burn_prescribed


def celik(phi, h):
    """phi, h ordered finest first (3 values). Returns dict(p, phi_ext, gci, oscillatory)."""
    f1, f2, f3 = phi
    r21, r32 = h[1]/h[0], h[2]/h[1]
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return dict(p=np.nan, phi_ext=f1, gci=0.0 if e21 == 0 else np.nan, oscillatory=False)
    s = np.sign(e32/e21)
    p = abs(np.log(abs(e32/e21)))/np.log(r21)
    for _ in range(200):
        q = np.log((r21**p - s)/(r32**p - s)) if (r21**p - s) > 0 and (r32**p - s) > 0 else 0.0
        p_new = abs(np.log(abs(e32/e21)) + q)/np.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    phi_ext = (r21**p*f1 - f2)/(r21**p - 1)
    gci = 1.25*abs((f1 - f2)/f1)/(r21**p - 1) if f1 != 0 else np.nan
    return dict(p=p, phi_ext=phi_ext, gci=gci, oscillatory=bool(s < 0))


def least_squares_uncertainty(phi, h):
    """Eca & Hoekstra (2014, J. Comput. Phys. 262:104) style estimate for scattered data on >= 4 grids.

    Fits phi = phi0 + alpha h^p (p scanned in [0.5, 4], linear LS for phi0, alpha) and, as a fallback
    when p is outside (0.95, 2.05) the fixed second-order form phi0 + alpha h^2. The uncertainty of
    the finest-grid value is U = Fs |phi_fine - phi0| + sigma + |residual_fine|, with Fs = 1.25 for a
    well-behaved order (0.5 <= p <= 2.1) and Fs = 3 otherwise; sigma is the fit standard deviation.
    phi, h ordered finest first.
    """
    phi, h = np.asarray(phi, float), np.asarray(h, float)

    def fit(p):
        A = np.stack([np.ones_like(h), h**p], 1)
        coef, *_ = np.linalg.lstsq(A, phi, rcond=None)
        r = phi - A @ coef
        return coef, r

    ps = np.linspace(0.5, 4.0, 351)
    sse = [np.sum(fit(p)[1]**2) for p in ps]
    p = float(ps[int(np.argmin(sse))])
    if not 0.95 < p < 2.05:
        p_used = 2.0
    else:
        p_used = p
    (phi0, alpha), r = fit(p_used)
    dof = max(len(phi) - 2, 1)
    sigma = float(np.sqrt(np.sum(r**2)/dof))
    Fs = 1.25 if 0.5 <= p <= 2.1 else 3.0
    U = Fs*abs(phi[0] - phi0) + sigma + abs(r[0])
    return dict(p=p, p_used=p_used, phi0=float(phi0), sigma=sigma, U=float(U),
                U_rel=float(U/abs(phi[0])) if phi[0] != 0 else np.nan)


METRICS = {
    'V_fuel0 [cm3]': lambda g, r, T: g.fuel_volume()*1e6,
    'V_port0 [cm3]': lambda g, r, T: r.V_port[0]*1e6,
    'A_burn(T/2) [cm2]': lambda g, r, T: np.interp(T/2, r.t, r.A_burn)*1e4,
    'mdot_f(T/2) [g/s]': lambda g, r, T: np.interp(T/2, r.t, r.mdot_f)*1e3,
    'm_burned(T) [g]': lambda g, r, T: r.m_burned[-1]*1e3,
}


@dataclass
class ConvergenceReport:
    resolutions: list
    h: list
    values: dict                       # metric -> list (same order as resolutions)
    timings: list
    analysis: dict = field(default_factory=dict)   # Celik, finest three grids
    ls: dict = field(default_factory=dict)         # least-squares (>= 4 grids)

    def __str__(self):
        cols = [f'n_xy={n}' for n in self.resolutions]
        lines = ['Grid convergence: values per grid; Celik et al. (2008) on the finest three grids;',
                 'least-squares fit phi0 + a h^p over all grids with uncertainty U of the finest value',
                 '(Eca & Hoekstra 2014).',
                 f"{'metric':20s} " + ' '.join(f'{c:>11s}' for c in cols) +
                 f" | {'p(Celik)':>8s} {'GCI':>7s} | {'p(LS)':>6s} {'phi0(LS)':>11s} {'U fine':>8s}"]
        for k, v in self.values.items():
            a, b = self.analysis[k], self.ls.get(k)
            osc = '*' if a['oscillatory'] else ' '
            ls = (f"{b['p']:6.2f} {b['phi0']:11.4f} {b['U_rel']*100:7.3f}%" if b else '')
            lines.append(f'{k:20s} ' + ' '.join(f'{x:11.4f}' for x in v) +
                         f" | {a['p']:8.2f} {a['gci']*100:6.2f}%{osc}| {ls}")
        lines.append(f"{'wall time [s]':20s} " + ' '.join(f'{t:11.1f}' for t in self.timings))
        lines.append('* oscillatory convergence on the finest three grids (Celik GCI not reliable)')
        return '\n'.join(lines)

    def to_csv(self, path):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric'] + [f'n_xy={n}' for n in self.resolutions])
            for k, v in self.values.items():
                w.writerow([k] + list(v))
            w.writerow([])
            w.writerow(['metric', 'celik_p', 'celik_extrapolated', 'celik_gci_fine', 'celik_oscillatory',
                        'ls_p', 'ls_phi0', 'ls_sigma', 'ls_U_fine', 'ls_U_fine_rel'])
            for k in self.values:
                a, b = self.analysis[k], self.ls.get(k, {})
                w.writerow([k, a['p'], a['phi_ext'], a['gci'], a['oscillatory'], b.get('p'), b.get('phi0'),
                            b.get('sigma'), b.get('U'), b.get('U_rel')])
            w.writerow(['h [m]'] + list(self.h))
            w.writerow(['time [s]'] + list(self.timings))


def grid_convergence(build, resolutions, law, mdot_ox, t_end, metrics=None, **burn_kw):
    """Run ``build(n_xy) -> Grain3D`` and a prescribed-oxidizer burn at each resolution (>= 3,
    coarsest first) and analyse the finest three with Celik's procedure."""
    metrics = METRICS if metrics is None else metrics
    values = {k: [] for k in metrics}
    hs, times = [], []
    for n in resolutions:
        t0 = time.time()
        g = build(n)
        res = burn_prescribed(g, law, mdot_ox, t_end, **burn_kw)
        times.append(time.time() - t0)
        hs.append(g.grid.h[0])
        for k, fn in metrics.items():
            values[k].append(float(fn(g, res, t_end)))
    rep = ConvergenceReport(list(resolutions), hs, values, times)
    for k, v in values.items():
        rep.analysis[k] = celik(v[::-1][:3], hs[::-1][:3])
        if len(v) >= 4:
            rep.ls[k] = least_squares_uncertainty(v[::-1], hs[::-1])
    return rep
