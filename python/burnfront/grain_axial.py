"""JAX engine part for tabulated axial grains (the 2D fast mode, and 3D segments).

The grain is represented by offset tables A[z, k] (open port area after in-plane offset k*dd),
slopes M = dA/dd and optional helical factor kappa, at n_z axial stations. Each station's depth
d_i is an integrated state. The physics is burnfront.burn3d.axial.station_eval, shared with the numpy
drivers, so engine and standalone burns use identical math:
  - local flux G(z) = (mdot_ox + fuel added upstream)/A_port(z)  ('total') or mdot_ox/A_port(z)
  - rdot = phi * 1e-3 * a * G^n * L^m along the surface normal
  - fuel generation rho * sum_z w P_eff rdot (mass-consistent with the tabulated volume)
"""
from functools import partial

import numpy as np
import jax.numpy as jnp

from burnfront.core import store_x, make_part
from burnfront.burn3d import measure
from burnfront.burn3d.axial import station_eval
from burnfront.burn3d.law import RegressionLaw


def _station_idx(xmap, prefix, nz):
    return jnp.array([xmap[f'{prefix}{i}'] for i in range(nz)])


def d_axial_grain(s, x, xmap, nz, n, flux, has_kappa):
    mdot_inj = x[xmap['tnk_mdot_inj']]
    i_d = _station_idx(xmap, 'grn_d', nz)
    d = x[i_d]
    ev = station_eval(s['grn_tab_A'], s['grn_tab_M'], s['grn_dd'], d, mdot_inj, s['grn_z'], s['grn_w'],
                      s['grn_k'], n, s['grn_rho'], flux, s['grn_tab_kappa'] if has_kappa else None, xp=jnp)
    w, L = s['grn_w'], s['grn_L']
    V = jnp.sum(w*(s['grn_A_env'] - ev['A']))          # fuel volume
    mdot = ev['mdot_f']
    Vdot = mdot / s['grn_rho']
    OF = jnp.where(mdot > 0.0, mdot_inj/jnp.where(mdot > 0.0, mdot, 1.0), 0.0)

    x = x.at[_station_idx(xmap, 'grn_ddot', nz)].set(ev['ddot'])
    x = x.at[_station_idx(xmap, 'grn_G', nz)].set(ev['G'])
    x = x.at[_station_idx(xmap, 'grn_rdot', nz)].set(ev['rdot'])
    x = store_x(x, xmap,
                grn_V=V, grn_Vdot=Vdot, grn_mdot=mdot, cmbr_OF=OF,
                grn_A=V/L,                                   # mean solid cross-section (legacy meaning)
                grn_A_port=jnp.sum(w*ev['A'])/L,             # mean open port area
                grn_Ab=jnp.sum(w*ev['P']),                   # burning surface area
                grn_G_head=ev['G'][0], grn_G_aft=ev['G'][-1],
                grn_mdot_exit=ev['mdot_z'][-1])
    return x


def make_axial_grain(A_tab, M_tab, dd, A_env, L, OD, law=None, kappa=None, **kwargs):
    """Tabulated axial grain part.

    A_tab, M_tab: (n_z, K) tables; A_env: (n_z,) or scalar case area; L: grain length; OD: case
    diameter (used for the default chamber volume); law: RegressionLaw (default ABS, total flux).
    """
    law = RegressionLaw() if law is None else law
    A_tab = np.asarray(A_tab, dtype=float)
    nz = A_tab.shape[0]
    z = np.linspace(0.0, L, nz)
    s = {
        'OD': OD, 'L': L, 'rho': law.rho, 'k': law.k(L), 'dd': float(dd),
        'tab_A': jnp.asarray(A_tab), 'tab_M': jnp.asarray(M_tab),
        'tab_kappa': jnp.asarray(np.ones_like(A_tab) if kappa is None else np.broadcast_to(kappa, A_tab.shape)),
        'z': jnp.asarray(z), 'w': jnp.asarray(measure.trapz_weights(z)),
        'A_env': jnp.asarray(np.broadcast_to(np.asarray(A_env, dtype=float), (nz,)).copy()),
    }
    x = {'A': 0.0, 'A_port': 0.0, 'V': 0.0, 'Vdot': 0.0, 'mdot': 0.0, 'Ab': 0.0,
         'G_head': 0.0, 'G_aft': 0.0, 'mdot_exit': 0.0}
    for i in range(nz):
        x[f'd{i}'] = 0.0
        x[f'G{i}'] = 0.0
        x[f'rdot{i}'] = 0.0
    return make_part(
        s=s, x=x, req_s=['OD', 'L'], req_x=[],
        dx={f'd{i}': f'ddot{i}' for i in range(nz)},
        typename='grn',
        fderiv=partial(d_axial_grain, nz=nz, n=law.n, flux=law.flux, has_kappa=kappa is not None),
        **kwargs,
    )


def make_section_grain(port, OD, L, n_z=21, law=None, n_xy=256, twist_rate=0.0,
                       helix_radius=0.0, helix_pitch=None, **kwargs):
    """2D fast mode: grain of one cross-section ``port`` (burnfront.burn3d.shapes.Shape) extruded over L.

    twist_rate / helix_radius+helix_pitch enable the optional helical 2D approximation.
    Returns (part, section, table, kappa).
    """
    from burnfront.burn3d.section import Section2D
    sec = Section2D(port, OD, n_xy=n_xy)
    tab = sec.table()
    kap = None
    if twist_rate or helix_radius:
        kap = sec.helical_kappa(tab, twist_rate=twist_rate, helix_radius=helix_radius, helix_pitch=helix_pitch)
    A, M = tab.stations(n_z)
    part = make_axial_grain(A, M, tab.dd, tab.A_env, L, OD, law=law, kappa=kap, **kwargs)
    return part, sec, tab, kap
