# Purpose: Provide fluid models
# Authors: Thomas A. Scott

import numpy as np
import jax.numpy as jnp
import CoolProp.CoolProp as CP
import interpax

# Wrapper for coolprop to provide JAX compilable property tables
def bake_sat_coolprop(fluid, T_eval):
    """Saturation properties along the dome as JAX PCHIP interpolators of T.

    Pv, rho_l, rho_v, Hv, Cp (vapor), Z (vapor) for the legacy tank model, plus the specific internal
    energies u_l, u_v and enthalpies h_l, h_v used by the equilibrium tank model.
    """
    T_eval = np.asarray(T_eval, dtype=float)
    keys = ['Pv', 'rho_l', 'rho_v', 'Hv', 'Cp', 'Z', 'u_l', 'u_v', 'h_l', 'h_v']
    tab = {k: np.zeros_like(T_eval) for k in keys}
    for i, T in enumerate(T_eval):
        tab['Pv'][i]    = CP.PropsSI('P', 'T', T, 'Q', 0, fluid)
        tab['rho_l'][i] = CP.PropsSI('D', 'T', T, 'Q', 0, fluid)
        tab['rho_v'][i] = CP.PropsSI('D', 'T', T, 'Q', 1, fluid)
        tab['Z'][i]     = CP.PropsSI('Z', 'T', T, 'Q', 1, fluid)
        tab['Cp'][i]    = CP.PropsSI('CPMASS', 'T', T, 'Q', 1, fluid)
        tab['h_l'][i]   = CP.PropsSI('H', 'T', T, 'Q', 0, fluid)
        tab['h_v'][i]   = CP.PropsSI('H', 'T', T, 'Q', 1, fluid)
        tab['u_l'][i]   = CP.PropsSI('U', 'T', T, 'Q', 0, fluid)
        tab['u_v'][i]   = CP.PropsSI('U', 'T', T, 'Q', 1, fluid)
    tab['Hv'] = tab['h_v'] - tab['h_l']

    # Monotone cubic splines (no extrapolation: callers keep T inside [T_min, T_max])
    interp = {k: interpax.PchipInterpolator(T_eval, v) for k, v in tab.items()}

    def get_my_sat_props(T, interp=interp):
        return {k: f(T) for k, f in interp.items()}

    # Expose the valid table range so consumers (e.g. the tank) can avoid extrapolating
    get_my_sat_props.T_min = float(np.min(T_eval))
    get_my_sat_props.T_max = float(np.max(T_eval))
    get_my_sat_props.fluid = fluid

    return get_my_sat_props


def bake_hem_coolprop(fluid, T_eval, n_r=160, r_min=0.01):
    """Homogeneous-equilibrium (HEM) injector mass flux for saturated liquid at T discharging to P2.

    Isentropic expansion from saturated liquid (s1 = s_l(T), h1 = h_l(T)) to P2:
        G(T, P2) = rho2 sqrt(2 (h1 - h2)),   h2 = h(P2, s1), rho2 = rho(P2, s1)
    choked at the maximum over P >= P2 (critical flow). Tabulated over T and r = P2/Pv(T) on a grid
    clustered at r -> 1 where G ~ sqrt(1 - r); returns hem(T, P2) (JAX, bilinear).
    """
    from jax.scipy.interpolate import RegularGridInterpolator
    T_eval = np.asarray(T_eval, dtype=float)
    r = np.sort(1.0 - (1.0 - r_min)*np.linspace(0.0, 1.0, n_r)**2)
    p_trip = CP.PropsSI('ptriple', fluid)
    G = np.zeros((T_eval.size, r.size))
    for i, T in enumerate(T_eval):
        Pv = CP.PropsSI('P', 'T', T, 'Q', 0, fluid)
        s1 = CP.PropsSI('S', 'T', T, 'Q', 0, fluid)
        h1 = CP.PropsSI('H', 'T', T, 'Q', 0, fluid)
        for j, rj in enumerate(r):
            if rj >= 1.0:
                G[i, j] = 0.0
                continue
            P2 = max(rj*Pv, 1.001*p_trip)
            h2 = CP.PropsSI('H', 'P', P2, 'S', s1, fluid)
            d2 = CP.PropsSI('D', 'P', P2, 'S', s1, fluid)
            G[i, j] = d2*np.sqrt(max(2.0*(h1 - h2), 0.0))
        # Critical (choked) flow: below the maximising back pressure the flux stays at the maximum
        G[i] = np.maximum.accumulate(G[i][::-1])[::-1]
    sat_P = interpax.PchipInterpolator(T_eval, [CP.PropsSI('P', 'T', T, 'Q', 0, fluid) for T in T_eval])
    interp = RegularGridInterpolator((T_eval, r), G, method='linear', fill_value=None)

    def hem(T, P2):
        rr = jnp.clip(P2/sat_P(T), r[0], 1.0)
        TT = jnp.clip(T, T_eval[0], T_eval[-1])
        return interp(jnp.stack([TT, rr])[None, :])[0]
    hem.table = (T_eval, r, G)
    return hem
