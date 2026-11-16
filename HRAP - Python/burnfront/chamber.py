# Purpose: Provide chamber dynamics from tabulated chemical equilibrum properties
# Authors: Drew Nickel, Thomas A. Scott (lumped mass/energy model: rewritten for consistency)
"""Lumped combustion chamber: ideal-gas mixture of the initial air and the combustion products.

States: P (chamber pressure), m_g (gas mass), m_air (air mass, the rest are products).
Mass:    dm_g/dt   = mdot_in - mdot_out,   dm_air/dt = -mdot_out * m_air/m_g
Energy:  d/dt[(m_a cv_a + m_p cv_p) T] = mdot_in cp_p T_in - mdot_out cp_mix T - P dV/dt
with P V = (m_a R_a + m_p R_p) T, which gives (C = sum m cv, Rm = sum m R)
    dP/dt = [ (E - Cdot T + C T Rmdot/Rm) Rm/C - P dV/dt ] / V.
For a pure-product chamber this reduces to dP/dt = [gamma R (mdot_in T_in - mdot_out T) - gamma P dV/dt]/V;
in the quasi-steady burn T -> T_in and mdot_out -> mdot_in, so P V/(m R T) holds at every instant
(the legacy dP/P = dm/m - dV/V form kept P V/m at the initial air value, i.e. ~12x too much stored
gas and ~12x too slow pressure transients).

Products: k, M, T_c from the propellant table at (O/F, Pc), clamped to the table (flagged in
cmbr_chem_clamp). Inflow temperature T_in = cstar_eff^2 T_c (c* ~ sqrt(T)), so the steady c* equals
cstar_eff * c*_ideal. With no fuel flow the inflow is unburned oxidizer vapor at tank temperature.
The nozzle mass flow is evaluated here (burnfront.nozzle.mass_flow) from the current gas state, so the
chamber never uses a stale nozzle flow.
"""
from burnfront.core import store_x, make_part
from burnfront.nozzle import mass_flow, gamma_fn, eq_lookup

import jax
import jax.numpy as jnp
from jax.lax import cond

RU = 8314.46
R_AIR, GAM_AIR, T_AMB = RU/28.965, 1.4, 294.0
CV_AIR = R_AIR/(GAM_AIR - 1.0)
OX_M, OX_GAM = 44.013, 1.27   # N2O vapor (cold-flow inflow)


def chem_lookup(s, OF, Pc):
    ig = s['chem_interp_k'].grid
    lo0, hi0, lo1, hi1 = ig[0][0], ig[0][-1], ig[1][0], ig[1][-1]
    of_c = jnp.clip(OF, lo0, hi0)
    pc_c = jnp.clip(Pc, lo1, hi1)
    ip = jnp.stack([of_c, pc_c])[None, :]
    k = s['chem_interp_k'](ip)[0]
    M = s['chem_interp_M'](ip)[0]
    T = s['chem_interp_T'](ip)[0]
    # bit 1: O/F outside the table, bit 2: Pc outside the table
    flag = jnp.where((OF < lo0) | (OF > hi0), 1.0, 0.0) + jnp.where((Pc < lo1) | (Pc > hi1), 2.0, 0.0)
    return k, M, T, flag


def d_chamber(s, x, xmap):
    P = x[xmap['cmbr_P']]
    m = jnp.maximum(x[xmap['cmbr_m_g']], 1e-12)
    ma = jnp.clip(x[xmap['cmbr_m_air']], 0.0, m)
    mp = m - ma
    V = x[xmap['cmbr_V0']] - x[xmap['grn_V']]
    Vdot = x[xmap['grn_Vdot']]
    mdot_f = x[xmap['grn_mdot']]
    mdot_in = mdot_f + x[xmap['tnk_mdot_inj']]
    OF = x[xmap['cmbr_OF']]

    k_c, M_c, T_c, flag = chem_lookup(s, OF, P)
    burning = mdot_f > 0.0
    k_p = jnp.where(burning, k_c, OX_GAM)
    R_p = RU/jnp.where(burning, M_c, OX_M)
    T_in = jnp.where(burning, s['cmbr_cstar_eff']**2*T_c, x[xmap['tnk_T']])
    flag = jnp.where(burning, flag, 0.0)
    cv_p = R_p/(k_p - 1.0)
    cp_p = cv_p + R_p

    Rm = ma*R_AIR + mp*R_p
    C = ma*CV_AIR + mp*cv_p
    T = P*V/Rm
    R_mix = Rm/m
    gam = (C + Rm)/C

    # Equilibrium expansion: choked flow with the equilibrium c* (c*_eq/c*_gamma from the expansion
    # tables), weighted by the combustion-product mass fraction
    y_burn = jnp.where(burning, mp/m, 0.0)
    if s['noz_expansion'] == 'equilibrium':
        fc = 1.0 + y_burn*(eq_lookup(s, OF, P)[3] - 1.0)
    else:
        fc = jnp.ones_like(P)
    mdot_out = mass_flow(P, s['Pa'], T, R_mix, gam, s['noz_Cd'], s['noz_thrt'], s['noz_ER'], cstar_factor=fc)
    ya, yp = ma/m, mp/m
    mdot_a = -mdot_out*ya
    mdot_p = mdot_in - mdot_out*yp
    E = mdot_in*cp_p*T_in - mdot_out*(ya*(CV_AIR + R_AIR) + yp*cp_p)*T - P*Vdot
    Cdot = mdot_a*CV_AIR + mdot_p*cv_p
    Rmdot = mdot_a*R_AIR + mdot_p*R_p
    Pdot = ((E - Cdot*T + C*T*Rmdot/Rm)*Rm/C - P*Vdot)/V
    Pdot = jnp.where((P <= s['Pa']) & (Pdot < 0.0), 0.0, Pdot)

    cstar = jnp.sqrt(R_mix*T)/gamma_fn(gam)*fc
    x = store_x(x, xmap, cmbr_mdot_g=mdot_in - mdot_out, cmbr_mdot_air=mdot_a, cmbr_Pdot=Pdot,
                cmbr_k=gam, cmbr_R=R_mix, cmbr_T=T, cmbr_T_in=T_in, cmbr_cstar=cstar,
                cmbr_chem_clamp=flag, cmbr_cstar_fac=fc, cmbr_y_burn=y_burn, noz_mdot=mdot_out)
    return x


# Preprocessing: default chamber volume = grain envelope
def p_chamber(s, x0, xmap):
    V0 = x0[xmap['cmbr_V0']]
    V0 = cond(V0 == 0.0, lambda Va, Vb: Vb, lambda Va, Vb: Va, V0, s['grn_L']*(jnp.pi/4*s['grn_OD']**2))
    return store_x(x0, xmap, cmbr_V0=V0)


# After the first derivative evaluation (grain volume known): fill the gas volume with ambient air
def i_chamber(s, x0, xmap):
    V_gas = x0[xmap['cmbr_V0']] - x0[xmap['grn_V']]
    m_g0 = s['Pa']*V_gas/(R_AIR*T_AMB)
    return store_x(x0, xmap, cmbr_P=s['Pa'], cmbr_m_g=m_g0, cmbr_m_air=m_g0)


def u_chamber(s, x, xmap):
    m = jnp.maximum(x[xmap['cmbr_m_g']], 0.0)
    return store_x(x, xmap,
        cmbr_m_g=m,
        cmbr_m_air=jnp.clip(x[xmap['cmbr_m_air']], 0.0, m),
        cmbr_P=jnp.maximum(x[xmap['cmbr_P']], s['Pa']),
    )


def make_chamber(**kwargs):
    part = make_part(
        s = {
            'cstar_eff': 1.0,
        },
        x = {
            'V0': 0.0, # Volume of the chamber including the grain envelope; 0 -> grain L * pi/4 OD^2
            'P':   101e3,
            'm_g': 1E-3,
            'm_air': 1E-3,
            # Calculated variables
            'k': 1.4,
            'R': R_AIR,
            'T': T_AMB,
            'T_in': 0.0,
            'cstar': 0.0,
            'chem_clamp': 0.0,
            'cstar_fac': 1.0,
            'y_burn': 0.0,
            'OF': 0.0, # Handled by grain file
        },
        req_s = [],
        req_x = [],
        dx = { 'P': 'Pdot', 'm_g': 'mdot_g', 'm_air': 'mdot_air' },
        typename = 'cmbr',
        fpreprs = p_chamber,
        fderiv  = d_chamber,
        fupdate = u_chamber,
        **kwargs,
    )
    part['fpostinit'] = i_chamber
    return part
