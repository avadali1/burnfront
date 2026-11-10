# Purpose: Model the blowdown of saturated fluids in a simple tank
# Authors: Drew Nickel, Thomas A. Scott

from hrap.core import store_x, make_part, StaticVar

import jax
import jax.numpy as jnp
from jax.lax import cond

from functools import partial

# Assume that pressure derivative is historical average
def avg_liq_blowdown(x, xmap, dP_dT):
    Pdot = x[xmap['tnk_Pdot_sum']] / x[xmap['tnk_Pdot_N']]
    Tdot = Pdot / dP_dT
    
    return Tdot, Pdot

# Evaporation based liquid blowdown that falls back to average pressure drop if condensation is indicated
def liq_blowdow(mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT):
    m_liq, m_vap = x[xmap['tnk_m_ox_liq']], x[xmap['tnk_m_ox_vap']]
    rho_v, rho_l = ox['rho_v'], ox['rho_l']
    
    # Find evap cooling rate, using total oxidizer mass since we consider thermal equilibrium
    A = (m_liq*drho_l__dT/(rho_l**2) + m_vap*drho_v__dT/(rho_v**2)) / (1/rho_v-1/rho_l)
    B = -mdot_ox/(rho_l/rho_v-1)
    C = -ox['Hv'] / ((m_liq+m_vap)*ox['Cp'])
    Tdot = B*C / (1-A*C)
    mdot_evap = Tdot*A+B
    
    Tdot, Pdot = cond(
        mdot_evap > 0.0,
        lambda x, xmap, Tdot, dP_dT:
            (Tdot, Tdot * dP_dT),
        lambda x, xmap, Tdot, dP_dT:
            avg_liq_blowdown(x, xmap, dP_dT),
        x, xmap, Tdot, dP_dT
    )
    
    return Tdot, Pdot

# Assume vapor remains along saturation line, which has been experimentally validated for nitrous oxide.
# Model (as in legacy HRAP): T(m) solves T = T1 (Z(T)/Z1 * m/m1)^0.3. Its exact time derivative by
# implicit differentiation at m = m1:
#     dT/dm = 0.3 (T/m) / (1 - 0.3 T Z'(T)/Z),   Tdot = dT/dm * mdot_ox
# This is the delta -> 0 limit of the legacy finite-difference/fixed-point version (kept below as
# sat_vap_blowdown_fd), whose fixed-point tolerance (1e-9 in Z) divided by delta = 1e-7 injected
# ~1 K/s of step-to-step noise into Tdot.
def sat_vap_blowdown(T, m_ox, mdot_ox, ox, dP_dT, get_sat_props):
    Z = ox['Z']
    dZ_dT = jax.grad(lambda T: get_sat_props(T)['Z'])(T)
    dT_dm = 0.3*(T/m_ox) / (1.0 - 0.3*T*dZ_dT/Z)
    Tdot = dT_dm*mdot_ox
    Pdot = Tdot * dP_dT

    return Tdot, Pdot

def sat_vap_blowdown_fd(T, m_ox, mdot_ox, ox, dP_dT, get_sat_props):
    delta = 1E-7 # FD step
    m_2__m_1 = (m_ox + mdot_ox*delta) / m_ox
    
    # TODO: precompute high order curve-fit using complex step results?
    def Z_body(args):
        eps, Z_i, Z_1, T_i, T_1, m_2__m_1 = args
        
        T_new = T_1*pow(Z_i/Z_1 * m_2__m_1, 0.3)
        Z_new = get_sat_props(T_new)['Z']
        
        # Get error and force convergence
        eps = jnp.abs(Z_i - Z_new)
        Z_new = (Z_i + Z_new) / 2
        
        return eps, Z_new, Z_1, T_new, T_1, m_2__m_1
    
    res = jax.lax.while_loop(lambda val: jnp.abs(val[0]) > 1E-9, Z_body, (1.0, ox['Z'], ox['Z'], T, T, m_2__m_1))
    # jax.debug.print('{a}', a=res[0])
    T_2 = res[3]
    
    Tdot = (T_2 - T) / delta
    Pdot = Tdot * dP_dT
    # jax.debug.print("TANK Debug {x} {y} {c} {d} {e}", x=Tdot, y=Pdot, c=m_ox, d=mdot_ox, e=dP_dT)
    
    return Tdot, Pdot

def d_sat_tank(s, x, xmap, get_sat_props):
    _sat_vap_blowdown = partial(sat_vap_blowdown, get_sat_props=get_sat_props)
    
    m_ox = x[xmap['tnk_m_ox']]
    T    = x[xmap['tnk_T']]
    inj_CdA = s['tnk_inj_CdA']
    inj_N   = s['tnk_inj_N']
    vnt_CdA = s['tnk_vnt_CdA']
    tnk_V   = s['tnk_V']
    
    # Find oxidizer thermophysical properties
    ox = get_sat_props(T)
    rho_v, rho_l = ox['rho_v'], ox['rho_l']
    
    # Use analytical derivative to get saturation pressure derivative
    dP_dT = jax.grad(lambda T: get_sat_props(T)['Pv'])(T)
    # Get saturation density derivatives w.r.t. temperature
    drho_v__dT = jax.grad(lambda T: get_sat_props(T)['rho_v'])(T)
    drho_l__dT = jax.grad(lambda T: get_sat_props(T)['rho_l'])(T)
    # dox_dT = jax.grad(get_sat_props)(T)
    
    # Get mass of oxidizer currently in the phases
    m_liq = jnp.maximum((tnk_V - (m_ox/rho_v))/ ((1/rho_l)-(1/rho_v)), 0.0)
    # Once the liquid is depleted the tank stays in the vapor-only phase (latched in u_sat_tank).
    # The vapor model cools slightly faster than the saturation line, so without the latch the
    # inferred liquid mass flickers between 0 and ~1e-4 kg and the blowdown model alternates every
    # time step (Tdot jumping between the vapor and liquid branches).
    m_liq = jnp.where(x[xmap['tnk_vap_only']] > 0.5, 0.0, m_liq)
    m_vap = m_ox - m_liq
    x = store_x(x, xmap, tnk_m_ox_liq=m_liq, tnk_m_ox_vap=m_vap) # Store here because needed in blowdown funcs
    x = store_x(x, xmap, tnk_rho_ox_liq=rho_l, tnk_rho_ox_vap=rho_v)
    
    Pc = x[xmap['cmbr_P']]
    Pt = ox['Pv']
    Pa = s['Pa']
    
    # Find oxidizer mass flow rate
    
    dP = Pt - Pc
    
    Mcc  = jnp.sqrt(ox['Z']*1.31*188.91*T*(Pc/Pt)**(0.31/1.31))
    Matm = jnp.sqrt(ox['Z']*1.31*188.91*T*(Pa/Pt)**(0.31/1.31))
    
    Mcc  = jnp.minimum(Mcc,  1)
    Matm = jnp.minimum(Matm, 1)
    dP   = jnp.maximum(dP,   0)
    
    # Get vented vapor mass flow rate
    mdot_vnt = cond(
        s['tnk_vnt_S'] == 0,
        lambda *args:
            0.0,
        lambda vnt_CdA, T, Z, Matm:
            (vnt_CdA*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(Z*188.91))*Matm*(1+(0.31)/2*Matm**2)**(-2.31/0.62),
        vnt_CdA, T, ox['Z'], Matm
    )
    
    # Select injector models at compile time
    def inj_liq_model(m_vap, inj_NCdA, T, Z, Mcc, rho_l, rho_v, dP):
        if s['tnk_inj_liq_model'] == 'Incompressible':
            return inj_NCdA*jnp.sqrt(2*rho_l*dP)
    def inj_vap_model(inj_NCdA, T, Z, Mcc, dP, rho_v):
        if s['tnk_inj_vap_model'] == 'Incompressible':
            return inj_NCdA*jnp.sqrt(2*rho_v*dP)
        elif s['tnk_inj_vap_model'] == 'Real Gas':
            return (inj_NCdA*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(Z*188.91))*Mcc*(1+(0.31)/2*Mcc**2)**(-2.31/0.62)

    # Get injected vapor or liquid oxidizer mass flow rate
    mdot_inj = cond(
        m_liq <= 1E-3,
        lambda m_vap, inj_NCdA, T, Z, Mcc, rho_l, rho_v, dP: cond(
            m_vap <= 0.0,
            lambda *vargs: 0.0,
            lambda *vargs: inj_vap_model(*vargs),
            inj_NCdA, T, ox['Z'], Mcc, dP, rho_v
        ),
        lambda *vargs: inj_liq_model(*vargs),
        m_vap, inj_N*inj_CdA, T, ox['Z'], Mcc, rho_l, rho_v, dP
    )
    # jax.debug.print("INJ Debug {a} {b} {x} {y} {c} {d} {e} {f}", a=m_liq, b=m_vap, x=Pt, y=T, c=ox['Z'], d=Mcc, e=mdot_inj, f=(inj_CdA*inj_N*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(ox['Z']*188.91))*Mcc*(1+(0.31)/2*Mcc**2)**(-2.31/0.62))
    
    # Total loss rate of oxidizer = base injected rate + vent rate
    mdot_ox = -(mdot_inj + mdot_vnt)
    
    # Add vent flow rate to injector rate if plumbed to chamber
    mdot_inj = cond(
        s['tnk_vnt_S'] != 2,
        lambda mdot_inj, mdot_vnt:
            mdot_inj,
        lambda mdot_inj, mdot_vnt:
            mdot_inj + mdot_vnt,
        mdot_inj, mdot_vnt
    )
    
    # Get temperature and pressure rates at various stages of blowdown
    Tdot, Pdot = cond(m_ox <= 0.0, lambda *args: (0.0, 0.0),
        lambda *args: cond(m_liq > 0.0,
            lambda T, m_ox, mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT:
                liq_blowdow(mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT),
            lambda T, m_ox, mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT:
                _sat_vap_blowdown(T, m_ox, mdot_ox, ox, dP_dT),
            *args
        ),
        T, m_ox, mdot_ox, x, xmap, ox, dP_dT, drho_v__dT, drho_l__dT
    )
    
    # Below the valid saturation-table range, extrapolated properties are meaningless (and go NaN).
    # At the lower table bound the vapor pressure is ~ambient for supported oxidizers, so treat the
    # tank as depleted: no further outflow and no further cooling.
    T_lo = getattr(get_sat_props, 'T_min', None)
    if T_lo is not None:
        depleted = T <= T_lo
        mdot_inj = jnp.where(depleted, 0.0, mdot_inj)
        mdot_vnt = jnp.where(depleted, 0.0, mdot_vnt)
        mdot_ox  = jnp.where(depleted, 0.0, mdot_ox)
        Tdot     = jnp.where(depleted & (Tdot < 0.0), 0.0, Tdot)
        Pdot     = jnp.where(depleted & (Pdot < 0.0), 0.0, Pdot)

    # Store state and derivatives
    x = store_x(x, xmap, tnk_P=Pt, tnk_Tdot=Tdot, tnk_Pdot=Pdot, tnk_mdot_ox=mdot_ox, tnk_mdot_inj=mdot_inj, tnk_mdot_vnt=mdot_vnt)
    
    return x

# TODO: remove
def p_sat_tank(s, x0, xmap):

    # x0 = store_x(x0, xmap, cmbr_V=V)

    return x0

def u_sat_tank(s, x, xmap):
    x = store_x(x, xmap,
        # Limit to reasonable values
        tnk_m_ox = jnp.maximum(x[xmap['tnk_m_ox']], 0.0),
        # Record needed for avg pressure drop blowdown model
        tnk_Pdot_sum = x[xmap['tnk_Pdot_sum']] + x[xmap['tnk_Pdot']],
        tnk_Pdot_N = x[xmap['tnk_Pdot_N']] + 1,
        tnk_vap_only = jnp.where(x[xmap['tnk_m_ox_liq']] <= 0.0, 1.0, x[xmap['tnk_vap_only']]),
    )

    return x

def make_sat_tank(get_sat_props, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'V': 0.0,
            'vnt_S': 0,
            'vnt_CdA': 0.0,
            'inj_CdA': 0.0,
            'inj_N': 1,
            'inj_vap_model': StaticVar('Incompressible'),
            'inj_liq_model': StaticVar('Incompressible'),
        },
        x = {
            'T':   293.0,
            'm_ox': 1.0,
            
            # Calculated variables
            'Pdot': 0.0,
            'mdot_inj': 0.0,
            'mdot_vnt': 0.0,
            'P':   101e3,
            'm_ox_liq': 0.0,
            'm_ox_vap': 0.0,
            'rho_ox_liq': 0.0,
            'rho_ox_vap': 0.0,
            'Pdot_sum': 0.0,
            'Pdot_N': 0,
            'vap_only': 0.0,
        },
        
        # Required and integrated variables
        req_s = [ 'V', 'inj_CdA' ],
        req_x = [ 'm_ox' ],
        
        # State name to derivative name for integrated variables
        dx = { 'm_ox': 'mdot_ox', 'T': 'Tdot' },

        # Designation and associated functions
        typename = 'tnk',
        fpreprs = p_sat_tank,
        fderiv  = partial(d_sat_tank, get_sat_props=get_sat_props),
        fupdate = u_sat_tank,

        # The user-specified s or x entries
        **kwargs,
    )


# --------------------------------------------------------------------------------------------------
# Equilibrium (thermodynamically consistent) saturated tank
# --------------------------------------------------------------------------------------------------
# The tank holds saturated two-phase fluid in equilibrium at T. With total mass m and volume V:
#     m_l = (V - m v_v(T)) / (v_l(T) - v_v(T)),   U(T, m) = m_l u_l(T) + (m - m_l) u_v(T)
# Energy (adiabatic walls, rigid tank):  dU/dt = -mdot_inj h_out - mdot_vnt h_v
#     h_out = h_l while bulk liquid feeds the injector; afterwards the outflow is the homogeneous
#     tank mixture, h_out = (U + P V)/m (N2O is a "wet" fluid: the expanding vapor stays inside the
#     dome, so the tank remains two-phase with a vanishing liquid fraction).
# Then  Tdot = (dU/dt - dU/dm * mdot) / dU/dT  with exact partial derivatives (autodiff of the
# property splines). Liquid run-out is latched (tnk_vap_only). Injector models for liquid feed:
#   'Incompressible' (SPI): CdA sqrt(2 rho_l dP)
#   'HEM'                 : CdA G_HEM(T, Pc)   (isentropic homogeneous equilibrium, choked)
#   'Dyer'                : CdA [k/(1+k) SPI + 1/(1+k) HEM],  k = sqrt((P1 - Pc)/(Pv - Pc)) = 1 for a
#                           self-pressurized tank (P1 = Pv)
# Mixture/vapor feed: 'Incompressible' with the tank mixture density m/V, or 'Real Gas' (legacy).

def _eq_liquid_mass(T, m, V, get_sat_props):
    p = get_sat_props(T)
    return (V - m/p['rho_v'])/(1.0/p['rho_l'] - 1.0/p['rho_v'])


def _eq_energy(T, m, V, get_sat_props):
    p = get_sat_props(T)
    ml = _eq_liquid_mass(T, m, V, get_sat_props)
    return ml*p['u_l'] + (m - ml)*p['u_v']


def d_eq_tank(s, x, xmap, get_sat_props, get_hem):
    m = x[xmap['tnk_m_ox']]
    T = x[xmap['tnk_T']]
    V = s['tnk_V']
    NCdA = s['tnk_inj_N']*s['tnk_inj_CdA']
    p = get_sat_props(T)
    rho_l, rho_v = p['rho_l'], p['rho_v']
    Pt = p['Pv']
    Pc = x[xmap['cmbr_P']]
    Pa = s['Pa']
    dP = jnp.maximum(Pt - Pc, 0.0)

    ml = _eq_liquid_mass(T, m, V, get_sat_props)
    latched = x[xmap['tnk_vap_only']] > 0.5
    bulk = (~latched) & (ml > 0.0)
    U = _eq_energy(T, m, V, get_sat_props)
    dU_dT, dU_dm = jax.grad(_eq_energy, argnums=(0, 1))(T, m, V, get_sat_props)
    dP_dT = jax.grad(lambda T: get_sat_props(T)['Pv'])(T)

    # Liquid feed
    spi = NCdA*jnp.sqrt(2.0*rho_l*dP)
    model = s['tnk_inj_liq_model']
    if model == 'Incompressible':
        m_liq_feed = spi
    elif model in ('HEM', 'Dyer'):
        if get_hem is None:
            raise ValueError(f"injector model {model.v!r} needs get_hem (hrap.fluid.bake_hem_coolprop)")
        hem = NCdA*get_hem(T, Pc)*jnp.where(dP > 0.0, 1.0, 0.0)
        m_liq_feed = hem if model == 'HEM' else 0.5*(spi + hem)
    else:
        raise ValueError(f'unknown liquid injector model {model}')
    # Mixture/vapor feed
    rho_mix = m/V
    if s['tnk_inj_vap_model'] == 'Real Gas':
        Mcc = jnp.minimum(jnp.sqrt(p['Z']*1.31*188.91*T*(Pc/Pt)**(0.31/1.31)), 1.0)
        m_vap_feed = (NCdA*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(p['Z']*188.91))*Mcc*(1 + 0.31/2*Mcc**2)**(-2.31/0.62)
    else:
        m_vap_feed = NCdA*jnp.sqrt(2.0*rho_mix*dP)
    mdot_inj = jnp.where(bulk, m_liq_feed, m_vap_feed)
    mdot_inj = jnp.where(m > 0.0, mdot_inj, 0.0)

    # Vent (vapor), legacy flow formula
    Matm = jnp.minimum(jnp.sqrt(p['Z']*1.31*188.91*T*(Pa/Pt)**(0.31/1.31)), 1.0)
    mdot_vnt = jnp.where(s['tnk_vnt_S'] == 0, 0.0,
                         (s['tnk_vnt_CdA']*Pt/jnp.sqrt(T))*jnp.sqrt(1.31/(p['Z']*188.91))*Matm*(1 + 0.31/2*Matm**2)**(-2.31/0.62))

    h_out = jnp.where(bulk, p['h_l'], (U + Pt*V)/jnp.maximum(m, 1e-12))
    dU = -mdot_inj*h_out - mdot_vnt*p['h_v']
    mdot_ox = -(mdot_inj + mdot_vnt)
    Tdot = (dU - dU_dm*mdot_ox)/dU_dT
    Pdot = dP_dT*Tdot

    T_lo = getattr(get_sat_props, 'T_min', None)
    if T_lo is not None:
        depleted = T <= T_lo
        mdot_inj = jnp.where(depleted, 0.0, mdot_inj)
        mdot_vnt = jnp.where(depleted, 0.0, mdot_vnt)
        mdot_ox = jnp.where(depleted, 0.0, mdot_ox)
        Tdot = jnp.where(depleted & (Tdot < 0.0), 0.0, Tdot)
        Pdot = jnp.where(depleted & (Pdot < 0.0), 0.0, Pdot)

    mdot_inj = jnp.where(s['tnk_vnt_S'] == 2, mdot_inj + mdot_vnt, mdot_inj)

    x = store_x(x, xmap, tnk_P=Pt, tnk_Tdot=Tdot, tnk_Pdot=Pdot, tnk_mdot_ox=mdot_ox, tnk_mdot_inj=mdot_inj,
                tnk_mdot_vnt=mdot_vnt, tnk_m_ox_liq=jnp.maximum(ml, 0.0), tnk_m_ox_vap=m - jnp.maximum(ml, 0.0),
                tnk_rho_ox_liq=rho_l, tnk_rho_ox_vap=rho_v, tnk_U=U, tnk_h_out=h_out)
    return x


def u_eq_tank(s, x, xmap):
    return store_x(x, xmap,
        tnk_m_ox=jnp.maximum(x[xmap['tnk_m_ox']], 0.0),
        tnk_vap_only=jnp.where(x[xmap['tnk_m_ox_liq']] <= 0.0, 1.0, x[xmap['tnk_vap_only']]),
    )


def make_eq_tank(get_sat_props, get_hem=None, **kwargs):
    """Equilibrium two-phase tank (see above). get_sat_props from hrap.fluid.bake_sat_coolprop (needs
    u_l, u_v, h_l, h_v); get_hem from hrap.fluid.bake_hem_coolprop for the 'HEM'/'Dyer' injectors."""
    return make_part(
        s = {
            'V': 0.0,
            'vnt_S': 0,
            'vnt_CdA': 0.0,
            'inj_CdA': 0.0,
            'inj_N': 1,
            'inj_vap_model': StaticVar('Incompressible'),
            'inj_liq_model': StaticVar('Dyer'),
        },
        x = {
            'T': 293.0, 'm_ox': 1.0,
            'Pdot': 0.0, 'mdot_inj': 0.0, 'mdot_vnt': 0.0, 'P': 101e3,
            'm_ox_liq': 0.0, 'm_ox_vap': 0.0, 'rho_ox_liq': 0.0, 'rho_ox_vap': 0.0,
            'vap_only': 0.0, 'U': 0.0, 'h_out': 0.0,
        },
        req_s = ['V', 'inj_CdA'],
        req_x = ['m_ox'],
        dx = {'m_ox': 'mdot_ox', 'T': 'Tdot'},
        typename = 'tnk',
        fpreprs = p_sat_tank,
        fderiv  = partial(d_eq_tank, get_sat_props=get_sat_props, get_hem=get_hem),
        fupdate = u_eq_tank,
        **kwargs,
    )
