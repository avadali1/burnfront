# Purpose: Provide throat mass flow rate and nozzle exit conditions (using frozen equilibrium from chamber)
# Authors: Drew Nickel, Thomas A. Scott
"""Converging-diverging nozzle, quasi-1D isentropic flow of the chamber gas (gamma, R, T from the
chamber model).

Mass flow (mass_flow): choked  mdot = Cd At P Gamma(g)/sqrt(R T); below the choking pressure ratio
the nozzle is a subsonic venturi with exit pressure Pa, mdot = Cd Ae P sqrt(2g/((g-1)RT) (r^(2/g) - r^((g+1)/g))),
r = Pa/P. mdot = min(choked, subsonic) is continuous and exact for both regimes.

Thrust: F = eff * [mdot u_e + (p_e - Pa) A_e]
  attached supersonic flow: p_e from the exit area ratio ER;
  separated flow (Summerfield criterion p_wall < sep_ratio*Pa, default 0.4): the jet leaves the wall
  where p = sep_ratio*Pa, the rest of the divergent section is at ~Pa;
  unchoked: subsonic exit at Pa.
Expansion models (s['noz_expansion']):
  'frozen'      : constant-gamma isentropic expansion of the chamber gas (gamma, R, T from the chamber)
  'equilibrium' : shifting-equilibrium expansion from hrap.expansion tables (Cantera, validated against
                  NASA CEA): exit/separation velocity u = U(OF, Pc, p/p*) c*, area ratio and throat
                  pressure from the same tables, choked flow with the equilibrium c*. Blended with the
                  frozen result by the chamber's combustion-product mass fraction (startup air,
                  cold oxidizer flow), and the frozen subsonic formulas when unchoked.
"""
from hrap.core import store_x, make_part, StaticVar

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import cond

from functools import partial


def eq_lookup(s, OF, Pc):
    """Bilinear (O/F, Pc) lookup in the expansion tables, clamped to the grid.
    Returns (U row, ln-eps row, r_throat, c*_eq/c*_gamma)."""
    # jnp.asarray: tables may arrive as numpy constants (closure) or traced arrays (solver carry)
    gOF, gPc = jnp.asarray(s['noz_eq_OF']), jnp.asarray(s['noz_eq_Pc'])

    def w(g, v):
        v = jnp.clip(v, g[0], g[-1])
        i = jnp.clip(jnp.searchsorted(g, v) - 1, 0, g.size - 2)
        return i, (v - g[i])/(g[i + 1] - g[i])
    i, a = w(gOF, OF)
    j, b = w(gPc, Pc)

    def bil(T):
        T = jnp.asarray(T)
        return ((1 - a)*(1 - b)*T[i, j] + a*(1 - b)*T[i + 1, j] + (1 - a)*b*T[i, j + 1] + a*b*T[i + 1, j + 1])
    return bil(s['noz_eq_U']), bil(s['noz_eq_LE']), bil(s['noz_eq_rt']), bil(s['noz_eq_fc'])


def gamma_fn(g):
    """Vandenkerckhove function Gamma(g) = sqrt(g) (2/(g+1))^((g+1)/(2(g-1)))."""
    return jnp.sqrt(g)*(2.0/(g + 1.0))**((g + 1.0)/(2.0*(g - 1.0)))


def _subsonic_flux(r, g):
    rc = (2.0/(g + 1.0))**(g/(g - 1.0))
    r = jnp.clip(r, rc, 1.0)
    return jnp.sqrt(jnp.maximum(2.0*g/(g - 1.0)*(r**(2.0/g) - r**((g + 1.0)/g)), 0.0))


def mass_flow(P, Pa, T, R, g, Cd, D_thrt, ER, return_choked=False, cstar_factor=1.0):
    """cstar_factor: c*_actual / c*_constant-gamma (1 for frozen; equilibrium correction otherwise)."""
    At = jnp.pi/4*D_thrt**2
    sq = jnp.sqrt(R*T)
    m_ch = Cd*At*P*gamma_fn(g)/sq/cstar_factor
    m_sub = Cd*ER*At*P*_subsonic_flux(Pa/P, g)/sq
    m = jnp.where(P > Pa, jnp.minimum(m_ch, m_sub), 0.0)
    if return_choked:
        return m, m_ch <= m_sub
    return m


def area_ratio(M, g):
    return (1.0/M)*((2.0/(g + 1.0))*(1.0 + 0.5*(g - 1.0)*M**2))**((g + 1.0)/(2.0*(g - 1.0)))


# Solve for exit mach given specific heat ratio and exit ratio
def M_solve(k, ER):
    def get_error(Me, k, ER):
        error = ((k+1)/2)**(-(k+1)/ \
            (2*(k-1)))*(1+(k-1)/2*Me**2)** \
            ((k+1)/(2*(k-1)))/Me- \
            ER

        return error
    get_derror_dMe = jax.grad(get_error)

    def newton_step(val):
        _, Me, k, ER, i = val
        error = get_error(Me, k, ER)
        Me -= error / get_derror_dMe(Me, k, ER)
        return (error, Me, k, ER, i+1)
    result = jax.lax.while_loop(lambda val: (jnp.abs(val[0]) > 1E-12) & (val[4]<50), newton_step, (1.0, 3.0, k, ER, 0))

    return result[1]


def d_cd_nozzle(s, x, xmap):
    Pc = x[xmap['cmbr_P']]
    Pa = s['Pa']
    g = x[xmap['cmbr_k']]
    R = x[xmap['cmbr_R']]
    T = x[xmap['cmbr_T']]
    At = np.pi/4*s['noz_thrt']**2
    ER = s['noz_ER']
    fc = x[xmap['cmbr_cstar_fac']] if 'cmbr_cstar_fac' in xmap else 1.0
    mdot, choked = mass_flow(Pc, Pa, T, R, g, s['noz_Cd'], s['noz_thrt'], ER, return_choked=True, cstar_factor=fc)
    cp = g*R/(g - 1.0)
    ex = (g - 1.0)/g

    def u_at(p):
        return jnp.sqrt(jnp.maximum(2.0*cp*T*(1.0 - (p/Pc)**ex), 0.0))

    # Attached supersonic expansion to ER
    Me = M_solve(g, ER)
    Pe = Pc*(1 + 0.5*(g - 1)*Me**2)**(-g/(g - 1))
    F_att = mdot*u_at(Pe) + (Pe - Pa)*ER*At
    # Separated: jet leaves the wall where p = sep_ratio*Pa (Summerfield), M >= 1
    p_s = jnp.minimum(s['noz_sep_ratio']*Pa, Pc*(2.0/(g + 1.0))**(g/(g - 1.0)))
    M_s = jnp.sqrt(jnp.maximum(2.0/(g - 1.0)*((Pc/p_s)**ex - 1.0), 1.0))
    eps_s = area_ratio(M_s, g)
    F_sep = mdot*u_at(p_s) + (p_s - Pa)*eps_s*At
    F_sup = jnp.where(Pe >= s['noz_sep_ratio']*Pa, F_att, F_sep)
    separated = Pe < s['noz_sep_ratio']*Pa
    if s['noz_expansion'] == 'equilibrium':
        # Shifting-equilibrium expansion; c* of the actual chamber gas (includes c* efficiency)
        U, LE, r_t, _ = eq_lookup(s, x[xmap['cmbr_OF']], Pc)
        cs = x[xmap['cmbr_cstar']]
        lnr = jnp.log(r_t*jnp.asarray(s['noz_eq_rho']))
        lnER = jnp.log(ER)
        Pe_eq = Pc*jnp.exp(jnp.interp(lnER, LE, lnr))
        F_att_eq = mdot*cs*jnp.interp(lnER, LE, U) + (Pe_eq - Pa)*ER*At
        p_s_eq = jnp.minimum(s['noz_sep_ratio']*Pa, Pc*r_t)
        lq = jnp.log(p_s_eq/(Pc*r_t))                        # <= 0, decreasing along the table
        lrho = jnp.log(jnp.asarray(s['noz_eq_rho']))
        U_s = jnp.interp(-lq, -lrho, U)
        eps_s = jnp.exp(jnp.interp(-lq, -lrho, LE))
        F_sep_eq = mdot*cs*U_s + (p_s_eq - Pa)*eps_s*At
        sep_eq = Pe_eq < s['noz_sep_ratio']*Pa
        F_eq = jnp.where(sep_eq, F_sep_eq, F_att_eq)
        wp = x[xmap['cmbr_y_burn']]
        F_sup = wp*F_eq + (1.0 - wp)*F_sup
        Pe = wp*Pe_eq + (1.0 - wp)*Pe
        separated = jnp.where(wp > 0.5, sep_eq, separated)
    # Unchoked: subsonic everywhere, exit at Pa
    F_sub = mdot*u_at(jnp.minimum(Pa, Pc))
    thrust = s['noz_eff']*jnp.where(choked, F_sup, F_sub)
    thrust = jnp.where(Pc <= Pa, 0.0, jnp.maximum(thrust, 0.0))

    x = store_x(x, xmap, noz_mdot=mdot, noz_Me=jnp.where(choked, Me, 0.0), noz_Pe=jnp.where(choked, Pe, Pa),
                noz_thrust=thrust, noz_separated=jnp.where(choked & separated, 1.0, 0.0))

    return x

def u_cd_nozzle(s, x, xmap):
    # Limit thrust to positive
    x = store_x(x, xmap,
        noz_thrust = jnp.maximum(x[xmap['noz_thrust']], 0.0),
    )

    return x

def make_cd_nozzle(**kwargs):
    return make_part(
        # Default static parameters
        s = {
            'Cd':    1.0,
            'eff':   1.0,
            'thrt':  None,
            'ER':    None,
            'sep_ratio': 0.4,   # Summerfield separation criterion p_sep/Pa (0 disables separation)
            'expansion': StaticVar('frozen'),   # or 'equilibrium' (needs the eq_* tables, see set_expansion_tables)
            'eq_OF': np.array([0.0, 1.0]), 'eq_Pc': np.array([0.0, 1.0]), 'eq_rho': np.array([1.0, 0.5]),
            'eq_U': np.zeros((2, 2, 2)), 'eq_LE': np.zeros((2, 2, 2)), 'eq_rt': np.ones((2, 2)),
            'eq_fc': np.ones((2, 2)),
        },

        # Default initial variables
        x = {
            'mdot':   0.0,
            'Me':     0.0,
            'Pe':     0.0,
            'thrust': 0.0,
            'separated': 0.0,
        },

        # Required parameters and variables
        req_s = ['thrt', 'ER'],
        req_x = [],

        # State name to derivative name for integrated variables
        dx = { },

        # Designation and associated functions
        typename = 'noz',
        fderiv   = d_cd_nozzle,
        fupdate  = u_cd_nozzle,

        # The user-specified s or x entries
        **kwargs,
    )


def expansion_kwargs(table):
    """make_cd_nozzle keyword arguments for shifting-equilibrium expansion from an
    hrap.expansion.ExpansionTable."""
    return dict(expansion=StaticVar('equilibrium'), eq_OF=np.asarray(table.OF), eq_Pc=np.asarray(table.Pc),
                eq_rho=np.asarray(table.rho), eq_U=np.asarray(table.U), eq_LE=np.asarray(table.LE),
                eq_rt=np.asarray(table.r_t), eq_fc=np.asarray(table.cstar/table.cstar_gamma))
