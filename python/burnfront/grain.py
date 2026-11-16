# Purpose: Model regression of motor grains
# Authors: Thomas A. Scott

import numpy as np
import jax.numpy as jnp
from jax.lax import cond
from functools import partial, reduce

import interpax
from burnfront.core import store_x, make_part
import burnfront.sdf as sdf
import matplotlib.cm as cm  # kept: examples rely on it via `from burnfront.grain import *`


import math


# Default regression law: ABS (burnfront/resources/propellant_configs/ABS.mat, prop_Reg)
#   rdot [m/s] = 1e-3 * a * G^n * L^m, a in mm/s, G in kg/(m^2 s), L in m
ABS_REG = (0.198, 0.325, 0.0)

# Floor on open port area to keep flux finite in degenerate states (m^2)
_A_PORT_MIN = 1e-12

def port_area(s, x, xmap):
    """Open (gas-filled) port cross-section area.

    grn_A is the *solid* fuel cross-section; the oxidizer flows through the
    complement within the case, pi/4*OD^2 - grn_A. Oxidizer mass flux must
    always be evaluated against this open area, never against grn_A.
    """
    return jnp.maximum(jnp.pi/4 * s['grn_OD']**2 - x[xmap['grn_A']], _A_PORT_MIN)

def regression_rate(G, L, Reg, phi=1.0):
    """Marxman-type power law rdot = phi * 1e-3*a * G^n * L^m (m/s); Reg = (a [mm/s], n, m)."""
    return phi * 0.001*Reg[0] * jnp.maximum(G, 0.0)**Reg[1] * L**Reg[2]

def d_grain_constOF(s, x, xmap, d2a):
    mdot_inj = x[xmap['tnk_mdot_inj']] # TODO: using vent?
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    OF = s['grn_OF']

    # Current arc length of exposed grain on the cross section
    arc = d2a(d, s, x, xmap)

    # Current volume
    V = L * A

    # Oxidizer flux through the open port (diagnostic only for this rate law)
    A_port = port_area(s, x, xmap)
    G = mdot_inj / A_port

    # Grain consumption rate (positive)
    mdot = mdot_inj / OF
    mdot = jnp.where(A <= 0.0, 0.0, mdot)

    # Rate of volume consumption (positive)
    Vdot = mdot / rho

    # Rate of cross section area loss
    Adot = Vdot / L

    # Cross sectional area linearization (i.e. volume of thin shell, Adot = arc * ddot)
    ddot = jnp.where(arc > 0.0, Adot / jnp.where(arc > 0.0, arc, 1.0), 0.0)

    # Store result
    x = store_x(x, xmap, grn_Adot=-Adot, grn_ddot=ddot, grn_V=V, grn_mdot=mdot, grn_Vdot=Vdot,
                grn_A_port=A_port, grn_G=G, cmbr_OF=OF)

    return x

def d_grain_shiftOF(s, x, xmap, d2a):
    mdot_inj = x[xmap['tnk_mdot_inj']] # TODO: using vent?
    A = x[xmap['grn_A']]
    d = x[xmap['grn_d']]
    L   = s['grn_L']
    rho = s['grn_rho']
    Reg = s['grn_Reg']
    phi = s['grn_phi']

    # Current arc length of exposed grain on the cross section
    arc = d2a(d, s, x, xmap)
    # Current volume
    V = L * A

    # Oxidizer mass flux through the OPEN port area (not the solid area grn_A)
    A_port = port_area(s, x, xmap)
    G      = mdot_inj / A_port

    # No regression once the web is consumed
    ddot = jnp.where(A <= 0.0, 0.0, regression_rate(G, L, Reg, phi))

    Adot = ddot*arc
    Vdot = Adot*L
    mdot = Vdot*rho

    # O/F is undefined without fuel flow; report 0 like the legacy MATLAB code instead of 0/0 = NaN
    OF = jnp.where(mdot > 0.0, mdot_inj / jnp.where(mdot > 0.0, mdot, 1.0), 0.0)

    # Store result
    x = store_x(x, xmap, grn_Adot=-Adot, grn_ddot=ddot, grn_V=V, grn_mdot=mdot, grn_Vdot=Vdot,
                grn_A_port=A_port, grn_G=G, cmbr_OF=OF)

    return x

def u_grain(s, x, xmap):
    x = store_x(x, xmap,
        grn_A = jnp.maximum(x[xmap['grn_A']], 0.0),
        grn_d = jnp.minimum(x[xmap['grn_d']], s['grn_OD']/2) # TODO: shouldnt be necessary
    )
    
    return x

def make_star_vertices(grn_ID, grn_TD, N_tip):
    grn_r = np.array([grn_ID/2, grn_TD/2]*N_tip)
    grn_t = np.linspace(0.0, 2*np.pi, 2*N_tip, endpoint=False)
    grn_xy = np.stack([grn_r*np.cos(grn_t), grn_r*np.sin(grn_t)], axis=1)
    return grn_xy

def _section_d2a(sec, Nd):
    """Shared 2D baking: exact offset tables of a Section2D sampled at Nd regression distances.

    Returns (A0 solid area, d samples, perimeter samples, table). A0 = pi/4 OD^2 - A_port(0) uses the
    tabulated port area, so the lumped models' A(d) = A0 - int P dd reaches 0 exactly at burnout.
    """
    from burnfront.burn3d.measure import hermite_eval
    tab = sec.table()
    pos = np.nonzero(tab.M > 0)[0]
    d_end = tab.levels[min(pos[-1] + 1, tab.levels.size - 1)] if pos.size else 0.0
    grn_d = np.linspace(0.0, d_end, Nd)
    A_s = np.tile(tab.A, (Nd, 1)); M_s = np.tile(tab.M, (Nd, 1))
    _, P = hermite_eval(A_s, M_s, tab.dd, grn_d)
    A0 = np.pi/4*sec.OD**2 - tab.A[0]
    return A0, grn_d, np.maximum(P, 0.0), tab


def bake_d2a(grn_OD, ports_xy, Nx, Nd):
    """Use description of ports to precompute the regression behavor of the grain

    Uses the same exact-offset level-set tables as the 2D/3D burn models (burnfront.burn3d.section):
    port boundaries are exact polygon signed distances, offset geometry is {phi < d}, and the
    burning perimeter is dA/dd from sub-cell areas clipped by the case.

    Args:
      grn_OD: outer diameter of the grain
      ports_xy: list of or single array of shape (N,2)
        each array can have a different leading dimension,
        each describes a closed loop representing an initial port,
        each cannot intersect itself or others upon specification.
      Nx: spatial resolution (cells across OD)
      Nd: regression resolution
    Returns:
      initial solid cross section area, tabulated regression distance (d2a input), tabulated exposed
      arc length (d2a output), list of contours for plotting
    """
    from burnfront.burn3d.section import Section2D
    if type(ports_xy) not in [list, tuple]: ports_xy = [ports_xy]
    sec = Section2D.from_polygons([np.asarray(p, dtype=float) for p in ports_xy], grn_OD, n_xy=Nx)
    A0, grn_d, grn_d2a, _ = _section_d2a(sec, Nd)
    all_contours = [np.append(p, [p[0, :]], axis=0) for p in ports_xy]
    for d in grn_d[1:]:
        all_contours.extend(sec.contours(d))
    return A0, grn_d, grn_d2a, all_contours

def make_arbitrary_shape(d2a_curve, **kwargs):
    def arbitrary_d2a(d, s, x, xmap, curve):
        return curve(d) #np.pi * (s['grn_shape_ID'] + 2*d)
    
    def preprs(s, x, xmap):
        x = x.at[xmap['grn_A']].set(s['grn_shape_A0'])
        
        return x
    
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'A0': 0.1,
        },
        x = {
        },
        
        # Required and integrated variables
        req_s = ['A0'],
        req_x = [],
        dx    = { },

        typename = 'shape',

        d2a = partial(arbitrary_d2a, curve=d2a_curve),
        fpreprs = preprs,
        
        # The user-specified static and initial dynamic variables
        **kwargs,
    )

# TODO: Helper for equivalent inner and outer diameter?

def make_circle_shape(**kwargs):
    def circle_d2a(d, s, x, xmap):
        return np.pi * (s['grn_shape_ID'] + 2*d)
    
    def preprs(s, x, xmap):
        OD, ID = s['grn_OD'], s['grn_shape_ID']
        A = np.pi/4 * (OD**2 - ID**2)
        x = x.at[xmap['grn_A']].set(A)
        
        return x
    
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'ID': 0.1,
        },
        x = {
        },
        
        # Required and integrated variables
        req_s = ['ID'],
        req_x = [],
        dx    = { },

        typename = 'shape',

        d2a = circle_d2a,
        fpreprs = preprs,

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def make_constOF_grain(shape, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'OD': 0.1,
            'L': 0.1,
            'OF': 1.0,
            'rho': 1000.0,
            **shape['s'],
        },
        x = {
            'A': 0.1,
            'd': 0.0,   # Distance regressed, i.e. increasing during burn
            
            # Calculated variables
            'V': 0.0,
            'Vdot': 0.0,
            'P': 101e3,
            'mdot': 0.0,
            'A_port': 0.0, # Open port cross-section area
            'G': 0.0,      # Oxidizer mass flux through the open port

            **shape['x'],
        },

        # Required and integrated variables
        req_s = ['OD', 'L', 'OF'],
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        # Designation and associated functions
        typename = 'grn',
        fderiv  = partial(d_grain_constOF, d2a=shape['shape_d2a']),
        fupdate = u_grain,
        fpreprs = shape['fpreprs'],

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def make_shiftOF_grain(shape, **kwargs):
    return make_part(
        # Default static and initial dynamic variables
        s = {
            'OD': 0.1,
            'L': 0.1,
            'Reg': jnp.array(ABS_REG), # Regression coefficient (mm/s), regression exponent, length exponent
            'phi': 1.0,                # Regression multiplier (e.g. swirl injection enhancement)
            'rho': 1070.0,             # ABS
            **shape['s'],
        },
        x = {
            'A': 0.1,
            'd': 0.0,   # Distance regressed, i.e. increasing during burn

            # Calculated variables
            'V': 0.0,
            'Vdot': 0.0,
            'P': 101e3,
            'mdot': 0.0,
            'A_port': 0.0, # Open port cross-section area
            'G': 0.0,      # Oxidizer mass flux through the open port

            **shape['x'],
        },

        # Required and integrated variables
        req_s = ['OD', 'L', 'Reg'],
        req_x = ['A'],
        dx    = {'A': 'Adot', 'd': 'ddot'},

        # Designation and associated functions
        typename = 'grn',
        fderiv  = partial(d_grain_shiftOF, d2a=shape['shape_d2a']),
        fupdate = u_grain,
        fpreprs = shape['fpreprs'],

        # The user-specified static and initial dynamic variables
        **kwargs,
    )

def bake_arbitrary_d2a(path, grain_diameter, n_step=100, n_visu=30, n_xy=512):
    """Bake regression tables from a cross-section image (grain dark/gray, port(s) light).

    Port outlines are sub-pixel grayscale iso-contours (burnfront.burn3d.section.image_port_polygons);
    regression uses the same exact-offset tables as the 2D/3D models. Returns
    (distances, perimeters, contours, grn_A0) with contours as [x, y, step_index] in image-style
    coordinates (origin at the lower-left of the grain's bounding square), grn_A0 the solid area.
    """
    from burnfront.burn3d.section import Section2D, image_port_polygons
    print(f"Baking file at: {path}")
    print(f"grain_diameter: {grain_diameter}")
    print(f"Settings: steps={n_step}, level curves={n_visu}")
    polys, info = image_port_polygons(path, grain_diameter)
    print(f"Resolution detected : {info['scale']:.6f} m/pixel, {len(polys)} port outline(s)")
    sec = Section2D.from_polygons(polys, grain_diameter, n_xy=n_xy)
    grn_A0, d, P, _ = _section_d2a(sec, n_step)
    # Truncate after the first zero perimeter (fully burnt), as before
    nz = np.nonzero(P <= 0.0)[0]
    n_keep = nz[0] + 1 if nz.size else n_step
    distances, perimeters = list(d[:n_keep]), list(P[:n_keep])
    every = math.ceil(n_step/n_visu)
    contour = []
    for i in range(0, n_keep, every):
        for c in sec.contours(distances[i]):
            contour.append([c[:, 0] + grain_diameter/2, c[:, 1] + grain_diameter/2, i])
    return (distances, perimeters, contour, grn_A0)
