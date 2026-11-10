"""Propellant chemistry tables (chamber k, M, T over O/F and Pc) and their JAX interpolators.

HRAP .mat propellant files store prop_k/prop_M/prop_T with shape (n_Pc, n_OF). The interpolators
are built on the grid (OF, Pc), so the values must be (n_OF, n_Pc): ``load_mat_tables`` returns them
in that orientation and ``make_interpolators`` refuses mismatched shapes (jax's
RegularGridInterpolator does not check, and a transposed table silently returns wrong values).
"""
import numpy as np


def orient(v, n_of, n_pc):
    """Return table v as (n_OF, n_Pc). Accepts (n_Pc, n_OF) (HRAP .mat convention) or a scalar."""
    v = np.asarray(v, dtype=float)
    if v.size == 1:
        return np.full((n_of, n_pc), v.item())
    if v.shape == (n_pc, n_of) and n_pc != n_of:
        return v.T.copy()
    if v.shape == (n_of, n_pc) and n_pc != n_of:
        return v.copy()
    if v.shape == (n_pc, n_of):          # square grid: follow the .mat convention (rows = Pc)
        return v.T.copy()
    raise ValueError(f'table shape {v.shape} does not match grid (OF {n_of}, Pc {n_pc})')


def load_mat_tables(path):
    """(OF, Pc, k, M, T) from an HRAP propellant .mat file, tables oriented (n_OF, n_Pc)."""
    import scipy.io
    c = scipy.io.loadmat(path)['s'][0][0]
    OF = np.asarray(c['prop_OF'], float).ravel()
    Pc = np.asarray(c['prop_Pc'], float).ravel()
    k, M, T = (orient(c[key], OF.size, Pc.size) for key in ('prop_k', 'prop_M', 'prop_T'))
    return OF, Pc, k, M, T


def make_interpolators(OF, Pc, k, M, T):
    """JAX RegularGridInterpolators for k, M, T on the (OF, Pc) grid; tables must be (n_OF, n_Pc)."""
    from jax.scipy.interpolate import RegularGridInterpolator
    OF, Pc = np.asarray(OF, float).ravel(), np.asarray(Pc, float).ravel()
    out = []
    for v, fill in ((k, 1.4), (M, 29.0), (T, 293.0)):
        v = np.asarray(v, float)
        if v.shape != (OF.size, Pc.size):
            raise ValueError(f'chemistry table shape {v.shape} != (n_OF, n_Pc) = {(OF.size, Pc.size)}')
        out.append(RegularGridInterpolator((OF, Pc), v, fill_value=fill))
    return tuple(out)


def mat_interpolators(path):
    return make_interpolators(*load_mat_tables(path))
