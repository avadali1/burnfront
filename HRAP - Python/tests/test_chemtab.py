"""Chemistry tables: the interpolators must reproduce the raw .mat values at every grid node
(the .mat files store [Pc, OF]; a transposed table is silently misread by jax's interpolator)."""
import numpy as np
import pytest
import scipy.io
import jax.numpy as jnp

from hrap.chemtab import load_mat_tables, make_interpolators, orient
from hrap.motor import PROP_DIR


@pytest.mark.parametrize('path', sorted(PROP_DIR.glob('*.mat')), ids=lambda p: p.stem)
def test_interpolator_reproduces_table_nodes(path):
    raw = scipy.io.loadmat(path)['s'][0][0]
    OF, Pc, k, M, T = load_mat_tables(path)
    ik, iM, iT = make_interpolators(OF, Pc, k, M, T)
    pts = jnp.array([[o, p] for o in OF for p in Pc])
    got = np.asarray(iT(pts)).reshape(OF.size, Pc.size)
    rawT = np.asarray(raw['prop_T'], float)
    assert rawT.shape == (Pc.size, OF.size)        # file convention: rows = Pc
    assert np.allclose(got, rawT.T, rtol=1e-12)
    # Physical sanity: near-stoichiometric flame temperatures are far above fuel-rich ones
    assert got.max() > 2800.0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        make_interpolators(np.arange(3.0), np.arange(4.0), *(np.zeros((4, 3)),)*3)
    assert orient(np.zeros((4, 3)), 3, 4).shape == (3, 4)
