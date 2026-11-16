"""Regression law parameters."""
from dataclasses import dataclass

# ABS, from burnfront/resources/propellant_configs/ABS.mat (prop_Reg, prop_Rho)
ABS_A, ABS_N, ABS_M, ABS_RHO = 0.198, 0.325, 0.0, 1070.0


@dataclass
class RegressionLaw:
    """rdot [m/s] = phi * 1e-3 * a * G^n * L^m  (HRAP legacy units: a in mm/s, G in kg/m^2/s).

    flux: 'total'    -> G(z) = (mdot_ox + fuel added upstream of z) / A_port(z)
          'oxidizer' -> G(z) = mdot_ox / A_port(z)
    """
    a: float = ABS_A
    n: float = ABS_N
    m: float = ABS_M          # grain-length exponent (legacy HRAP), uses total grain length
    phi: float = 1.0          # multiplier, e.g. swirl injection enhancement
    rho: float = ABS_RHO      # fuel density, kg/m^3
    flux: str = 'total'

    def __post_init__(self):
        if self.flux not in ('total', 'oxidizer'):
            raise ValueError("flux must be 'total' or 'oxidizer'")
        if self.n <= 0:
            raise ValueError('regression exponent n must be positive')

    def k(self, L):
        """Coefficient in rdot = k * G^n (SI)."""
        return self.phi * 1e-3 * self.a * L**self.m

    @classmethod
    def from_mat(cls, path, **overrides):
        """Load a, n, m and rho from an HRAP propellant .mat file."""
        import numpy as np
        import scipy.io
        c = scipy.io.loadmat(path)['s'][0][0]
        a, n, m = np.asarray(c['prop_Reg'], dtype=float).ravel()
        kw = dict(a=a, n=n, m=m, rho=float(np.asarray(c['prop_Rho']).ravel()[0]))
        kw.update(overrides)
        return cls(**kw)
