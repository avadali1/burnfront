"""Equilibrium (shifting) nozzle expansion tables, computed with Cantera.

For each (O/F, Pc) of a propellant's chemistry grid:
  1. chamber: adiabatic, isobaric (HP) equilibrium of liquid N2O + fuel at Pc; gas phase (NASA
     thermo, C/H/O/N species) plus condensed graphite C(gr) where carbon is in excess
  2. isentropic expansion to p = r Pc, re-equilibrating at every pressure ('shifting') or with the
     chamber composition held fixed ('frozen'): h(r) and gas density rho(r) per kg of mixture,
     u = sqrt(2 (h0 - h)), G = rho u (homogeneous two-phase flow, condensed volume neglected)
  3. throat = maximum of G (critical flow), c* = Pc/G_max, area ratio eps(r) = G_max/G(r)
Supersonic-branch quantities are stored against the fixed grid rho_k = r/r_throat, dimensionless:
    U[k] = u/c*,  LE[k] = ln eps,  plus r_throat and c*_eq / c*_gamma.

Graphite equilibrium: carbon condenses when the gas-phase carbon potential
lambda_C = (2 mu_CO - mu_CO2)/RT exceeds g_C(gr)/RT; the condensed amount solves lambda_C = g/RT.
Validated against NASA CEA (RocketCEA) in tests/test_expansion.py; the chamber temperature also
reproduces HRAP's CEA-generated .mat tables (e.g. paraffin, HDPE: within ~1 K, fuel-rich included).

The engine keeps the propellant's .mat chamber tables and c* efficiency for the chamber and uses
these tables as dimensionless corrections: c* = c*_model (c*_eq/c*_gamma), exit velocity
u = U c*, where c*_gamma is the constant-gamma c* of the same Cantera chamber state (frozen gamma
cp/cv of the gas, R = Ru/M with M = mixture mass per kmol of gas, as in CEA/HRAP tables).

Assumptions: N2O enters as saturated liquid at 298.15 K; 1-D isentropic nozzle flow; condensed
carbon in velocity and thermal equilibrium with the gas; no metals (metallized fuels unsupported).
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

RU = 8314.462618
MC = 12.011
# Reactant cards: element counts per formula unit and standard enthalpy of formation (J/mol), as in
# burnfront/thermo.dat; ABS (C3.85 H4.85 N0.43, +62.63 kJ/mol) after Whitmore et al.
REACTANTS = {
    'N2O(L)': ({'N': 2, 'O': 1}, 65100.0),
    'ABS': ({'C': 3.85, 'H': 4.85, 'N': 0.43}, 62630.0),
    'Paraffin': ({'C': 32, 'H': 66}, -967800.0),
    'HTPB': ({'H': 302, 'C': 240, 'O': 2}, -997885.677),
    'HDPE': ({'C': 2, 'H': 4}, -53170.613),
    'Sorbitol': ({'C': 6, 'H': 14, 'O': 6}, -1353700.0),
}
# Fuel blends by mass fraction (propellant file stem -> {reactant: mass fraction})
FUELS = {
    'ABS': {'ABS': 1.0},
    'Paraffin': {'Paraffin': 1.0},
    'HTPB': {'HTPB': 1.0},
    'HDPE': {'HDPE': 1.0},
    'Sorbitol': {'Sorbitol': 1.0},
    'HTPB_Paraffin': {'HTPB': 0.5, 'Paraffin': 0.5},
}
ATOMIC_MASS = {'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007}
RHO_GRID = np.concatenate([[1.0], np.geomspace(0.97, 1e-4, 55)])   # r / r_throat, supersonic branch (eps up to ~300)
VERSION = 3


def molar_mass(comp):
    return sum(n*ATOMIC_MASS[e] for e, n in comp.items())


def supported(propellant):
    return Path(str(propellant)).stem in FUELS


def reactant_mixture(fuel_blend, OF, ox='N2O(L)'):
    """Element kmol per kg of propellant and its enthalpy (J/kg) at mixture ratio OF."""
    parts = {ox: OF/(1.0 + OF)}
    for name, mf in fuel_blend.items():
        parts[name] = parts.get(name, 0.0) + mf/(1.0 + OF)
    elem, h = {}, 0.0
    for name, mass in parts.items():
        comp, hf = REACTANTS[name]
        kmol = mass/molar_mass(comp)
        h += kmol*hf*1e3
        for e, n in comp.items():
            elem[e] = elem.get(e, 0.0) + kmol*n
    return elem, h


class Equilibrium:
    """Gas (Cantera, NASA C/H/O/N species) + condensed graphite equilibrium per kg of mixture."""

    def __init__(self):
        import cantera as ct
        sp = [s for s in ct.Species.list_from_file('nasa_gas.yaml')
              if set(s.composition) <= {'C', 'H', 'O', 'N'} and s.charge == 0 and s.composition.get('C', 0) <= 3]
        self.gas = ct.Solution(thermo='ideal-gas', species=sp)
        self.gr = [s for s in ct.Species.list_from_file('nasa_condensed.yaml') if s.name == 'C(gr)'][0].thermo
        self.iCO, self.iCO2 = self.gas.species_index('CO'), self.gas.species_index('CO2')
        self.nc_guess = None

    # -- condensed carbon
    def g_gr(self, T):
        return (self.gr.h(T) - T*self.gr.s(T))/(RU*T)

    def lam_C(self):
        mu = self.gas.chemical_potentials
        return (2*mu[self.iCO] - mu[self.iCO2])/(RU*self.gas.T)

    def _gas_tp(self, T, P, elem):
        self.gas.TPX = T, P, {e: n for e, n in elem.items() if n > 0}
        self.gas.equilibrate('TP')

    def tp(self, T, P, elem):
        """TP equilibrium; returns condensed carbon kmol per kg (gas left in self.gas)."""
        self._gas_tp(T, P, elem)
        if elem.get('C', 0.0) <= 0.0 or self.lam_C() <= self.g_gr(T):
            return 0.0
        g = self.g_gr(T)
        C = elem['C']

        def f(nc):
            e = dict(elem); e['C'] = C - nc
            self._gas_tp(T, P, e)
            return self.lam_C() - g
        # lambda_C decreases monotonically as carbon is removed; warm-started bracket
        lo, hi = 0.0, C*(1 - 1e-9)
        if self.nc_guess is not None and 0.0 < self.nc_guess < hi:
            a, b = 0.97*self.nc_guess, min(1.03*self.nc_guess, hi)
            fa, fb = f(a), f(b)
            if fa > 0 > fb:
                lo, hi = a, b
            elif fa <= 0:
                hi = a
            else:
                lo = b
        nc = brentq(f, lo, hi, xtol=1e-13*C, rtol=1e-11)
        f(nc)
        self.nc_guess = nc
        return nc

    def state(self, nc):
        T = self.gas.T
        mg = 1.0 - nc*MC
        return dict(T=T, P=self.gas.P, nc=nc, mg=mg, h=mg*self.gas.h + nc*self.gr.h(T),
                    s=mg*self.gas.s + nc*self.gr.s(T), rho=1.0/(mg/self.gas.density),
                    M=self.gas.mean_molecular_weight/mg, X=self.gas.X.copy())

    def _solve_T(self, prop, target, P, elem, T_lo=250.0, T_hi=6000.0, T_guess=None):
        def f(T):
            return self.state(self.tp(T, P, elem))[prop] - target
        lo, hi = T_lo, T_hi
        if T_guess is not None:
            # h and s increase with T: try a tight bracket around the guess, widen if needed
            for w in (0.01, 0.05, 0.25):
                a, b = max(T_lo, T_guess*(1 - w)), min(T_hi, T_guess*(1 + w))
                if f(a) < 0 < f(b):
                    lo, hi = a, b
                    break
        T = brentq(f, lo, hi, xtol=1e-9*max(lo, 1.0), rtol=1e-11)
        return self.state(self.tp(T, P, elem))

    def hp(self, h, P, elem):
        # Fast path: gas-only HP from Cantera; accept if no carbon would condense
        try:
            self._gas_tp(3000.0, P, elem)
            self.gas.HP = h, P
            self.gas.equilibrate('HP')
            if elem.get('C', 0.0) <= 0.0 or self.lam_C() <= self.g_gr(self.gas.T):
                return self.state(0.0)
        except Exception:
            pass   # frozen-composition HP step failed (fuel-rich): use the robust root-find path
        return self._solve_T('h', h, P, elem)

    def sp_shifting(self, s, P, elem, T_guess):
        self._gas_tp(T_guess, P, elem)
        try:
            self.gas.SP = s, P
            self.gas.equilibrate('SP')
            if elem.get('C', 0.0) <= 0.0 or self.lam_C() <= self.g_gr(self.gas.T):
                return self.state(0.0)
        except Exception:
            pass
        return self._solve_T('s', s, P, elem, T_lo=60.0, T_guess=T_guess)

    def sp_frozen(self, s, P, X0, nc):
        mg = 1.0 - nc*MC

        def f(T):
            self.gas.TPX = T, P, X0
            return mg*self.gas.s + nc*self.gr.s(T) - s
        T = brentq(f, 20.0, 6000.0, xtol=1e-7, rtol=1e-12)
        self.gas.TPX = T, P, X0
        return self.state(nc)


def nozzle_curves(eq, fuel_blend, OF, Pc, shifting=True):
    """Chamber state and supersonic expansion curves for one (OF, Pc)."""
    elem, h_in = reactant_mixture(fuel_blend, OF)
    eq.nc_guess = None
    c = eq.hp(h_in, Pc, elem)
    h0, s0, T0, X0, nc0 = c['h'], c['s'], c['T'], c['X'], c['nc']
    gam_f = eq.gas.cp/eq.gas.cv
    M0 = c['M']
    T_last = [T0]

    def at(r):
        if shifting:
            st = eq.sp_shifting(s0, r*Pc, elem, T_last[0])
        else:
            st = eq.sp_frozen(s0, r*Pc, X0, nc0)
        T_last[0] = st['T']
        u = np.sqrt(max(2.0*(h0 - st['h']), 0.0))
        return u, st['rho']*u

    # Throat: maximise G(r); coarse scan then golden section
    rr = np.linspace(0.45, 0.70, 11)
    Gs = []
    for r in rr:
        Gs.append(at(r)[1])
    j = int(np.argmax(Gs))
    a, b = rr[max(j - 1, 0)], rr[min(j + 1, len(rr) - 1)]
    gr = (np.sqrt(5) - 1)/2
    c_, d_ = b - gr*(b - a), a + gr*(b - a)
    Gc, Gd = at(c_)[1], at(d_)[1]
    for _ in range(30):
        if Gc > Gd:
            b, d_, Gd = d_, c_, Gc
            c_ = b - gr*(b - a); Gc = at(c_)[1]
        else:
            a, c_, Gc = c_, d_, Gd
            d_ = a + gr*(b - a); Gd = at(d_)[1]
    r_t = 0.5*(a + b)
    G_t = at(r_t)[1]
    cstar = Pc/G_t
    T_last[0] = T0
    U, LE = np.empty(RHO_GRID.size), np.empty(RHO_GRID.size)
    for k, q in enumerate(RHO_GRID):
        u, G = at(r_t*q)
        U[k] = u/cstar
        LE[k] = 0.0 if k == 0 else np.log(max(G_t/max(G, 1e-300), 1.0))
    R = RU/M0
    Gam = np.sqrt(gam_f)*(2/(gam_f + 1))**((gam_f + 1)/(2*(gam_f - 1)))
    return dict(cstar=cstar, r_t=r_t, U=U, LE=LE, T0=T0, M0=M0, gamma_f=gam_f,
                cstar_gamma=np.sqrt(R*T0)/Gam, nc0=nc0)


@dataclass
class ExpansionTable:
    OF: np.ndarray           # (nOF,)
    Pc: np.ndarray           # (nPc,)
    rho: np.ndarray          # (nk,) r/r_throat grid
    U: np.ndarray            # (nOF, nPc, nk) u/c*
    LE: np.ndarray           # (nOF, nPc, nk) ln(area ratio)
    r_t: np.ndarray          # (nOF, nPc) throat pressure ratio
    cstar: np.ndarray        # (nOF, nPc) equilibrium c*
    cstar_gamma: np.ndarray  # (nOF, nPc) constant-gamma c* of the same chamber state
    T0: np.ndarray           # (nOF, nPc) chamber temperature
    M0: np.ndarray           # (nOF, nPc) mixture mass per kmol gas
    gamma_f: np.ndarray
    nc0: np.ndarray          # (nOF, nPc) condensed carbon in the chamber, kmol/kg
    shifting: bool = True

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(**{k: (bool(d[k]) if k == 'shifting' else d[k]) for k in cls.__dataclass_fields__})


_KEYS2 = ('r_t', 'cstar', 'cstar_gamma', 'T0', 'M0', 'gamma_f', 'nc0')


def _row(args):
    propellant, of, Pc, shifting = args
    eq = Equilibrium()
    blend = FUELS[Path(str(propellant)).stem]
    return [nozzle_curves(eq, blend, float(of), float(pc), shifting) for pc in Pc]


def compute_table(propellant, OF, Pc, shifting=True, n_jobs=None, verbose=False):
    nk = RHO_GRID.size
    out = {k: np.zeros((len(OF), len(Pc), nk)) for k in ('U', 'LE')}
    out.update({k: np.zeros((len(OF), len(Pc))) for k in _KEYS2})
    tasks = [(propellant, of, list(Pc), shifting) for of in OF]
    n_jobs = n_jobs or min(len(OF), os.cpu_count() or 1)
    if n_jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context('spawn')) as ex:
            rows = list(ex.map(_row, tasks))
    else:
        rows = [_row(t) for t in tasks]
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            for k in out:
                out[k][i, j] = c[k]
    return ExpansionTable(OF=np.asarray(OF, float), Pc=np.asarray(Pc, float), rho=RHO_GRID.copy(),
                          shifting=shifting, **out)


def cache_dir():
    base = Path(os.environ.get('BURNFRONT_CACHE', Path.home()/'.burnfront_cache'))
    base.mkdir(parents=True, exist_ok=True)
    return base


def table_for(propellant, OF, Pc, shifting=True, n_jobs=None, verbose=False):
    """Cached expansion table on the propellant's (OF, Pc) grid."""
    stem = Path(str(propellant)).stem
    if stem not in FUELS:
        raise ValueError(f'no reactant definition for {stem!r}; equilibrium expansion supports {sorted(FUELS)}')
    key = json.dumps([VERSION, stem, FUELS[stem], list(map(float, OF)), list(map(float, Pc)), shifting,
                      {k: v for k, v in REACTANTS.items()}], sort_keys=True)
    path = cache_dir()/f'expansion_{stem}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz'
    if path.exists():
        return ExpansionTable.load(path)
    if verbose:
        print(f'Computing equilibrium expansion table for {stem} (cached afterwards)...')
    tab = compute_table(propellant, OF, Pc, shifting, n_jobs, verbose)
    tab.save(path)
    return tab
