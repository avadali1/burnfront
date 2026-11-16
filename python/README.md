# Burnfront (Python)

Burnfront simulates self-pressurizing hybrid rocket motors: a saturated N2O tank, the injector, a regressing fuel grain, the combustion chamber and the nozzle, integrated together in time.

It is a modified version of the Python Hybrid Rocket Analysis Program (HRAP) by Robert Nickel and contributors; its package lists Drew Nickel and Thomas A. Scott as authors ([github.com/rnickel1/HRAP_Source](https://github.com/rnickel1/HRAP_Source)), changed by Arushi Vadali starting in November 2026. It is licensed under GPL v3; see the [top-level README](../README.md#license) for credits and license terms.

Compared with HRAP, it adds the following:

- **A 3D burn simulation of the fuel grain.** It handles any port shape, including STEP/STL CAD files, twisted stars, helical ports and multi-port grains. The 2D outline model is kept as a fast mode.
- **Energy-consistent tank and chamber models.**
  - **Injector:** two-phase Dyer/HEM models.
  - **Nozzle expansion:** shifting chemical equilibrium, validated against NASA CEA.
- **Nozzle and exhaust-plume CFD,** with live contours, plus a transient nozzle-wall heating analysis.
- **Parameter sweeps** that run in parallel and write CSV.
- **A GUI** that exposes all of the above.

> **Results from HRAP's Python version are not comparable.** Several bugs that affected every simulation were fixed (see [Changes that affect results](#changes-that-affect-results)). In particular, earlier versions read the propellant chemistry tables transposed, so chamber temperature, c* and Isp were wrong. Re-run any motor you designed with an older version.

---

## Installation

Python 3.10 or newer is required. Burnfront is not on PyPI yet (note that `pip install hrap` installs the original HRAP, not Burnfront).

**Quick install from GitHub** (no clone needed):

```
python -m pip install "burnfront[3d] @ git+https://github.com/avadali1/burnfront.git#subdirectory=python"
```

**From a clone** (for development):

```
git clone https://github.com/avadali1/burnfront.git
cd burnfront
cd python
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
python -m pip install --upgrade pip
python -m pip install -e ".[3d]"
```

The `3d` extra installs STEP/STL import (`cadquery-ocp`, `trimesh`, `rtree`, `scikit-fmm`). Without it, the GUI and API work with analytic port shapes only.

For running the test suite and the NASA CEA validation tests, also install the dev extra:

```
python -m pip install -e ".[3d,dev]"
```

## Running the GUI

```
burnfront
```

The original HRAP GUI is still available as `burnfront-legacy`.

### Typical workflow
1. **Pick a grain.**
   - **From CAD:** click the blue **+ Add STEP / STL grain file…** button. The mesh is checked immediately, and a line under the button reports whether it is watertight, the outer diameter, length, number of ports and initial burning area. For STL files, set the units and the port axis.
   - **Analytic port:** choose Circle or Star under **Grain**, with optional twist or helix, or load a cross-section image.
2. **Choose the grain model.** **3D (STEP/STL)** runs the full level-set burn. **2D outline** is a single cross-section extruded along the port, and runs in seconds.
3. **Set the starting conditions** in the left panel:
   - **Tank:** volume, temperature or pressure, liquid fill.
   - **Injector:** number of holes, hole diameter, Cd.
   - **Grain and fuel:** fuel, grain size, regression-law overrides.
   - **Chamber and nozzle:** free chamber volume, throat diameter, expansion ratio, c* efficiency.
4. **Click Run.** The run happens in the background, and the status line shows the elapsed time.
5. **Look at the results** in the tabs on the right.

| Tab | What it shows |
|---|---|
| **Grain view** | 3D render of the port surface at any snapshot time. Burning front in orange, exposed case in grey. Rotate with the sliders. |
| **Cross-section** | The burning front at each snapshot time over the case outline, at any position along the grain. |
| **Histories** | Thrust, chamber and tank pressure, O/F, local mass flux (head, mid, aft), oxidizer and fuel flow. |
| **Flux map** | Local mass flux G over position along the grain and time. |
| **CFD: nozzle & plume** | Live axisymmetric compressible-flow solution (Mach, pressure, temperature, density, velocity) at a chosen time in the burn. Includes residuals, wall distributions and a results table: discharge coefficient, CFD thrust and thrust coefficient against 1D, exit state, and any internal shock. |
| **Nozzle thermal** | Gas-side heat flux along the nozzle and wall temperature over the whole burn, compared with the wall material's limit. |
| **Summary** | Impulse, Isp, burn time, pressures, propellant used, sliver, burn-through time. Also the time spent outside the chemistry table or with nozzle flow separation, and the CAD mesh report. |
| **Sweep** | Vary one or two inputs (grid or paired values) in parallel; results go to a table and a CSV. |

**File** menu: save and load configurations (JSON), and export the time history (CSV) or the thrust curve (ENG).

Two operations are slow the first time:
- The first run with a new fuel builds its shifting-equilibrium expansion table, which takes about 2 minutes on 16 cores. It is cached in `~/.burnfront_cache` and loads instantly afterwards.
- JAX compiles the engine on the first run of each configuration.

---

## Physics

### Grain
- **3D level-set burn** (`burnfront.burn3d`).
  - **Port representation:** the port surface is a signed distance field on a voxel grid, and the outer case is inert. For CAD files the distance is computed exactly to the tessellated surface; for twisted and helical shapes, to a fine ruled mesh.
  - **Regression:** the front recedes along the local normal using the exact (Hopf–Lax) solution of the level-set equation.
  - **Regression rate:** `rdot = phi * a * G^n * L^m`. The local mass flux G(z) comes from an axial flow model, using either oxidizer only or oxidizer plus fuel added upstream, divided by the local open port area. a and n default to the propellant file (ABS: 0.198 mm/s, 0.325); phi is a swirl multiplier.
  - **Outputs:** fuel mass flow, port volume, burning area, burn-through time and sliver.
- **2D outline mode** (`burnfront.burn3d.section`). The same regression and flux definitions on a single cross-section, with an optional helical correction that approximates twisted ports in 2D.
- **Engine coupling** (`burnfront.engine3d`). The tank, chamber and nozzle are integrated in JAX. The grain geometry enters as offset tables measured from the level set, and these are re-measured every time the grain regresses one grid cell.

### Tank and injector
- **Equilibrium tank** (default). Saturated two-phase N2O with mass, volume and energy conserved exactly, plus optional wall heat capacity. Liquid run-out is latched, and afterwards the tank delivers its homogeneous vapor mixture.
- **Legacy tank.** The original evaporative-cooling model, kept for comparison.
- **Liquid injector models:**
  - **Dyer** (default).
  - **HEM:** homogeneous equilibrium, tabulated from CoolProp, choked.
  - **Incompressible.**
- **Vapor injector:** compressible orifice flow (choked or subsonic).

### Chamber and nozzle
- **Chamber gas.** The chamber is an ideal-gas mixture of the initial air and the combustion products, with mass and energy balances. The nozzle mass flow is computed from the current chamber state.
- **Chamber chemistry** comes from the bundled CEA-generated propellant tables over O/F and Pc. Lookups outside a table's range are clamped, and how long that happened is reported.
- **Nozzle expansion:**
  - **Shifting equilibrium** (default). The products keep reacting as they expand, computed with Cantera's NASA thermo data plus condensed graphite. Supported fuels: ABS, paraffin, HTPB, HDPE, sorbitol, 50/50 HTPB/paraffin. Other fuels fall back to frozen expansion, with a note in the results.
  - **Frozen.** A single γ for the whole expansion.
- **Nozzle flow regimes** handled: choked, subsonic (unchoked), and flow separation (Summerfield criterion, `p_sep / Pa = 0.4` by default).

### CFD and nozzle thermal (`burnfront.cfd`)
- **CFD solver.** Axisymmetric Euler equations, cell-centred finite volume (HLLC Riemann solver, MUSCL reconstruction with the van Albada limiter, SSP-RK2 with local time stepping), JAX-compiled.
- **CFD domain.** A body-fitted grid over the aft chamber and the conical converging-diverging nozzle, plus an exhaust-plume block.
- **Gas properties.** The chamber state comes from the motor run at the chosen time. An effective γ is used so that the frozen-γ flow reproduces the shifting-equilibrium exit pressure.
- **Nozzle wall heating.** Gas-side heat transfer from the Bartz correlation, and transient radial conduction through the wall. Materials: graphite, copper, 4130 steel, 6061 aluminium, silica phenolic.

## Validation

All checks below are automated in `tests/` (85 tests).

| Check | Result |
|---|---|
| 3D cylinder against the analytic regression solution | second order; burn-through time error 5e-5 |
| 3D against 2D (straight star), and against the JAX engine | about 0.3%; 4e-7 |
| Twisted star | mass conserved to 1e-5; burning area matches exact ruled-surface areas |
| STEP grid convergence (twisted star, 4 grids) | volumes and burned mass converge at order ≈ 2; uncertainty 0.07–0.3% |
| Engine mass and c* balance | chamber mass ledger exact; c* balance within 5e-4 |
| Tank energy conservation | through liquid run-out |
| Nozzle expansion against NASA CEA (RocketCEA) | chamber temperature, c* and Isp (vacuum and sea level) within 0.1%, O/F 1.5–8, frozen and shifting |
| Chamber temperature against the bundled CEA `.mat` tables | about 1 K (paraffin, HDPE), fuel-rich points with graphite included |
| CFD discharge coefficient against the Kliegel–Levine analytic transonic solution | 0.9944 vs 0.9945 |
| CFD mass conservation | exact |
| Internal normal shock (over-expanded case) | detected |
| Wall conduction against the semi-infinite erf solution | 0.04 K on a 1000 K step; energy balance within 0.1% |

## Python API

```python
from burnfront.motor import MotorConfig, run_motor, sweep

cfg = MotorConfig(
    mode='3d',                                   # or '2d'
    port={'type': 'star', 'n_tips': 6, 'r_valley': 0.010, 'r_tip': 0.018, 'twist_turns': 1.0},
    # port={'type': 'file', 'path': 'grain.step', 'units': 'mm', 'axis': 'z'},
    grain_OD=0.07, grain_L=0.30, propellant='ABS',
    tank_V=3e-3, P_tank=50e5, T_tank=None, fill=0.85,
    inj_N=6, inj_d=1.5e-3, inj_Cd=0.65, inj_liq_model='Dyer',
    throat_d=0.015, ER=5.0, cstar_eff=0.95, V_free=2e-4,
    noz_expansion='equilibrium', T=8.0,
)
res = run_motor(cfg, snapshot_times=(0, 2, 4))
print(res.summary())                 # impulse, Isp, burn time, pressures, ...
res.to_csv('history.csv')

rows = sweep(cfg, {'inj_N': [4, 6, 8], 'fill': [0.7, 0.9]}, n_jobs=4, csv_path='sweep.csv')
```

All quantities are SI. See the docstrings of `burnfront.motor`, `burnfront.burn3d` (`cad.load_grain`, `shapes`, `burn_prescribed`), `burnfront.expansion`, `burnfront.cfd.nozzle_flow` and `burnfront.cfd.thermal`.

The engine is written with JAX for just-in-time compilation. If you modify the engine parts, note JAX's functional style for loops and conditionals: https://docs.jax.dev/en/latest/control-flow.html.

## Examples

See [`examples/README.md`](examples/README.md).

## Tests

```
python -m pytest tests -q
```

The full suite takes about 11 minutes. `tests/test_expansion.py` needs `rocketcea` (dev extra) for the NASA CEA comparisons, and builds the cached expansion table on first use.

## Changes that affect results

Fixed in Burnfront relative to HRAP, roughly in order of impact:

1. **Chemistry tables were read transposed.** `.mat` files store k, M and T as [Pc, O/F], but the interpolators assumed [O/F, Pc]. Every run used wrong chamber properties; for example, ABS at O/F 6 and 20 bar gave 2093 K instead of 3247 K.
2. **Chamber pressure model.** The legacy `dP/P = dm/m − dV/V` form kept the initial air's R·T, which meant about 12× too much stored gas and pressure transients about 12× too slow. The nozzle flow was also read one evaluation late.
3. **Oxidizer flux** used the solid area instead of the open port area (shiftOF grain).
4. **Injector model default.** Dyer replaces the incompressible model, which over-predicts flashing N2O flow.
5. **Vapor-phase tank.** The tank switched between the liquid and vapor models on every step after liquid run-out, and the vapor injector formula was always choked.
6. **Initial chamber gas** was computed from the full chamber volume, fuel included.
7. **Nozzle Cd in examples:** the examples passed `C_d=` (ignored) instead of `Cd=`.

## Limitations

- **CFD** is inviscid: no boundary layers and no turbulent mixing. The plume is shown as a pseudo-time average. A normal shock inside the nozzle is flagged rather than modelled as boundary-layer separation. A single γ is used, and the ambient air is treated as the same gas.
- **Wall heating** uses constant material properties, with no ablation and no axial conduction.
- **3D burning area at sharp corners** is first order in grid size (about 0.2% at 64 cells). Volumes and integrated quantities are second order.
- **Twisted STEP grains** take several minutes to build at high resolution; 48–64 cells is usually enough.
- **Equilibrium expansion** has gas-phase products plus graphite only, so metallized fuels use frozen expansion. N2O is assumed to enter the chamber as saturated liquid at 298 K.

## Developer notes

The source lives in `burnfront/`. Main modules:

| Module | Contents |
|---|---|
| `motor.py` | Configs, runs, sweeps |
| `engine3d.py` | 3D coupling |
| `grain_axial.py` | Tabulated grain part |
| `tank.py`, `chamber.py`, `nozzle.py` | Engine parts |
| `expansion.py` | Equilibrium tables |
| `chemtab.py` | Chemistry tables |
| `burn3d/` | Level-set core, CAD import, convergence studies |
| `cfd/` | Nozzle CFD and thermal |
| `gui/app.py` | GUI |
| `gui/model.py` | GUI logic (no display needed) |

The original GUI is `gui/main.py`.
