# Burnfront

Burnfront is a simulator for self pressurizing hybrid rocket motors (N2O oxidizer with a solid fuel grain). It is a modified version (a fork) of the **Hybrid Rocket Analysis Program (HRAP)** by Robert Nickel and the University of Tennessee Rocket Engineering Team. Burnfront is not affiliated with or endorsed by the HRAP authors.

The name comes from the main addition, the burning surface of the fuel grain (the burn front) is tracked in full 3D through any port shape, including grains imported from CAD.

> **Modification notice.** This is a modified version of HRAP ([github.com/rnickel1/HRAP_Source](https://github.com/rnickel1/HRAP_Source)), changed by Arushi Vadali starting in November 2026. The changes are listed under [What Burnfront changes](#what-burnfront-changes), and in detail in the commit history.

## Python version

The Python version is where all of Burnfront's work is. See [python/README.md](python/README.md) for installation and use. 

Compared with HRAP, it adds:
- **3D fuel-grain burn simulation** from STEP/STL CAD files or analytic ports, including twisted and helical ports and multi-port grains. A fast 2D outline mode is also available.
- **Energy-conserving tank and chamber models.**
  - **Injector:** Dyer and HEM two phase models.
  - **Nozzle expansion:** shifting chemical equilibrium which is validated against NASA CEA.
- **Nozzle and plume CFD** with live contours, plus nozzle wall heating over the burn.
- **Parallel parameter sweeps** written to CSV. 
- **A GUI** for all of the above. Start it with `burnfront`.

The Python code lives in the `python` folder; the package and command are `burnfront`.

## What Burnfront changes

**New features**
- 3D level-set grain regression, with STEP/STL import.
- Coupled tank, chamber and nozzle integration.
- An equilibrium two-phase tank.
- Dyer and HEM injectors.
- Shifting-equilibrium nozzle expansion.
- Axisymmetric nozzle and plume CFD.
- Nozzle wall heating.
- Parameter sweeps and a new GUI.

**Fixed bugs.** These change simulation results compared with HRAP's Python version:
- The propellant chemistry tables were read transposed. This affected every simulation.
- The chamber pressure model kept the initial air's gas constant.
- The oxidizer mass flux was computed with the wrong area.
- The tank flipped between liquid and vapor models after liquid run-out.
- The example scripts mistyped the nozzle discharge-coefficient keyword.

**Unchanged.** The MATLAB version (below) and the original Python GUI (`burnfront-legacy`) are kept as they were in HRAP.

Results from HRAP's Python version are not comparable with Burnfront's (see "Changes that affect results" in the [Python README](python/README.md#changes-that-affect-results)).

## MATLAB version (original HRAP, unmodified)

Go to the "Installers/MATLAB version" directory. Prerequisites for the installation include having the MATLAB runtime installed (free.) and in order to save/export data the application needs to be run as administrator.

## Validation cases (original HRAP)

These hot fire comparisons were made with the original HRAP. They have not been repeated with Burnfront.

Baltic Space HyPEx hot fire (r/rocketry user FlyingBanana)

![image](https://github.com/user-attachments/assets/4e048de8-12b3-4299-89f5-ec4241b3ccb2)

Equatorial Space Systems 750 N subscale demonstrator (James Anderson)

![image](https://github.com/user-attachments/assets/63330cf6-2fe3-4712-a0ec-310baa33e389)

Student Space Technology Association 38mm Subscale Motor

![image](https://github.com/user-attachments/assets/508ee5ae-4a46-4c41-8743-3ec823128cfe)

Student Space Technology Association 127mm Flight Motor

![image](https://github.com/user-attachments/assets/89afa8e9-32a9-4057-92d3-aaae7cde900f)

## Work ahead
- A non equilibrium tank model, with separate liquid and vapor temperatures and heat and mass transfer between them.
- Viscous and turbulent CFD (boundary layers, plume mixing, separation), and ablating nozzle materials.
- Equilibrium expansion for metallized fuels (condensed alumina).
- More propellant config files, and hot fire data for validating regression laws.

## Credits

**HRAP**, the original program:
- Created by Robert Nickel for the University of Tennessee Rocket Engineering Team.
- The Python version's package metadata lists Drew Nickel and Thomas A. Scott as authors.
- Source: [github.com/rnickel1/HRAP_Source](https://github.com/rnickel1/HRAP_Source).

**Burnfront** modifications: Arushi Vadali.

## License

Burnfront is free software licensed under the **GNU General Public License v3.0**, the same license as HRAP. The full text is in [`LICENSE`](LICENSE).

What that means:
- You may use, copy, modify and redistribute Burnfront, including commercially.
- If you distribute Burnfront or a modified version, it must stay under GPL v3. You must keep this license and the credits above, and provide the source code.
- Modified versions must say that they are modified, and when.
- There is no warranty. Rocket motors are dangerous: verify every result independently before building or firing anything.
