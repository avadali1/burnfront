# HRAP_Source
The Hybrid Rocket Analysis Program (HRAP) was developed by Robert Nickel for use by the University of Tennessee Rocket Engineering Team. HRAP is a versatile tool utilizing a thermodynamic equilibrium model for  simulation of self-pressurizing hybrid rocket motors, specifically those powered with Nitrous Oxide stored as a saturated liquid-vapor mixture. 

## Python - Hybrid Rocket Analysis Program
Navigate to the "HRAP - Python" directory for installation and usage instructions ([HRAP - Python/README.md](HRAP%20-%20Python/README.md)).

The Python version now includes:
- **3D fuel-grain burn simulation** from STEP/STL CAD files or analytic ports, including twisted and helical ports and multi-port grains. A fast 2D outline mode is also available.
- **Energy-conserving tank and chamber models.**
  - **Injector:** Dyer and HEM two-phase models.
  - **Nozzle expansion:** shifting chemical equilibrium, validated against NASA CEA.
- **Nozzle and plume CFD** with live contours, plus nozzle-wall heating over the burn.
- **Parallel parameter sweeps** written to CSV.
- **A GUI** for all of the above. Start it with `hrap`.

Results from older Python versions are not comparable: an error in reading the propellant chemistry tables affected every simulation (see the Python README).

## MATLAB - Hybrid Rocket Analysis Program
Navigate to the "Installers/MATLAB version" directory. Prerequisites for the installation include having the MATLAB runtime installed (free.) and in order to save/export data the application needs to be run as administrator.

## Validation Cases:
Baltic Space HyPEx hot fire (r/rocketry user FlyingBanana)

![image](https://github.com/user-attachments/assets/4e048de8-12b3-4299-89f5-ec4241b3ccb2)

Equatorial Space Systems 750 N subscale demonstrator (James Anderson)

![image](https://github.com/user-attachments/assets/63330cf6-2fe3-4712-a0ec-310baa33e389)

Student Space Technology Association 38mm Subscale Motor

![image](https://github.com/user-attachments/assets/508ee5ae-4a46-4c41-8743-3ec823128cfe)

Student Space Technology Association 127mm Flight Motor

![image](https://github.com/user-attachments/assets/89afa8e9-32a9-4057-92d3-aaae7cde900f)

## Call for volunteers!

In its current state the Python version models the following:
- an equilibrium (energy-conserving) oxidizer tank, with Dyer/HEM two-phase injectors
- 3D or 2D grain regression driven by local mass flux
- a mixture-based combustion chamber
- a nozzle with shifting-equilibrium or frozen expansion, including separation and unchoked flow
- semi-empirical correction factors such as combustion and nozzle efficiency

It also includes axisymmetric CFD of the nozzle and plume and a nozzle-wall heating analysis. HRAP can export .RSE and .ENG engine files for OpenRocket and RockSim.

HRAP could be significantly improved by some helpful volunteers in the following categories:
- Non-equilibrium tank model (separate liquid and vapor temperatures with heat and mass transfer)
- Viscous/turbulent CFD (boundary layers, plume mixing, separation) and ablating nozzle materials
- Equilibrium expansion with metallized fuels (condensed alumina)
- More config files describing burn characteristics of different propellants, and hot-fire data for validating regression laws
