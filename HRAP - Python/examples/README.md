# Examples

Run these from this folder with the project's virtual environment active. Outputs are written to `./results/`.

| Script | What it does |
|---|---|
| `motor_sweep.py` | Runs one motor from arbitrary starting conditions (tank pressure, fill, injector, twisted-star 3D grain), prints the summary and writes its history CSV. Then runs a parallel sweep over injector holes and fill fraction into `motor_sweep.csv`. Use `--mode 2d` for the fast outline model and `--jobs N` to set the parallelism. |
| `step_grain_convergence.py` | Grid-convergence report for a STEP or STL grain: Celik et al. (2008) GCI on the three finest grids, plus a least-squares uncertainty over all grids. Run with no file, it generates a twisted-star STEP. Options: `--units`, `--axis`, `--res 32,48,64,96`. |
| `compare_legacy.py` | The original comparison of constant-O/F and shifting-O/F grains with empirical and CoolProp N2O properties (legacy engine parts). |
| `nitrous_plastisol.py` | Legacy API example: N2O with aluminized plastisol (constant-O/F grain), chemistry from the in-house equilibrium solver. |
| `Arbitrary_Port_Nos_Paraffin.py` | Legacy API example: N2O and paraffin with a port shape baked from `Grain.png` (shifting-O/F grain). |
