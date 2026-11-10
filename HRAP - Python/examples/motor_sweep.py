# Purpose: Run a motor with arbitrary starting conditions in 3D (or 2D) grain mode, and sweep parameters.
# Usage:   python motor_sweep.py                      (single 3D run of a twisted-star motor + a sweep)
#          python motor_sweep.py --mode 2d --jobs 4
import argparse
from pathlib import Path

from hrap.motor import MotorConfig, run_motor, sweep

if __name__ == '__main__':          # required for parallel sweeps on Windows (spawned workers)
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='3d', choices=['3d', '2d'])
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    out = Path('results'); out.mkdir(exist_ok=True)

    cfg = MotorConfig(
        mode=args.mode,
        propellant='ABS', phi_swirl=1.0,
        # Oxidizer tank: 3 L of N2O at 50 bar (saturation temperature from pressure), 85% liquid
        fluid='NitrousOxide', tank_V=3.0e-3, P_tank=50e5, T_tank=None, fill=0.85,
        inj_N=6, inj_d=1.5e-3, inj_Cd=0.65,
        # Grain: 6-point star twisted one full turn
        port={'type': 'star', 'n_tips': 6, 'r_valley': 0.010, 'r_tip': 0.018, 'twist_turns': 1.0},
        grain_OD=0.07, grain_L=0.30, n_xy=48, n_z=33,
        # Chamber and nozzle
        V_free=2.0e-4, cstar_eff=0.95, throat_d=0.015, ER=5.0, noz_Cd=0.98, noz_eff=0.97,
        T=10.0, dt=1e-3,
    )
    res = run_motor(cfg, snapshot_times=(1.0, 3.0), verbose=True)
    for k, v in res.summary().items():
        print(f'{k:20s} {v:.6g}')
    res.to_csv(out/'motor_history.csv')
    print('wrote', out/'motor_history.csv')

    rows = sweep(cfg, {'inj_N': [4, 6, 8], 'fill': [0.7, 0.9]}, n_jobs=args.jobs,
                 csv_path=out/'motor_sweep.csv', history_dir=out/'sweep_histories')
    print('wrote', out/'motor_sweep.csv')
