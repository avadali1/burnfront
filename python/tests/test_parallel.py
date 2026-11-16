"""Parallel work started from a user script that has no ``if __name__ == '__main__':`` guard."""
import subprocess
import sys
import textwrap


def test_unguarded_script_can_build_tables_in_parallel(tmp_path):
    script = tmp_path/'user_script.py'
    marker = tmp_path/'runs.txt'
    script.write_text(textwrap.dedent(f"""
        from pathlib import Path
        with open({str(marker)!r}, 'a') as f:      # counts how many times this script body runs
            f.write('run\\n')
        from burnfront.expansion import compute_table
        tab = compute_table('ABS', [4.0, 6.0], [20e5], True, 2, False)
        assert tab.OF.shape == (2,)
        print('OK')
    """))
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert 'OK' in r.stdout
    assert marker.read_text().count('run') == 1          # the workers did not re-run the script
