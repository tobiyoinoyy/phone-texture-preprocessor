"""Run all bundled suites in isolated processes to avoid legacy module-name collisions."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    suites = (root, root / 'skills/phone-aluminum-frame-restore', root / 'skills/phone-six-view-restore')
    for suite in suites:
        print(f'Testing {suite.relative_to(root)}', flush=True)
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], cwd=suite)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
