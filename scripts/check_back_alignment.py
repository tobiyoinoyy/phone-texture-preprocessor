"""Check exported back PNGs. Exit nonzero when any supplied image fails."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phone_texture.back_inspection import inspect_back


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('iphone17','iphone17pro','iphone17promax'))
    parser.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()
    passed = True
    for path in args.files:
        try:
            result = inspect_back(path, args.model)
        except (OSError, ValueError) as exc:
            result = dict(verified=False, error=str(exc))
        print(json.dumps(dict(file=str(path), **result), ensure_ascii=False))
        passed &= result['verified']
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
