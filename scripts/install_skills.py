"""Register the four bundled skills; preserve existing installations on replacement."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

SKILLS = ('phone-six-view-process', 'phone-texture-preprocessor',
          'phone-aluminum-frame-restore', 'phone-six-view-restore')


def install(root, dest, replace=False):
    root, dest = Path(root).resolve(), Path(dest).absolute()
    links = []
    for name in SKILLS:
        source, target = root / 'skills' / name, dest / name
        if not (source / 'SKILL.md').is_file():
            raise ValueError(f'Incomplete repository: {source}')
        if target.is_symlink() and target.resolve() == source:
            continue
        if os.path.lexists(target) and not replace:
            raise ValueError(f'Existing skill: {target}. Use --replace to back it up before replacement.')
        # Replacing an ancestor of the checkout would move our own source.
        if target.exists() and (target.resolve() == root or target.resolve() in root.parents):
            raise ValueError(f'Cannot replace a directory containing the repository: {target}')
        links.append((source, target))
    dest.mkdir(parents=True, exist_ok=True)
    backup = None
    if any(os.path.lexists(target) for _, target in links):
        parent = dest.parent / 'skill-backups'
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')
        backup = Path(tempfile.mkdtemp(prefix=stamp, dir=parent))
    completed, moved = [], []
    try:
        for source, target in links:
            if os.path.lexists(target):
                old = backup / target.name
                target.rename(old)
                moved.append((old, target))
            target.symlink_to(source, target_is_directory=True)
            completed.append(target)
    except Exception:
        for target in reversed(completed):
            target.unlink()
        for old, target in reversed(moved):
            old.rename(target)
        raise
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    codex = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')))
    parser.add_argument('--dest', type=Path, default=codex / 'skills')
    parser.add_argument('--replace', action='store_true', help='Back up existing skills before replacing them')
    args = parser.parse_args()
    try:
        backup = install(Path(__file__).resolve().parents[1], args.dest, args.replace)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'Four bundled skills ready: {args.dest}')
    if backup:
        print(f'Previous installations preserved: {backup}')


if __name__ == '__main__':
    main()
