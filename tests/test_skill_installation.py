from pathlib import Path

import pytest

from scripts.install_skills import SKILLS, install


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'checkout'
    for name in SKILLS:
        folder = root / 'skills' / name
        folder.mkdir(parents=True)
        (folder / 'SKILL.md').write_text(name)
    return root


def test_install_and_idempotent(checkout, tmp_path):
    dest = tmp_path / 'codex/skills'
    assert install(checkout, dest) is None
    assert install(checkout, dest) is None
    for name in SKILLS:
        assert (dest / name).is_symlink()
        assert (dest / name).resolve() == checkout / 'skills' / name


def test_conflict_preflight_makes_no_partial_changes(checkout, tmp_path):
    dest = tmp_path / 'skills'
    old = dest / SKILLS[-1]
    old.mkdir(parents=True)
    (old / 'custom.txt').write_text('keep')
    with pytest.raises(ValueError, match='Existing skill'):
        install(checkout, dest)
    assert list(dest.iterdir()) == [old]
    backup = install(checkout, dest, replace=True)
    assert (backup / SKILLS[-1] / 'custom.txt').read_text() == 'keep'


def test_missing_skill_prevents_registration(checkout, tmp_path):
    (checkout / 'skills' / SKILLS[-1] / 'SKILL.md').unlink()
    dest = tmp_path / 'skills'
    with pytest.raises(ValueError, match='Incomplete repository'):
        install(checkout, dest)
    assert not dest.exists()


def test_replace_dangling_symlink_preserves_it(checkout, tmp_path):
    dest = tmp_path / 'skills'
    dest.mkdir()
    target = dest / SKILLS[0]
    target.symlink_to(tmp_path / 'missing')
    backup = install(checkout, dest, replace=True)
    assert (backup / SKILLS[0]).is_symlink()
    assert (backup / SKILLS[0]).readlink() == tmp_path / 'missing'


def test_failed_link_rolls_back_old_installation(checkout, tmp_path, monkeypatch):
    dest = tmp_path / 'skills'
    old = dest / SKILLS[0]
    old.mkdir(parents=True)
    (old / 'custom.txt').write_text('keep')
    original = Path.symlink_to
    def fail_second(self, *args, **kwargs):
        if self.name == SKILLS[1]:
            raise OSError('synthetic link failure')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'symlink_to', fail_second)
    with pytest.raises(OSError, match='synthetic link failure'):
        install(checkout, dest, replace=True)
    assert (old / 'custom.txt').read_text() == 'keep'
    assert not (dest / SKILLS[1]).exists()
