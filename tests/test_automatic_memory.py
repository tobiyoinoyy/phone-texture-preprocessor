"""Platform RSS units and their persisted automatic-failure diagnostics."""
import json
import sys
from types import SimpleNamespace

import pytest

from phone_texture import automatic


@pytest.mark.parametrize('platform,raw,expected,unit', [
    ('linux', 7_277_568, 7_452_229_632, 'KiB'),
    ('darwin', 7_452_229_632, 7_452_229_632, 'bytes'),
    ('unknown-os', 7_277_568, None, None),
])
def test_peak_rss_normalizes_known_units_and_preserves_provenance(monkeypatch, platform, raw, expected, unit):
    calls = []
    monkeypatch.setattr(automatic.sys, 'platform', platform)
    monkeypatch.setattr(automatic.resource, 'getrusage',
        lambda who: calls.append(who) or SimpleNamespace(ru_maxrss=raw))
    measurement = automatic.peak_process_memory()
    assert calls == [automatic.resource.RUSAGE_SELF]
    assert measurement['peak_process_rss_bytes'] == expected
    assert measurement['peak_process_rss_measurement'] == dict(
        source='getrusage(RUSAGE_SELF).ru_maxrss', raw_value=raw, raw_unit=unit,
        platform=platform, scope='worker_process_lifetime')


def test_failed_sku_persists_normalized_rss_without_loading_a_model(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setattr(automatic.sys, 'platform', 'linux')
    monkeypatch.setattr(automatic.resource, 'getrusage', lambda _: SimpleNamespace(ru_maxrss=2048))
    def fail_intake(*args, **kwargs):
        raise ValueError('intentional intake failure')
    monkeypatch.setattr(automatic.w, 'intake', fail_intake)
    case = tmp_path / 'case'
    result = automatic.process_one(('SKU_01', str(tmp_path / 'source'), str(case), 'cpu', 1, 'iphone17pro'))
    saved = json.loads((case / 'result.json').read_text())
    assert saved == result
    assert saved['status'] == 'failed'
    assert saved['error'] == 'intentional intake failure'
    assert saved['six_faces_processed'] is False
    assert saved['peak_process_rss_bytes'] == 2_097_152
    assert saved['peak_process_rss_measurement']['raw_unit'] == 'KiB'
    assert not (case / 'candidate').exists()
