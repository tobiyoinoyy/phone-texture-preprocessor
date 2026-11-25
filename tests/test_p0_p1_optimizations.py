"""Targeted unit tests for P0 performance/storage and P1 workflow/quality fixes."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from phone_texture import automatic, back_lens, lighting, source_cap_recognition as scr, stage_processing as stages, web_app, workspace as w
from phone_texture.shadow_lift_profile import NAME as SHADOW_PROFILE


def _bright_cap(*, with_port=True, with_holes=True, with_projection=False, wide_plateau=False):
    """Synthetic bright aluminum cap (gray=175 on bg=235) with optional features."""
    a = np.full((800, 800, 3), 235, np.uint8)
    a[370:440, 110:690] = 175
    if wide_plateau:
        # Full-width camera plateau ramping up inside [0.025..0.05] of bw=580 (14..29px),
        # leaving outer shoulders [0.008..0.028] (5..16px) flat at y=370.
        for x in range(128, 672):
            raise_px = 18 if x < 400 else 32
            a[370 - raise_px:370, x] = 175
        # Smooth ramp across x in [125..135] and [665..675] so [0.025..0.05] has err > 2.5
        for i, x in enumerate(range(125, 135)):
            a[370 - i * 2:370, x] = 175
        for i, x in enumerate(range(675, 665, -1)):
            a[370 - i * 3:370, x] = 175
    elif with_projection:
        a[355:370, 170:650] = 175
    if with_port:
        cv2.rectangle(a, (385, 392), (455, 415), (95, 95, 95), -1)
    if with_holes:
        for x in [240, 260, 280, 300, 320, 510, 530, 550, 570, 590]:
            cv2.circle(a, (x, 402), 6, (95, 95, 95), -1)
    return a


def test_source_cap_bright_bottom_hardware_overrides_distant_camera_projection():
    proof = scr.classify(_bright_cap(with_port=True, with_holes=True, with_projection=True))
    assert proof['face'] == 'bottom'
    assert proof['camera_projection'] is True
    assert len(proof['consistent_port_thresholds']) >= 2


def test_source_cap_wide_plateau_top_uses_outer_shoulder_fallback():
    proof = scr.classify(_bright_cap(with_port=False, with_holes=False, wide_plateau=True))
    assert proof['face'] == 'top'
    assert proof['upper_rail_mode'] == 'outer_shoulder_fallback'


def test_source_cap_bright_plain_still_returns_none():
    proof = scr.classify(_bright_cap(with_port=False, with_holes=False, with_projection=False))
    assert proof['face'] is None


def test_clone_file_is_isolated_and_respects_copy2_monkeypatch(tmp_path, monkeypatch):
    src = tmp_path / 'src.bin'
    dst = tmp_path / 'dst.bin'
    src.write_bytes(b'original-bytes')
    w.clone_file(src, dst)
    assert dst.read_bytes() == b'original-bytes'
    # Writing to destination must never mutate the cloned source.
    dst.write_bytes(b'modified-bytes')
    assert src.read_bytes() == b'original-bytes'

    calls = []
    def fake_copy2(a, b):
        calls.append((Path(a), Path(b)))
        Path(b).write_bytes(Path(a).read_bytes())
    monkeypatch.setattr(w.shutil, 'copy2', fake_copy2)
    dst2 = tmp_path / 'dst2.bin'
    w.clone_file(src, dst2)
    assert calls == [(src, dst2)]


def test_continuity_candidate_retries_rear_upper_join_on_both_metal_domains(tmp_path, monkeypatch):
    src = tmp_path / 'back.png'
    Image.new('RGBA', (16, 16), (180, 180, 180, 255)).save(src)
    m1 = tmp_path / 'camera_surround.png'
    m2 = tmp_path / 'main_surface.png'
    Image.new('L', (16, 16), 255).save(m1)
    Image.new('L', (16, 16), 255).save(m2)
    job = dict(
        views=[dict(
            name='back',
            source=str(src),
            domains=[
                dict(name='camera_surround', mask=str(m1), strength=[0.98, 0, 0]),
                dict(name='main_surface', mask=str(m2), strength=[0.98, 0, 0]),
            ],
        )]
    )
    job_path = tmp_path / 'job.json'
    job_path.write_text(json.dumps(job))
    seen_jobs = []
    def fake_run(material, cfg_path):
        cfg = json.loads(Path(cfg_path).read_text())
        seen_jobs.append(cfg)
        out = Path(cfg['output_dir']) / 'back.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGBA', (16, 16), (185, 185, 185, 255)).save(out)
        return dict(views={'back': {}})
    monkeypatch.setattr(lighting, 'run', fake_run)
    monkeypatch.setattr(stages, 'measure_effect', lambda *a, **kw: dict(original_effect_check_passed=True))

    err = ValueError('Lighting correction discontinuity at rear_upper_join_center: 1.40 L')
    cfg_path, _, out, effect = stages._continuity_candidate(
        err, job, job_path, 'back', src, {}, np.ones((16, 16), bool), np.zeros((16, 16), bool), SHADOW_PROFILE
    )
    assert len(seen_jobs) == 1
    qc = seen_jobs[0]['views'][0]['quantization_continuity']
    assert [entry['mask'] for entry in qc] == [str(m1), str(m2)]
    assert effect['continuity_retry']['method'] == 'rounding'
    assert out.is_file() and cfg_path.is_file()


def test_back_lens_plateau_clearance_uses_prelighting_and_dark_body_fallback(monkeypatch):
    from phone_texture import rear_shadow
    roi = (400, 400)
    lit = np.full((400, 400, 4), 255, np.uint8)
    lit[..., :3] = 100
    raw = np.full((400, 400, 4), 255, np.uint8)
    raw[..., :3] = 180
    proof = dict(corners=[dict(radius=10) for _ in range(4)])

    # Case 1: post-lighting fails strict, pre-lighting succeeds strict.
    def fake_core(L):
        if float(L.mean()) > 55:
            return [60, 50, 350, 260], proof
        raise rear_shadow.InsufficientRearShadowEvidence('lit plateau flat')
    monkeypatch.setattr(rear_shadow, '_plateau_core', fake_core)
    clearance, label = back_lens._plateau_clearance(lit, roi, plateau_source=raw)
    assert clearance.shape == roi and clearance.max() > 0
    assert label == 'pre_lighting_source_measured'

    # Case 2: both fail strict, pre-lighting succeeds dark-body top fallback.
    def always_fail(L):
        raise rear_shadow.InsufficientRearShadowEvidence('strict plateau failed')
    def fake_dark(L):
        if float(L.mean()) > 55:
            return [60, 58, 350, 260], proof
        raise rear_shadow.InsufficientRearShadowEvidence('dark lit failed')
    monkeypatch.setattr(rear_shadow, '_plateau_core', always_fail)
    monkeypatch.setattr(back_lens, '_dark_body_plateau_core', fake_dark)
    clearance, label = back_lens._plateau_clearance(lit, roi, plateau_source=raw)
    assert clearance.shape == roi and clearance.max() > 0
    assert label == 'pre_lighting_dark_body_source_measured'


def test_automatic_progress_summary_and_debug_evidence_validation(tmp_path):
    row = dict(
        sku='SKU_01', status='processed_needs_visual_review', seconds=12.5,
        error=None, six_faces_generated=True, six_faces_processed=True,
        stage_status=dict(geometry='complete'), lighting_checks={'big': [1] * 1000},
    )
    summary = automatic._progress_summary(row)
    assert 'lighting_checks' not in summary
    assert summary['sku'] == 'SKU_01' and summary['six_faces_processed'] is True
    with pytest.raises(ValueError, match='debug_evidence'):
        automatic.run(tmp_path, tmp_path / 'w', tmp_path / 'o', debug_evidence='yes')


def test_lighting_engine_is_cached():
    lighting.engine.cache_clear()
    first = lighting.engine('aluminum')
    second = lighting.engine('aluminum')
    assert first is second


def test_create_overview_truncates_long_sku_labels(tmp_path):
    long_sku = '17ProMax_06_silver_warn_1235181065_extra_long_identifier_name'
    sku_dir = tmp_path / 'out' / long_sku
    sku_dir.mkdir(parents=True)
    for face in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        Image.new('RGBA', (64, 128), (120, 140, 160, 255)).save(sku_dir / (face + '.png'))
    dest = tmp_path / 'overview.jpg'
    web_app.create_overview(tmp_path / 'out', [long_sku], dest)
    assert dest.is_file() and dest.stat().st_size > 0


def test_lens_exterior_resists_1deg_grain_hole_and_restores_deep_blue_chroma():
    from phone_texture.lens_exterior import lift

    y, x = np.ogrid[:340, :340]
    cx, cy, r = 160, 160, 96.0
    dist = np.hypot(x - cx, y - cy) - r
    ang = np.arctan2(y - cy, x - cx)
    flat = dist > 2
    # Deep-blue aluminum plateau (L=42, a=4, b=-19) with a pitch-black cast shadow
    # around 0 rad (L=2, a=0, b=-2 -> delta_b = -17, beyond old -10 clip).
    shade = np.exp(-((ang / 0.45) ** 2)) * np.exp(-(((dist - 7.5) / 5.0) ** 4))
    L = np.where(flat, 42.0 - 40.0 * shade, 10.0)
    a = np.where(flat, 4.0 - 4.0 * shade, 0.0)
    b = np.where(flat, -19.0 + 17.0 * shade, 0.0)
    # Inject 1-degree localized grain fluctuation ONLY in the outer reference band
    # (dist 20..29) right at angle=0 (simulating anodized aluminum grain that used
    # to trip norm(first - second) >= 5 on a 1-degree ray and punch a round dark spot).
    grain_band = flat & (dist >= 20) & (dist < 29) & (np.abs(ang) < np.deg2rad(0.9))
    L = np.where(grain_band, L + 6.5, L)
    raw = np.stack([L, a, b], axis=-1)
    lab = raw.copy()
    edge = np.full(360, r)
    ok = np.ones(360, bool)
    lens = dict(centre=[cx, cy, r], outer=edge, supported=ok)
    proof = lift(raw, lab, flat, lens, edge, ok)
    assert proof['status'] == 'applied'
    assert 0 not in proof['unsupported_degrees']
    core_spot = flat & (dist >= 6) & (dist <= 9) & (np.abs(ang) < 0.05)
    # Shadow must be lifted cleanly with no dark spot and blue chroma restored past -10.
    assert np.median(lab[..., 0][core_spot]) > 35.0
    assert np.median(lab[..., 2][core_spot]) < -14.0


def test_rear_shadow_plateau_core_accepts_dark_body_top_bevel_and_rejects_flat():
    from phone_texture import rear_shadow

    h, w = 1500, 720
    L = np.full((h, w), 30.0, dtype=float)
    # Draw dark-body plateau with strong left/right/bottom bevels and a weak-but-sharp
    # top bevel where one of the 3 top windows dips to ~0.9 L/px (median >= 1.2, ptp=0).
    x0, y0, x1, y1 = 45, 45, 675, 390
    L[y0:y1, x0:x1] = 44.0
    # Lower contrast above y0 in the third top window only so gradient stays at y0 with ~0.9 L/px
    L[:y0, int(w * 0.56):int(w * 0.66)] = 41.6
    # Add rounded corner contrast via circle masks on a 4-corner plateau
    yy, xx = np.ogrid[:h, :w]
    rad = 70
    for cx, cy, sx, sy in [(x0 + rad, y0 + rad, 1, 1), (x1 - rad, y0 + rad, -1, 1),
                           (x0 + rad, y1 - rad, 1, -1), (x1 - rad, y1 - rad, -1, -1)]:
        q = ((xx - cx) * sx < 0) & ((yy - cy) * sy < 0)
        outside = q & ((xx - cx) ** 2 + (yy - cy) ** 2 > rad ** 2)
        L[outside] = 30.0
    bounds, proof = rear_shadow._plateau_core(L)
    assert len(bounds) == 4
    assert np.ptp(proof['top']['positions']) <= 1
    assert 0.6 <= min(proof['top']['strengths']) < 1.2
    # Completely flat top edge must still fail closed.
    L[:y0 + 10, int(w * 0.38):int(w * 0.67)] = 44.0
    with pytest.raises(rear_shadow.InsufficientRearShadowEvidence):
        rear_shadow._plateau_core(L)
