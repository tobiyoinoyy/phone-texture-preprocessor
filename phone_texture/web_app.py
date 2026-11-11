"""Loopback-only browser shell for the original-only six-face pipeline.

No processing implementation lives here: every batch runs `phone-texture auto`
in an isolated process group. Only complete, checked batches get an export ZIP.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse
import webbrowser
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps
from . import __app_version__, workspace as w
from .pipeline import FACES, EXTENSIONS
from .view_recognition import inspect_images

ACTIVE = {'starting', 'running', 'stopping', 'packaging'}
LABELS = dict(front='正面', back='背面', left='左侧', right='右侧', top='顶面', bottom='底面')
REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name('web_assets')
GENERATED_STATUSES = {'processed_needs_visual_review', 'source_preserved_needs_review',
                      'full_stages_effect_warning_needs_review'}
SAVED_REVIEW_ASSEMBLY_MODE = 'saved_bound_face_outputs_review_v1'


def current_side_review_ready(done):
    """UI/retry closure only; producer checks remain authoritative, never certify quality."""
    def digest(value):
        return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
    def descriptor(value, executed=False):
        return (isinstance(value, dict) and isinstance(value.get('path'), str)
            and bool(value['path'].strip()) and digest(value.get('sha256'))
            and (not executed or value.get('executed') is True))
    evidence = done.get('current_side_review') or {}
    final = done.get('final_sha256') or {}
    stages = done.get('stage_status') or {}
    if not all(isinstance(value, dict) for value in (evidence, final, stages)):
        return False
    if (done.get('side_candidate_review_complete') is not True
            or evidence.get('schema') != 'current_side_review/v1'
            or evidence.get('execution_mode') != 'fresh_current_sources'
            or done.get('saved_candidate_review_assembly') is True
            or done.get('six_faces_generated') is not True
            or done.get('status') != 'source_preserved_needs_review'
            or done.get('six_faces_processed') is not False
            or done.get('strict_structure_verified') is not False
            or done.get('strict_endpoint_evidence_complete') is not False
            or done.get('structural_alignment_certified') is not False
            or done.get('quality_checks_complete') is not False
            or done.get('visual_review_passed') is not False
            or done.get('source_hashes_unchanged') is not True
            or stages.get('geometry') != 'complete'
            or stages.get('structure') != 'review_required'
            or stages.get('seam') == 'complete'
            or set(final) != set(FACES) or not all(digest(v) for v in final.values())):
        return False
    faces = evidence.get('faces') or {}
    details = done.get('per_face_stages') or {}
    if not isinstance(faces, dict) or not isinstance(details, dict) or set(faces) != {'left', 'right'}:
        return False
    for face in ('left', 'right'):
        selected, record = details.get(face, {}), faces[face]
        if (not isinstance(record, dict) or not isinstance(selected, dict)
                or record.get('selected_lighting') is not True
                or record.get('selected_structure') is not True
                or not descriptor(record.get('observation'))
                or not descriptor(record.get('lighting'), executed=True)
                or not descriptor(record.get('structure'), executed=True)
                or selected.get('lighting') not in ('complete', 'effect_warning', 'review_applied')
                or selected.get('structure') != 'review_required'
                or selected.get('selected_stage') != 'structure_review'
                or selected.get('structure_operation_applied') is not True
                or selected.get('strict_structure_verified') is not False
                or selected.get('selected_sha256') != final[face]):
            return False
    return True


def processing_state(done):
    """Describe performed stages, including old reports without per-stage fields."""
    generated = bool(done.get('six_faces_generated', done.get('six_faces_processed')))
    full_legacy = done.get('status') in ('processed_needs_visual_review', 'full_stages_effect_warning_needs_review')
    structure = done.get('structural_alignment_applied', bool(done.get('structure_checks')) or full_legacy)
    seams = done.get('seam_accepted', done.get('frame_uv_seam_checks', {}).get('status') == 'four_frame_uv_color_seams_passed' or full_legacy)
    details = done.get('per_face_stages', {})
    face_processing = dict(done.get('face_processing') or {})
    for face in FACES:
        if face not in face_processing:
            face_processing[face] = details.get(face, {}).get('selected_stage') or done.get('preserved_faces', {}).get(face, {}).get('stage')
        if not face_processing[face] and full_legacy:
            face_processing[face] = 'seam' if face in ('left', 'right', 'top', 'bottom') else 'lighting'
        if not face_processing[face]:
            face_processing[face] = 'unknown'
    lighting_faces = [f for f in FACES if details.get(f, {}).get('lighting') in ('complete', 'effect_warning', 'review_applied')
                      or f in done.get('lighting_checks', {}) or full_legacy]
    stages = dict(geometry='complete' if generated else 'unknown',
                  lighting='complete' if len(lighting_faces) == 6 else 'partial' if lighting_faces else 'unknown',
                  structure='complete' if structure else 'blocked' if generated else 'unknown',
                  seam='complete' if seams else 'blocked' if generated else 'unknown')
    stages.update(done.get('stage_status') or {})
    # Explicit current stage evidence takes precedence over legacy status names.
    structure = stages['structure'] == 'complete'
    # A selected review remap is a performed operation, not strict certification.
    # Require matching per-face facts; a top-level list alone is insufficient.
    declared_review_faces = done.get('structural_review_applied_faces') or []
    review_applied_faces = [face for face in ('left', 'right')
        if face in declared_review_faces
        and details.get(face, {}).get('structure_operation_applied') is True
        and details.get(face, {}).get('structure') == 'review_required'
        and details.get(face, {}).get('selected_stage') == 'structure_review']
    operation_applied = structure or bool(review_applied_faces)
    all_stages = all(stages[k] == 'complete' for k in ('geometry', 'lighting', 'structure', 'seam'))
    final_hashes = done.get('final_sha256') or {}
    saved_review = (done.get('saved_candidate_review_assembly') is True
        and done.get('assembly_mode') == SAVED_REVIEW_ASSEMBLY_MODE
        and done.get('review_continuation_saved') is True
        and done.get('automatic_retry_eligible') is False
        and generated and done.get('status') == 'source_preserved_needs_review'
        and done.get('six_faces_processed') is False
        and done.get('quality_checks_complete') is False
        and done.get('visual_review_passed') is False
        and done.get('strict_structure_verified') is False
        and set(final_hashes) == set(FACES)
        and all(isinstance(v, str) and len(v) == 64
                and all(c in '0123456789abcdef' for c in v) for v in final_hashes.values()))
    fresh_review = current_side_review_ready(done)
    continuation_ready = saved_review or fresh_review
    retry_required = not all_stages and not continuation_ready
    warnings = list(done.get('review_warnings') or [])
    if structure:
        warnings = [x for x in warnings if x.get('code') != 'source_positions_preserved_alignment_unverified']
    brightness_kinds = {'lighting_effect_target_warning', 'local_brightness', 'perimeter_brightness'}
    for item in done.get('lighting_effect_warnings', []):
        entries = list(item.get('warnings') or [item])
        if item.get('kind') == 'normal_coverage_unavailable' and not any(e.get('kind') == item['kind'] for e in entries):
            entries.insert(0, item)
        if item.get('effect_failures') and not any(not e.get('kind') or e.get('kind') in brightness_kinds for e in entries):
            entries.append(dict(kind='lighting_effect_target_warning',
                error='; '.join(str(e.get('error', '')) for e in item['effect_failures'])))
        for entry in entries:
            kind = entry.get('kind')
            if kind == 'normal_coverage_unavailable':
                code, message = kind, '局部法向对照不足，保留已调光候选待核对'
            elif kind in ('normal_continuity_warning','quantization_compatible_pixel_warning'):
                code,message=kind,entry.get('message') or '局部明暗变化需核对'
            elif not kind or kind in brightness_kinds:
                code, message = 'lighting_effect_target_warning', '局部提亮未达预定目标，保留已处理候选待核对'
            else:
                code, message = kind, entry.get('message') or entry.get('error') or '局部光照检查待核对'
            warnings.append(dict(face=entry.get('face', item.get('face')), code=code,
                                 message=message, detail=entry.get('error')))
    for item in done.get('item_effect_warnings', []):
        code = item.get('code') or item.get('kind') or 'item_effect_review_required'
        message = item.get('message') or item.get('error') or '处理效果待核对'
        if not any(w.get('code')==code and w.get('face')==item.get('face') and w.get('message')==message for w in warnings):
            warnings.append(dict(face=item.get('face'),code=code,message=message,detail=item.get('error')))
    reasons = []
    blocks = done.get('blocked_checks') or ([done['blocked_check']] if done.get('blocked_check') else [])
    for item in blocks:
        error = item.get('error', '')
        reasons.append(dict(stage=item.get('stage', item.get('kind')), face=item.get('face'),
                            message=failure_hint(error), detail=error))
    for item in warnings:
        if (item.get('code') not in ('source_positions_preserved_alignment_unverified', 'optional_processing_check_failed')
                and not (item.get('code', '').endswith('_blocked') and any(item.get('message') == b.get('error') for b in blocks))):
            reasons.append(dict(face=item.get('face'), message=item.get('message'), detail=item.get('detail')))
    if generated:
        if all_stages:
            # 2026-11-06 user decision: full-stage results pass directly; warnings stay as notes.
            summary = '调光、天线对齐和接缝处理已完成'
        else:
            if len(lighting_faces) == 6 and any(details.get(f, {}).get('lighting') == 'review_applied' for f in FACES):
                light = '六面已应用调光候选，部分待核对'
            else:
                light = '六面已调光' if stages['lighting'] == 'complete' else '部分面已调光' if lighting_faces else '调光未完成'
            if structure:
                alignment = '天线已对齐'
            elif len(review_applied_faces) == 2:
                alignment = '天线位置已调整，待核对'
            elif review_applied_faces:
                alignment = ('左侧' if review_applied_faces[0] == 'left' else '右侧') + '天线位置已调整，另一侧未调整，待核对'
            else:
                alignment = '天线未对齐'
            summary = light + ' · ' + alignment + (' · 接缝未完成' if stages['seam'] != 'complete' else '')
    else:
        summary = None
    bottom = done.get('bottom_alignment') or {}
    bottom_summary = {'applied_with_notice':'底面已校正',
        'unchanged':'底面无需校正', 'kept_source_with_notice':'底面保持原图位置',
        'disabled':'底面校正已关闭'}.get(bottom.get('status'))
    if summary and bottom_summary:
        summary += ' · ' + bottom_summary
    return dict(bottom_alignment=bottom, bottom_alignment_summary=bottom_summary,
                stage_status=stages, structural_alignment_applied=operation_applied,
                structural_review_applied_faces=review_applied_faces,
                structure_operation_status='applied_needs_review' if review_applied_faces else 'complete' if structure else 'not_applied',
                strict_structure_verified=done.get('strict_structure_verified', structure),
                all_stages_complete=all_stages, retry_required=retry_required,
                automatic_retry_eligible=retry_required, review_continuation_saved=saved_review,
                current_side_review_ready=fresh_review, review_continuation_ready=continuation_ready,
                review_only=generated and all_stages, stage_summary=summary,
                face_processing=face_processing, per_face_stages=details,
                review_warnings=warnings, review_reasons=reasons)


def verified_export_rows(report, job):
    """Fail closed on missing/mutated exports; warning candidates remain included."""
    partial = report.get('status') == 'partial_candidates_needs_visual_review'
    review = report.get('status') == 'review_candidates_needs_visual_review'
    rows = [row for row in report.get('attempts', [])
            if row.get('status') in GENERATED_STATUSES | {None}
            and row.get('six_faces_generated', row.get('six_faces_processed', not partial and not review))]
    names = [row.get('sku') for row in rows]
    expected = report.get('generated', report.get('processed'))
    if (report.get('status') not in ('processed_needs_visual_review', 'partial_candidates_needs_visual_review', 'review_candidates_needs_visual_review')
            or not rows or expected != len(rows) or len(names) != len(set(names))
            or not set(names) <= set(job['names']) or (not partial and len(rows) != len(job['names']))):
        raise ValueError('候选完整性检查未通过，未生成下载。')
    if 'exported_skus' in report and (len(report['exported_skus']) != len(names) or set(report['exported_skus']) != set(names)):
        raise ValueError('导出清单与已生成六面候选不一致，未生成下载。')
    for row in rows:
        if set(row.get('final_sha256', {})) != set(FACES):
            raise ValueError('六面校验记录不完整：' + row['sku'])
        for face in FACES:
            path = Path(job['output']) / row['sku'] / (face + '.png')
            if not path.is_file():
                raise ValueError('导出缺少图片：' + row['sku'] + '/' + face + '.png')
            if w.sha(path) != row['final_sha256'][face]:
                raise ValueError('结果校验不一致：' + row['sku'] + '/' + face + '.png')
            with Image.open(path) as im:
                if im.mode != 'RGBA' or im.format != 'PNG':
                    raise ValueError('导出图片不是 RGBA PNG：' + row['sku'] + '/' + face)
                im.verify()
    return rows


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def save(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def within(path, parent):
    return path == parent or parent in path.parents


def host_sample():
    import psutil
    memory = psutil.virtual_memory()
    pressure = None
    if sys.platform == 'darwin':
        pressure = int(subprocess.check_output(
            ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True, timeout=3).strip())
    return dict(pressure=pressure, available=memory.available, total=memory.total,
                swap=psutil.swap_memory().used, time=time.time())


def guard_reason(sample, initial):
    if sample['pressure'] is not None and sample['pressure'] >= 4:
        return '系统内存压力过高，已停止本批处理。请关闭占内存的程序或降低并发后重试。'
    if sample['swap'] - initial['swap'] > 8 * 1024**3:
        return '本批交换内存增加超过 8 GB，已停止。请降低并发后重试。'
    if sample['available'] < 1024**3:
        return '可用内存不足 1 GB，已停止。请释放内存后重试。'
    return None


R9_GUARD_LOCK = Path('/tmp/phone-texture-global-processing.lock')
R9_GUARD_OWNER = Path('/tmp/phone-texture-global-processing.owner.json')


def r9_guard_context():
    """Validate the explicit external guard contract; ordinary launches stay ordinary."""
    if os.environ.get('PHONE_TEXTURE_GUARD_R9') != '1':
        return None
    import psutil
    fd = int(os.environ.get('PHONE_TEXTURE_GLOBAL_LOCK_FD', '-1'))
    if fd < 0:
        raise ValueError('R9 guard lock descriptor is absent')
    descriptor, canonical = os.fstat(fd), R9_GUARD_LOCK.stat()
    if (descriptor.st_dev, descriptor.st_ino) != (canonical.st_dev, canonical.st_ino):
        raise ValueError('R9 guard canonical lock identity changed')
    owner = json.loads(R9_GUARD_OWNER.read_text())
    run_id = os.environ.get('PHONE_TEXTURE_GUARD_R9_RUN_ID')
    current = psutil.Process()
    if (not run_id or owner.get('version') != 'owned_ancestry_guard_r9/v1'
            or owner.get('run_id') != run_id or owner.get('state') != 'running'
            or owner.get('child_pid') != os.getpid()
            or owner.get('child_create_time') != current.create_time()
            or owner.get('wrapper_pid') != current.ppid()):
        raise ValueError('R9 guard process ancestry or run identity changed')
    parent = psutil.Process(current.ppid())
    if parent.create_time() != owner.get('wrapper_create_time'):
        raise ValueError('R9 guard wrapper identity changed')
    if os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(name) != '1' for name in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('R9 guard low-resource contract missing')
    # This same inherited open-file-description must own the exclusive lock.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return dict(run_id=run_id, lock_fd=fd, wrapper_pid=parent.pid,
                wrapper_create_time=parent.create_time(), paused_at_start=float(owner['paused_seconds']))


def r9_paused_seconds(context):
    if context is None:
        return 0.
    owner = json.loads(R9_GUARD_OWNER.read_text())
    if owner.get('run_id') != context['run_id'] or owner.get('wrapper_create_time') != context['wrapper_create_time']:
        raise ValueError('R9 guard owner changed during processing')
    value = owner.get('paused_seconds')
    if type(value) not in (float, int) or not 0 <= value < float('inf'):
        raise ValueError('Invalid guard pause duration')
    return max(0., value-context['paused_at_start'])


def r9_worker_identity(proc):
    import psutil
    if type(proc.pid) is not int or proc.pid <= 0:
        raise ValueError('Invalid child PID; no process group may be signaled')
    process = psutil.Process(proc.pid)
    created = process.create_time()
    if process.ppid() != os.getpid():
        raise ValueError('Worker is no longer a verified direct child')
    proc._r9_owned = {proc.pid: (created, process)}
    return dict(child_pid=proc.pid, child_create_time=created,
                child_pgid=os.getpgid(proc.pid), child_sid=os.getsid(proc.pid))


def r9_worker_members(proc):
    """Track live ancestry; retained identities remain owned after reparenting."""
    import psutil
    ledger = proc._r9_owned
    pending = list(ledger); visited = set(); alive = []
    while pending:
        pid = pending.pop(0)
        if pid in visited:
            continue
        visited.add(pid); created, process = ledger[pid]
        try:
            if process.create_time() != created or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            alive.append(process)
            for child in process.children(recursive=False):
                if child.pid <= 0 or child.pid == os.getpid():
                    raise ValueError('Unsafe descendant identity')
                child_created = child.create_time()
                if child.ppid() != pid or not process.is_running() or process.create_time() != created:
                    continue
                if child_created < created:
                    raise ValueError('Invalid worker descendant creation time')
                if child.pid in ledger and ledger[child.pid][0] != child_created:
                    raise ValueError('Worker descendant PID was reused')
                ledger[child.pid] = (child_created, child); pending.append(child.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return alive


def r9_stop_worker(proc):
    """Only recorded identities, never a process group; outer guard also audits cleanup."""
    import psutil
    members = r9_worker_members(proc)
    for process in reversed(members):
        try:
            if process.pid > 0 and process.create_time() == proc._r9_owned[process.pid][0]:
                process.terminate()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    _, alive = psutil.wait_procs(members, timeout=5)
    for process in alive:
        try:
            if process.pid > 0 and process.create_time() == proc._r9_owned[process.pid][0]:
                process.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    proc.wait(timeout=5)


def stop_group(proc):
    if type(proc.pid) is not int or proc.pid <= 0:
        raise ValueError('Invalid process PID; refusing any group signal')
    if hasattr(proc, "_r9_owned"):
        return r9_stop_worker(proc)
    # The leader may have exited while its workers still exist. Always address
    # the entire original group, including on cancellation and server shutdown.
    import psutil
    members = []
    for p in psutil.process_iter(['pid']):
        try:
            if p.pid > 0 and os.getpgid(p.pid) == proc.pid and p.status() != psutil.STATUS_ZOMBIE:
                members.append(p)
        except (ProcessLookupError, PermissionError, psutil.Error):
            pass
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can return EPERM for an already dying group. Signal retained
        # process handles instead of assuming EPERM proves a live worker.
        for p in members:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
    _, alive = psutil.wait_procs(members, timeout=5)
    for p in alive:
        try:
            if p.status() != psutil.STATUS_ZOMBIE:
                p.kill()
        except psutil.NoSuchProcess:
            pass
    proc.wait(timeout=5)


def scan_folder(value):
    source = Path(value).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError('请选择文件夹。')
    direct = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith('.')]
    subfolders = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.'))
    single = any(p.stem.lower() in FACES for p in direct) or (bool(direct) and not subfolders) or len(direct)==6
    folders = [source] if single else sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if len(folders) > 500:
        raise ValueError('单批最多扫描 500 个子文件夹，请分批选择。')
    rows = []
    for folder in folders:
        files = {}; errors = []
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith('.'))
        for p in paths:
            if p.stem.lower() in FACES:
                face = p.stem.lower()
                if face in files:
                    errors.append(f'{LABELS[face]}存在重复文件')
                files[face] = str(p)
        missing = [LABELS[f] for f in FACES if f not in files]
        if missing:
            errors.append('缺少：' + '、'.join(missing))
        recognition = inspect_images(paths)
        identified = recognition['complete']
        changes = [dict(source=r['name'], target=r['face']+Path(r['file']).suffix.lower())
                   for r in recognition['images'] if r['face'] and Path(r['file']).stem.lower()!=r['face']]
        needs_rename = identified and bool(changes)
        if identified:
            errors = [] if not needs_rename else ['文件名与图片方向不符']
        else:
            errors = recognition['errors'] + errors
        rows.append(dict(name=folder.name, files=files, valid=identified and not needs_rename,
                         can_rename=needs_rename, changes=changes, recognition=recognition, errors=errors))
    if not rows:
        raise ValueError('没有找到六面图片。请选择含六张原图的文件夹，或包含多个 SKU 的父文件夹。')
    return dict(source=str(source), single=single, rows=rows,
                valid=sum(r['valid'] for r in rows), renameable=sum(r['can_rename'] for r in rows), total=len(rows))


_FOLDER_PICKER_LOCK = threading.Lock()


def pick_system_folder(kind, initial=None):
    """Use macOS NSOpenPanel; paths are arguments, never executable source."""
    if sys.platform != 'darwin':
        raise ValueError('此版本的系统文件夹选择器适用于 macOS。')
    if kind not in ('input', 'output'):
        raise ValueError('无效的文件夹类型。')
    if not _FOLDER_PICKER_LOCK.acquire(blocking=False):
        raise ValueError('请先完成当前文件夹选择。')
    try:
        folder = Path(initial or Path.home()/'Documents').expanduser()
        if not folder.is_dir():
            folder = Path.home()/'Documents'
        title = '选择原图文件夹' if kind == 'input' else '选择导出文件夹'
        script = """function run(argv) {
            ObjC.import('AppKit');
            ObjC.import('Foundation');
            var app = $.NSApplication.sharedApplication;
            app.setActivationPolicy(1);
            var panel = $.NSOpenPanel.openPanel;
            panel.setTitle(argv[0]);
            panel.setPrompt('选择');
            panel.setCanChooseDirectories(true);
            panel.setCanChooseFiles(false);
            panel.setAllowsMultipleSelection(false);
            panel.setCanCreateDirectories(true);
            panel.setDirectoryURL($.NSURL.fileURLWithPath(argv[1]));
            app.activateIgnoringOtherApps(true);
            if (panel.runModal === $.NSModalResponseOK) {
                return ObjC.unwrap(panel.URL.path);
            }
            return '';
        }"""
        result = subprocess.run(['/usr/bin/osascript', '-l', 'JavaScript', '-e', script, title, str(folder)],
                                capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise ValueError('系统文件夹选择器未能打开：'+result.stderr.strip())
        selected = result.stdout.strip()
        return dict(path=str(Path(selected).resolve()) if selected else None, cancelled=not bool(selected))
    finally:
        _FOLDER_PICKER_LOCK.release()


def list_folders(value=None):
    """Read directory names only; selection stays local and needs no OS dialog."""
    folder = Path(value or Path.home()/'Documents').expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('这不是文件夹，请重新选择。')
    rows = []
    for p in folder.iterdir():
        try:
            if not p.name.startswith('.') and p.is_dir():
                rows.append(dict(name=p.name, path=str(p)))
        except PermissionError:
            continue
    return dict(path=str(folder), parent=str(folder.parent), directories=sorted(rows, key=lambda r:r['name'].casefold()),
                shortcuts=[dict(name=label, path=str(Path.home()/suffix)) for label,suffix in [('主文件夹',''),('文稿','Documents'),('桌面','Desktop')]])


def create_overview(output, names, dest):
    # Group the same face together, as required by the project's review rules.
    cols = min(5, len(names)); cell_w, cell_h = 260, 240
    rows_per_face = (len(names) + cols - 1) // cols
    canvas = Image.new('RGB', (cols*cell_w+40, 6*(rows_per_face*cell_h+50)+30), '#f3f4f2')
    draw = ImageDraw.Draw(canvas)
    font_path = '/System/Library/Fonts/STHeiti Light.ttc'
    font = ImageFont.truetype(font_path, 18) if Path(font_path).exists() else ImageFont.load_default()
    y = 15
    for face in FACES:
        draw.text((20, y), LABELS[face]+' / '+face, font=font, fill='#172a2a'); y += 40
        for i, sku in enumerate(names):
            x, yy = 20+(i % cols)*cell_w, y+(i // cols)*cell_h
            draw.rounded_rectangle((x, yy, x+cell_w-12, yy+cell_h-12), radius=10, fill='#e0e5e3')
            with Image.open(output/sku/(face+'.png')) as original:
                im = original.convert('RGBA'); im.thumbnail((cell_w-32, cell_h-55))
                canvas.paste(im, (x+(cell_w-12-im.width)//2, yy+10+(cell_h-55-im.height)//2), im)
            draw.text((x+12, yy+cell_h-38), sku, fill='#172a2a', font=font)
        y += rows_per_face*cell_h+10
    canvas.save(dest, quality=90)


def saved_review_binding(value, expected_sha256=None):
    """Pin explicit delivery configuration; schema/source validation is in auto."""
    if value is None:
        if expected_sha256 is not None:
            raise ValueError('保存候选清单 SHA-256 必须与清单路径一同提供')
        return None, None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError('保存候选清单必须是明确文件路径或 null')
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('保存候选清单不是文件')
    digest = w.sha(path)
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha256)):
            raise ValueError('保存候选清单 SHA-256 格式无效')
        if digest != expected_sha256:
            raise ValueError('保存候选清单已改变，未开始处理')
    return str(path), digest


def geometry_resume_path(value, required=False):
    if value is None:
        if required:
            raise ValueError('必须复用几何时需显式提供检查点过程目录')
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError('几何续接目录必须是明确路径或 null')
    path = Path(value).expanduser().resolve(strict=True)
    if not (path/'cases').is_dir():
        raise ValueError('几何续接目录缺少 cases；未启动重新分割')
    return str(path)


class App:
    def __init__(self, home, *, saved_review_manifest=None,
                 saved_review_manifest_sha256=None, source_body_reference_review=False,
                 side_candidate_review=False, resume_geometry_from=None, require_geometry_resume=False):
        if type(source_body_reference_review) is not bool:
            raise ValueError('source_body_reference_review 必须是明确的布尔值')
        self.source_body_reference_review = source_body_reference_review
        if type(side_candidate_review) is not bool or type(require_geometry_resume) is not bool:
            raise ValueError('侧面复核与必需几何续接选项必须是明确布尔值')
        self.side_candidate_review = side_candidate_review
        self.require_geometry_resume = require_geometry_resume
        self.resume_geometry_from = geometry_resume_path(resume_geometry_from, require_geometry_resume)
        self.saved_review_manifest, self.saved_review_manifest_sha256 = saved_review_binding(
            saved_review_manifest, saved_review_manifest_sha256)
        self.home = Path(home).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.registry = self.home/'任务记录'; self.registry.mkdir(exist_ok=True)
        self.default_output = self.home/'处理结果'; self.default_output.mkdir(exist_ok=True)
        self.runs = self.home/'运行记录'; self.runs.mkdir(exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.scans = {}; self.jobs = {}; self.cancel_events = {}; self.threads = {}
        for p in sorted(self.registry.glob('*.json')):
            job = read(p)
            if not job.get('id'):
                continue
            if job['status'] in ACTIVE:
                # Resume is explicit and always creates a separate batch.
                job.update(status='interrupted', message='上次服务已中断。可重新处理未完成项目。', ended=time.time())
                save(p, job)
            self.jobs[job['id']] = job

    def update(self, job, **values):
        with self.lock:
            job.update(values); save(self.registry/(job['id']+'.json'), job)

    def scan(self, value):
        result = scan_folder(value); result['id'] = secrets.token_hex(12)
        with self.lock:
            self.scans[result['id']] = result
            if len(self.scans) > 32:
                del self.scans[next(iter(self.scans))]
        return result

    def apply_names(self, scan_id):
        with self.lock:
            scan = self.scans.get(scan_id)
            if not scan:
                raise ValueError('请重新读取文件夹。')
            if scan.get('normalized_scan_id') in self.scans:
                return self.scans[scan['normalized_scan_id']]
            if any(j['status'] in ACTIVE for j in self.jobs.values()):
                raise ValueError('请等待当前批次完成。')
            if not any(r['can_rename'] for r in scan['rows']):
                raise ValueError('没有需要命名的图片。')
            eligible = [r for r in scan['rows'] if r['valid'] or r['can_rename']]
            if len(eligible)>100:
                raise ValueError('每次最多命名 100 台，请分批选择文件夹。')
            for row in eligible:
                for item in row['recognition']['images']:
                    if w.sha(Path(item['file'])) != item['sha256']:
                        raise ValueError('图片已改变，请重新读取后命名。')
            base = self.home/'命名副本'
            if within(base.resolve(), Path(scan['source'])):
                raise ValueError('原图目录包含工具目录，请选择具体的原图文件夹。')
            base.mkdir(exist_ok=True)
            stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3)
            pending = base/('.'+stamp); pending.mkdir()
            final = base/stamp
            manifest = dict(source=scan['source'], profile='iphone17-studio-views-v1', original_files_renamed=False,
                            rows=[], skipped=[r['name'] for r in scan['rows'] if r not in eligible])
            try:
                for row in eligible:
                    folder = pending/row['name']; folder.mkdir()
                    for item in row['recognition']['images']:
                        original = Path(item['file']); target = folder/(item['face']+original.suffix.lower())
                        shutil.copy2(original, target)
                        if w.sha(original)!=item['sha256'] or w.sha(target)!=item['sha256']:
                            raise ValueError('复制校验失败，请重新读取。')
                        manifest['rows'].append(dict(sku=row['name'], original=str(original), sha256=item['sha256'],
                                                     named=str(target.relative_to(pending)), recognition=item))
                save(pending/'命名记录.json', manifest)
                pending.rename(final)
            except Exception:
                shutil.rmtree(pending, ignore_errors=True)
                raise
            result = self.scan(final)
            result.update(named_copy=True, skipped=manifest['skipped'])
            scan['normalized_scan_id'] = result['id']
            return result

    def start(self, body):
        with self.lock:
            if any(j['status'] in ACTIVE for j in self.jobs.values()):
                raise ValueError('已有批次正在运行，请等待完成或停止后再开始。')
            scan = self.scans.get(body.get('scan_id'))
            if not scan:
                raise ValueError('文件夹扫描已失效，请重新检查。')
            names, jobs = body.get('skus'), body.get('concurrency')
            # 2026-11-06 user decision: concurrency unlocked back to the 0.1.1 range; memory guard still applies.
            if type(jobs) is not int or not 1 <= jobs <= 8:
                raise ValueError('并发数需为 1–8。')
            valid = {r['name'] for r in scan['rows'] if r['valid']}
            if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names) or len(names) != len(set(names)) or not set(names) <= valid:
                raise ValueError('请选择至少一个检查通过的 SKU。')
            if len(names) > 100:
                raise ValueError('预览版每批最多处理 100 台，请分批勾选。')
            resume_job = self.jobs.get(body.get('resume_job_id')) if body.get('resume_job_id') else None
            if body.get('resume_job_id') and (resume_job is None or resume_job['status'] in ACTIVE):
                raise ValueError('无法续接该批次，请等待处理结束。')
            if resume_job and (scan['source'] != resume_job['source'] or not set(names) <= set(self.retry_names(resume_job))):
                raise ValueError('续接范围必须是原批次中处理阶段未完成的项目。')
            # New jobs use only explicitly configured/requested continuation.
            # A retry keeps its original options even if service defaults changed.
            default_reference = (resume_job.get('source_body_reference_review', False)
                                 if resume_job else self.source_body_reference_review)
            reference_review = body.get('source_body_reference_review', default_reference)
            if type(reference_review) is not bool:
                raise ValueError('source_body_reference_review 必须是明确的布尔值')
            if resume_job and reference_review != default_reference:
                raise ValueError('续接必须保留原批次的本机参考选项')
            default_side = resume_job.get('side_candidate_review', False) if resume_job else self.side_candidate_review
            side_review = body.get('side_candidate_review', default_side)
            default_required = resume_job.get('require_geometry_resume', False) if resume_job else self.require_geometry_resume
            required_resume = body.get('require_geometry_resume', default_required)
            if type(side_review) is not bool or type(required_resume) is not bool:
                raise ValueError('侧面复核与必需几何续接选项必须是明确布尔值')
            if resume_job and (side_review != default_side or required_resume != default_required):
                raise ValueError('重试必须保留原批次的侧面复核与几何续接选项')
            default_resume = resume_job['work'] if resume_job else self.resume_geometry_from
            requested_resume = body.get('resume_geometry_from', default_resume)
            resume_path = geometry_resume_path(requested_resume, required_resume)
            if resume_job and resume_path != str(Path(default_resume).resolve()):
                raise ValueError('重试必须使用原批次的几何检查点')
            default_manifest = (resume_job.get('saved_review_manifest') if resume_job
                                else self.saved_review_manifest)
            default_digest = (resume_job.get('saved_review_manifest_sha256') if resume_job
                              else self.saved_review_manifest_sha256)
            requested_manifest = body.get('saved_review_manifest', default_manifest)
            requested_digest = body.get('saved_review_manifest_sha256',
                default_digest if requested_manifest == default_manifest else None)
            manifest_path, manifest_digest = saved_review_binding(requested_manifest, requested_digest)
            if resume_job and (manifest_path != default_manifest or manifest_digest != default_digest):
                raise ValueError('续接必须保留原批次的保存候选清单及其校验值')
            if side_review and (not reference_review or manifest_path is not None):
                raise ValueError('新侧面复核需显式本机参考，且不能同时选择保存候选清单')
            from .frame_seams import profile
            model = body.get('model')
            profile(model)
            # Recheck the selection just before start, including duplicates and
            # image corruption introduced since the scan.
            source = Path(scan['source'])
            for name in names:
                w.face_files(source if scan['single'] else source/name)
                row = next(r for r in scan['rows'] if r['name']==name)
                for item in row['recognition']['images']:
                    if w.sha(Path(item['file']))!=item['sha256']:
                        raise ValueError('图片已改变，请重新读取以检查方向。')
            output_base = Path(body.get('output') or self.default_output).expanduser().resolve()
            if within(output_base, source) or within(source, output_base):
                raise ValueError('保存目录与原图目录不能互相包含，请选择独立的结果目录。')
            if within(output_base, self.runs) or within(self.runs, output_base):
                raise ValueError('导出文件夹需与工具运行记录分开，请选择其他文件夹。')
            if within(source, self.runs) or within(self.runs, source):
                raise ValueError('原图文件夹需与工具运行记录分开，请选择具体的原图文件夹。')
            output_base.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(output_base).free < max(2*1024**3, len(names)*160*1024**2):
                raise ValueError('保存目录空间不足，请至少预留 2 GB，较大批次每台预留 160 MB。')
            if shutil.disk_usage(self.runs).free < max(2*1024**3, len(names)*160*1024**2):
                raise ValueError('工具运行记录所在磁盘空间不足，请至少预留 2 GB。')
            sample = host_sample()
            reason = guard_reason(sample, sample)
            if reason:
                raise ValueError(reason.replace('已停止本批处理', '暂不能开始'))
            stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3)
            root = self.runs/stamp; root.mkdir()
            job = dict(id=stamp, status='starting', created=time.time(), source=str(source), single=scan['single'],
                       names=names, model=model, protection_policy='none', review_fallback='preserve-source', lighting_profile='accepted-shadow-lift-v2', bottom_alignment=resume_job.get('bottom_alignment','appearance') if resume_job else 'appearance', concurrency=min(jobs, len(names)), requested_concurrency=jobs,
                       source_body_reference_review=reference_review, side_candidate_review=side_review,
                       resume_geometry_from=resume_path, require_geometry_resume=required_resume,
                       cpu_threads=1, inference_slots=min(jobs, len(names), 2), saved_review_manifest=manifest_path,
                       saved_review_manifest_sha256=manifest_digest,
                       root=str(root), output=str(output_base/stamp), output_base=str(output_base), work=str(root/'过程'),
                       message='准备处理', elapsed=0, host=sample)
            if resume_job:
                job.update(resume_geometry_from=resume_job['work'], parent_job_id=resume_job['id'],
                           message='准备继续未完成步骤；复用已校验的本批几何结果')
            self.jobs[stamp] = job; self.cancel_events[stamp] = threading.Event()
            self.update(job)
            thread = threading.Thread(target=self.run, args=(job,), daemon=True)
            self.threads[stamp] = thread; thread.start()
            return self.snapshot(job)

    def retry_names(self, job):
        return [name for name in job['names']
                if processing_state(read(Path(job['work'])/'cases'/name/'result.json'))['retry_required']]

    def retry(self, body):
        with self.lock:
            job = self.jobs[body['id']]
            if job['status'] in ACTIVE:
                raise ValueError('请等待当前批次结束后继续。')
            names = self.retry_names(job)
            selected = body.get('skus', names)
            if not selected or not isinstance(selected, list) or not set(selected) <= set(names):
                raise ValueError('没有需要继续的处理阶段；已完成候选仅需复核。')
            scan = self.scan(job['source'])
            return self.start(dict(scan_id=scan['id'], skus=selected, model=job['model'],
                concurrency=body.get('concurrency', 1),
                output=body.get('output') or job.get('output_base') or str(Path(job['output']).parent),
                resume_job_id=job['id'],
                source_body_reference_review=job.get('source_body_reference_review', False),
                side_candidate_review=job.get('side_candidate_review', False),
                require_geometry_resume=job.get('require_geometry_resume', False),
                saved_review_manifest=job.get('saved_review_manifest'),
                saved_review_manifest_sha256=job.get('saved_review_manifest_sha256')))

    def package(self, job, report):
        rows = verified_export_rows(report, job)
        partial = report.get('status') == 'partial_candidates_needs_visual_review'
        review = report.get('status') == 'review_candidates_needs_visual_review'
        root, output = Path(job['root']), Path(job['output'])
        note = None
        if partial or review:
            note = read(Path(job['work'])/'partial-candidate-manifest.json')
            if (not isinstance(note.get('items'), list)
                    or note.get('generated') != len(rows)
                    or {item.get('sku') for item in note['items']} != {item.get('sku') for item in report['attempts']}):
                raise ValueError('本批处理说明与候选不一致，未生成下载。')
            note = dict(note, exported_skus=[r['sku'] for r in rows],
                        exported_files={r['sku']+'/'+f+'.png':r['final_sha256'][f] for r in rows for f in FACES})
        temporary = root/'六面图片.zip.tmp'
        try:
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                for row in rows:
                    for face in FACES:
                        name = row['sku']+'/'+face+'.png'
                        archive.write(output/name, name)
                if note is not None:
                    archive.writestr('本批处理说明.json', json.dumps(note, ensure_ascii=False, indent=2))
            with zipfile.ZipFile(temporary) as archive:
                for row in rows:
                    for face in FACES:
                        if hashlib.sha256(archive.read(row['sku']+'/'+face+'.png')).hexdigest() != row['final_sha256'][face]:
                            raise ValueError('打包时图片发生变化，未生成下载。')
            create_overview(output, [row['sku'] for row in rows], root/'验收总览.jpg')
            temporary.replace(root/'六面图片.zip')
        finally:
            temporary.unlink(missing_ok=True)
        artifacts={kind:w.sha(root/name) for kind,name in [('zip','六面图片.zip'),('overview','验收总览.jpg')]}
        for row in rows:
            package=row.get('portable_review_package')
            if package is not None:
                path=output/row['sku']/'候选证据包.zip'
                expected=package.get('zip_sha256')
                if not isinstance(expected,str) or len(expected)!=64 or not path.is_file() or w.sha(path)!=expected:
                    raise ValueError('复核记录包缺失或校验不一致，未完成打包。')
                artifacts['bundle:'+row['sku']]=expected
        self.update(job, artifact_sha256=artifacts)
        return rows

    def run(self, job):
        proc = None; guard = None; started = time.monotonic(); event = self.cancel_events[job['id']]
        try:
            guard = r9_guard_context()
            initial = host_sample(); n = job['concurrency']
            if type(n) is not int or not 1 <= n <= 8:
                raise ValueError('并发数需为 1–8，未启动子进程')
            cmd = [sys.executable, '-m', 'phone_texture', 'auto', job['source'], '--work', job['work'],
                   '--output', job['output'], '--jobs', str(n), '--inference-slots', str(min(n, 2)), '--cpu-threads', '1',
                   '--device', 'auto', '--model', job['model'], '--protection-policy', job.get('protection_policy','none'), '--review-fallback', 'preserve-source', '--lighting-profile', job.get('lighting_profile','accepted-shadow-lift-v2'), '--bottom-alignment', job.get('bottom_alignment','appearance'), '--skus', *job['names']]
            if job['single']:
                cmd += ['--single-name', job['names'][0]]
            if n > 4:
                cmd += ['--experimental-concurrency']
            required_resume = job.get('require_geometry_resume', False)
            side_review = job.get('side_candidate_review', False)
            if type(required_resume) is not bool or type(side_review) is not bool:
                raise ValueError('保存的侧面复核或几何续接选项不是布尔值')
            resume_path = geometry_resume_path(job.get('resume_geometry_from'), required_resume)
            if resume_path:
                cmd += ['--resume-geometry-from', resume_path]
            if required_resume:
                cmd += ['--require-geometry-resume']
            if side_review:
                cmd += ['--side-candidate-review']
            reference_review = job.get('source_body_reference_review', False)
            if type(reference_review) is not bool:
                raise ValueError('保存的本机参考选项不是布尔值，未开始处理')
            if reference_review:
                if (job.get('protection_policy','none') != 'none'
                        or job.get('lighting_profile','accepted-shadow-lift-v2') != 'accepted-shadow-lift-v2'):
                    raise ValueError('本机参考复核只支持当前已验证调光配置，未开始处理')
                cmd += ['--source-body-reference-review']
            manifest_path, manifest_digest = saved_review_binding(
                job.get('saved_review_manifest'), job.get('saved_review_manifest_sha256'))
            if manifest_path is not None:
                if job.get('saved_review_manifest_sha256') is None:
                    raise ValueError('保存候选清单缺少批次校验值，未开始处理')
                cmd += ['--saved-review-manifest', manifest_path,
                        '--saved-review-manifest-sha256', manifest_digest]
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            with (Path(job['root'])/'运行日志.txt').open('w') as log:
                options = dict(cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if guard is not None:
                    options['pass_fds'] = (guard['lock_fd'],)
                self.update(job, worker_command=cmd, guard_run_id=guard['run_id'] if guard else None,
                            guard_paused_seconds=0., active_elapsed=0., worker_returncode=None)
                proc = subprocess.Popen(cmd, **options)
                if guard is not None:
                    self.update(job, **r9_worker_identity(proc))
                self.update(job, status='running', message='正在校验并续接保存候选；其余项目按原图处理' if manifest_path else '正在继续未完成步骤，校验并复用原几何结果' if job.get('resume_geometry_from') else '正在从原图处理六面')
                reason = None
                while proc.poll() is None:
                    if guard is not None:
                        r9_worker_members(proc)
                    paused = r9_paused_seconds(guard)
                    elapsed = time.monotonic()-started
                    sample = host_sample(); reason = guard_reason(sample, initial)
                    self.update(job, host=sample, elapsed=elapsed, guard_paused_seconds=paused, active_elapsed=max(0., elapsed-paused))
                    if event.is_set() or reason:
                        self.update(job, status='stopping', message=reason or '正在停止所有处理进程…')
                        stop_group(proc); break
                    # A stalled model must not keep the only queue forever.
                    if elapsed-paused > max(900, len(job['names'])*300):
                        reason = '本批长时间未结束，已停止。请查看日志后重试。'; stop_group(proc); break
                    event.wait(1)
            self.update(job, worker_returncode=proc.returncode)
            if reason or event.is_set():
                self.update(job, status='stopped' if reason else 'cancelled', message=reason or '已停止。已完成的单台结果保留，整批未完成。')
            elif proc.returncode != 0 and read(Path(job['work'])/'report.json').get('status') not in ('partial_candidates_needs_visual_review','review_candidates_needs_visual_review'):
                self.update(job, status='failed', message='本批未生成完整六面候选。可查看逐台原因，并重试未完成项目。')
            else:
                report = read(Path(job['work'])/'report.json')
                partial=report.get('status')=='partial_candidates_needs_visual_review'
                review=report.get('status')=='review_candidates_needs_visual_review'
                self.update(job, status='packaging', message='正在校验已生成六面候选并制作下载包')
                completed_rows = self.package(job, report)
                if review:self.update(job,status='review',message=f"已生成 {len(completed_rows)} / {len(job['names'])} 组六面候选；含待核对项，请查看逐组说明，尚未完成视觉及当前模型UV验收。")
                elif partial:self.update(job,status='partial',message=f"已生成 {len(completed_rows)} / {len(job['names'])} 组六面候选，可下载已生成部分；其余项目保留检查原因，所有结果仍待验收。")
                else:self.update(job, status='complete', message='六面图片已生成，等待你验收。实际模型 UV 接缝仍需在模型中检查。')
        except Exception as exc:
            if proc is not None and (guard is None or hasattr(proc, '_r9_owned')):
                stop_group(proc)
            # If guarded identity establishment failed, never fall back to an
            # unverified group signal. The external guard owns bounded cleanup.
            self.update(job, status='failed', message='处理未完成：'+str(exc),
                        guard_cleanup_required=guard is not None and proc is not None)
        finally:
            elapsed = time.monotonic()-started
            paused = job.get('guard_paused_seconds', 0.)
            self.update(job, ended=time.time(), elapsed=elapsed, active_elapsed=max(0., elapsed-paused),
                        worker_returncode=proc.returncode if proc is not None else None)

    def snapshot(self, job):
        with self.lock:
            result = dict(job)
        # Older batches keep their original on-disk layout and retry destination.
        result.setdefault('output_base', str(Path(job['root']).parent))
        rows = []; units = 0
        for name in result['names']:
            case = Path(result['work'])/'cases'/name
            done = read(case/'result.json'); live = read(case/'live-progress.json')
            complete = bool(done.get('six_faces_generated',done.get('six_faces_processed'))) and done.get('status')!='failed'
            processing = processing_state(done)
            # Effect warnings no longer hold a full-stage row back; only unfinished steps stay amber.
            retained=complete and (not processing['all_stages_complete'] or done.get('status')=='source_preserved_needs_review')
            status = 'review' if retained else 'complete' if complete else 'failed' if done.get('status') == 'failed' else 'running' if case.exists() and result['status'] in ACTIVE else 'waiting' if result['status'] in ACTIVE else 'unfinished'
            units += 15 if complete else live.get('completed_units', 0)
            bundle=done.get('portable_review_package') or {}
            bundle_path=Path(result['output'])/name/'候选证据包.zip'
            bundle_digest=result.get('artifact_sha256',{}).get('bundle:'+name)
            bundle_available=(complete and result['status'] in ('complete','partial','review')
                and isinstance(bundle_digest,str) and bundle_digest==bundle.get('zip_sha256')
                and bundle_path.is_file() and w.sha(bundle_path)==bundle_digest)
            rows.append(dict(name=name, status=status, stage=live.get('stage', 'loading'), face=live.get('face'),
                             seconds=done.get('seconds'), error=done.get('error'), error_hint=failure_hint(done.get('error')),
                             preview=complete, portable_review_available=bundle_available, **processing))
        result.update(rows=rows, completed=sum(r['status'] in ('complete','review') for r in rows),
                      review_count=sum(r['status']=='review' for r in rows),
                      program_checked_count=sum(r['status']=='complete' for r in rows),
                      retry_count=sum(r['retry_required'] for r in rows),
                      review_only_count=sum(r['review_only'] and r['status']=='review' for r in rows),
                      failed=sum(r['status']=='failed' for r in rows), total=len(rows),
                      progress=min(99, round(100*units/(15*len(rows)))) if result['status'] in ACTIVE else round(100*units/(15*len(rows))),
                      downloadable=result['status'] in ('complete','partial','review') and (Path(job['root'])/'六面图片.zip').is_file())
        # Display only: a review batch whose rows all passed reads as complete, old batches included.
        if result['status'] == 'review' and rows and all(r['status'] == 'complete' for r in rows):
            result.update(status='complete', message='六面图片已生成。')
        return result

    def state(self):
        with self.lock:
            jobs = list(self.jobs.values())
        return dict(default_output=str(self.default_output), jobs=[self.snapshot(j) for j in reversed(jobs)],
                    host=host_sample(), profile='iPhone 17 系列 · 铝金属 · 标准六面原图')

    def cancel(self, id):
        with self.lock:
            job = self.jobs[id]
            if job['status'] == 'packaging':
                raise ValueError('图片已处理完，正在保存结果，请稍候。')
            if job['status'] in ACTIVE:
                self.cancel_events[id].set(); self.update(job, status='stopping', message='正在停止所有处理进程…')
        return self.snapshot(job)

    def shutdown(self):
        for event in self.cancel_events.values():
            event.set()
        for thread in self.threads.values():
            thread.join(timeout=12)

    def artifact(self, query):
        id, kind = query.get('job'), query.get('kind')
        job = self.jobs[id]; root = Path(job['root'])
        if kind in ('zip', 'overview', 'log'):
            if kind != 'log' and job['status'] not in ('complete','partial','review'):
                raise ValueError('整批尚未完成，暂不能下载。')
            path = root/{'zip':'六面图片.zip', 'overview':'验收总览.jpg', 'log':'运行日志.txt'}[kind]
            if not path.is_file():
                raise ValueError('下载文件不存在，请查看批次记录。')
            if kind in job.get('artifact_sha256', {}) and w.sha(path) != job['artifact_sha256'][kind]:
                raise ValueError('下载文件校验不一致，请查看批次记录。')
            return path
        name, face = query.get('sku'), query.get('face')
        if kind == 'bundle':
            if name not in job['names'] or job['status'] not in ('complete','partial','review'):
                raise ValueError('该项目复核记录尚未完成')
            done=read(Path(job['work'])/'cases'/name/'result.json')
            expected=job.get('artifact_sha256',{}).get('bundle:'+name)
            package=done.get('portable_review_package') or {}
            path=Path(job['output'])/name/'候选证据包.zip'
            if (done.get('status')=='failed' or done.get('six_faces_generated') is not True
                    or not isinstance(expected,str) or expected!=package.get('zip_sha256')
                    or not path.is_file() or w.sha(path)!=expected):
                raise ValueError('复核记录包缺失或校验不一致。')
            return path
        if name not in job['names'] or face not in FACES:
            raise ValueError('无效图片')
        case = Path(job['work'])/'cases'/name
        done = read(case/'result.json')
        if kind == 'source':
            manifest = read(case/'work/intake.json')
            return Path(manifest['skus'][name][face]['snapshot'])
        if kind == 'image' and done.get('status')!='failed' and done.get('six_faces_generated',done.get('six_faces_processed')):
            path = case/'candidate'/name/(face+'.png')
            if not path.is_file() or w.sha(path) != done.get('final_sha256', {}).get(face):
                raise ValueError('候选图片缺失或校验不一致。')
            return path
        raise ValueError('该项目图片尚未完成')


def failure_hint(error):
    text=(error or '').lower()
    if text.startswith('ambiguous crossing side hardware contours'):
        return '同一控件存在多个候选材料边界，需核对；这不代表原图有缺陷。'
    if 'bottom paper material boundary unresolved' in text:
        return '纸尾与金属的材料边界待核对；该面保留几何结果。'
    if any(word in text for word in ('antenna','endpoint','paired source','normalization')):
        return '天线位置未能可靠确定，请查看原图与检测记录。'
    if any(word in text for word in ('top_','corner','plate','body rail','geometry')):
        return '机身轮廓或摆正需要核对，请查看对应面的检测记录。'
    if any(word in text for word in ('illumination','lighting','perimeter','correction','seam')):
        return '校光或接缝检查需要核对，原图和过程结果已保留。'
    if 'qr' in text:return '标识核验结果不一致，请查看原图与候选。'
    return error


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed(self, mutate=False):
            host = self.headers.get('Host')
            valid = f'127.0.0.1:{self.server.server_port}'
            if host != valid:
                raise PermissionError('仅支持本机地址访问。')
            origin = self.headers.get('Origin')
            if origin and origin != 'http://'+valid:
                raise PermissionError('不允许跨站访问。')
            if mutate and self.headers.get('X-App-Token') != app.token:
                raise PermissionError('页面已失效，请刷新后重试。')

        def respond(self, data, status=200, mime='application/json; charset=utf-8', filename=None):
            body = json.dumps(data, ensure_ascii=False).encode() if mime.startswith('application/json') else data
            self.send_response(status)
            self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if filename:
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''"+quote(filename))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            try:
                self.allowed()
                url = urlparse(self.path); q = {k:v[0] for k,v in parse_qs(url.query).items()}
                if url.path == '/':
                    return self.respond((ASSETS/'index.html').read_bytes().replace(b'__TOKEN__', app.token.encode()), mime='text/html; charset=utf-8')
                if url.path in ('/app.js', '/style.css'):
                    p = ASSETS/url.path[1:]
                    return self.respond(p.read_bytes(), mime='text/javascript; charset=utf-8' if p.suffix=='.js' else 'text/css; charset=utf-8')
                if url.path == '/api/state':
                    return self.respond(app.state())
                if url.path == '/api/health':
                    return self.respond(dict(app='phone-six-view-browser', version=__app_version__, home=str(app.home)))
                if url.path == '/api/scan-image':
                    scan = app.scans[q['scan']]
                    row = next((r for r in scan['rows'] if r['name']==q['sku']), None)
                    if row is None:
                        raise ValueError('无效图片')
                    item = next((r for r in row['recognition']['images'] if r['name']==q['name']), None)
                    if item is None:
                        raise ValueError('无效图片')
                    with Image.open(item['file']) as im:
                        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((320,320))
                        buf=io.BytesIO();im.save(buf,format='JPEG',quality=85)
                    return self.respond(buf.getvalue(),mime='image/jpeg')
                if url.path == '/api/artifact':
                    p = app.artifact(q)
                    if q.get('thumb') == '1' and p.suffix.lower() in EXTENSIONS:
                        with Image.open(p) as im:
                            im = ImageOps.exif_transpose(im).convert('RGBA'); im.thumbnail((500, 500))
                            buf = io.BytesIO(); im.save(buf, format='PNG')
                        return self.respond(buf.getvalue(), mime='image/png')
                    self.send_response(200)
                    self.send_header('Content-Type', mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
                    self.send_header('Content-Length', str(p.stat().st_size))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    if q.get('download') == '1':
                        self.send_header('Content-Disposition', "attachment; filename*=UTF-8''"+quote(p.name))
                    self.end_headers()
                    with p.open('rb') as f:
                        shutil.copyfileobj(f, self.wfile, length=256*1024)
                    return
                self.respond(dict(error='未找到'), 404)
            except PermissionError as exc:
                self.respond(dict(error=str(exc)), 403)
            except (ValueError, KeyError, OSError) as exc:
                self.respond(dict(error=str(exc)), 400)

        def do_POST(self):
            try:
                self.allowed(mutate=True)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 128*1024:
                    raise ValueError('请求内容无效')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('请求格式无效')
                if self.path == '/api/scan':
                    result = app.scan(body['path'])
                elif self.path == '/api/name':
                    result = app.apply_names(body['scan_id'])
                elif self.path == '/api/start':
                    result = app.start(body)
                elif self.path == '/api/retry':
                    result = app.retry(body)
                elif self.path == '/api/cancel':
                    result = app.cancel(body['id'])
                elif self.path == '/api/pick-folder':
                    result = pick_system_folder(body.get('kind'), body.get('path'))
                elif self.path == '/api/folders':
                    result = list_folders(body.get('path'))
                elif self.path == '/api/shutdown':
                    if any(j['status'] in ACTIVE for j in app.jobs.values()):
                        raise ValueError('请先停止当前批次，再退出服务。')
                    self.respond(dict(ok=True))
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                elif self.path == '/api/open':
                    job = app.jobs[body['id']]
                    path = Path(job['output'] if job['status'] in ('complete','partial','review') else job['root'])
                    subprocess.Popen(['open', str(path)])
                    result = dict(ok=True)
                else:
                    return self.respond(dict(error='未找到'), 404)
                self.respond(result)
            except PermissionError as exc:
                self.respond(dict(error=str(exc)), 403)
            except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
                self.respond(dict(error=str(exc)), 400)
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description='六面图本地浏览器工具')
    p.add_argument('--home', type=Path, default=REPO.parent/'outputs/六面图浏览器')
    p.add_argument('--port', type=int, default=18765)
    p.add_argument('--no-open', action='store_true')
    p.add_argument('--source-body-reference-review', action='store_true')
    p.add_argument('--side-candidate-review', action='store_true')
    p.add_argument('--resume-geometry-from', type=Path)
    p.add_argument('--require-geometry-resume', action='store_true')
    p.add_argument('--saved-review-manifest', type=Path)
    p.add_argument('--saved-review-manifest-sha256')
    args = p.parse_args(argv)
    # A home-level lock also prevents multiple queues on different ports.
    home = args.home.expanduser().resolve(); home.mkdir(parents=True, exist_ok=True)
    lockfile = (home/'.service.lock').open('a')
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('此目录的服务已启动，请使用启动程序重新打开网页。')
    app = App(args.home, saved_review_manifest=args.saved_review_manifest,
              saved_review_manifest_sha256=args.saved_review_manifest_sha256,
              source_body_reference_review=args.source_body_reference_review,
              side_candidate_review=args.side_candidate_review,
              resume_geometry_from=args.resume_geometry_from,
              require_geometry_resume=args.require_geometry_resume)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(app))
    url = f'http://127.0.0.1:{server.server_port}'
    save(home/'.service.json', dict(url=url, pid=os.getpid()))
    print('六面图处理工具：'+url, flush=True)
    if not args.no_open:
        webbrowser.open(url)
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown(); server.server_close(); lockfile.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
