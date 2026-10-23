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


def stop_group(proc):
    # The leader may have exited while its workers still exist. Always address
    # the entire original group, including on cancellation and server shutdown.
    import psutil
    members = []
    for p in psutil.process_iter(['pid']):
        try:
            if os.getpgid(p.pid) == proc.pid and p.status() != psutil.STATUS_ZOMBIE:
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


class App:
    def __init__(self, home):
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
                # No automatic resume: fresh analysis is required on retry.
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
            if type(jobs) is not int or not 1 <= jobs <= 8:
                raise ValueError('并发数需为 1–8。10 并发未完成验证，暂不开放。')
            valid = {r['name'] for r in scan['rows'] if r['valid']}
            if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names) or len(names) != len(set(names)) or not set(names) <= valid:
                raise ValueError('请选择至少一个检查通过的 SKU。')
            if len(names) > 100:
                raise ValueError('预览版每批最多处理 100 台，请分批勾选。')
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
                       names=names, model=model, concurrency=min(jobs, len(names)), requested_concurrency=jobs,
                       root=str(root), output=str(output_base/stamp), output_base=str(output_base), work=str(root/'过程'),
                       message='准备处理', elapsed=0, host=sample)
            self.jobs[stamp] = job; self.cancel_events[stamp] = threading.Event()
            self.update(job)
            thread = threading.Thread(target=self.run, args=(job,), daemon=True)
            self.threads[stamp] = thread; thread.start()
            return self.snapshot(job)

    def run(self, job):
        proc = None; started = time.monotonic(); event = self.cancel_events[job['id']]
        try:
            initial = host_sample(); n = job['concurrency']
            cmd = [sys.executable, '-m', 'phone_texture', 'auto', job['source'], '--work', job['work'],
                   '--output', job['output'], '--jobs', str(n), '--inference-slots', str(min(n, 2)),
                   '--device', 'auto', '--model', job['model'], '--skus', *job['names']]
            if job['single']:
                cmd += ['--single-name', job['names'][0]]
            if n > 4:
                cmd += ['--experimental-concurrency']
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            with (Path(job['root'])/'运行日志.txt').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.update(job, status='running', message='正在从原图处理六面')
                reason = None
                while proc.poll() is None:
                    sample = host_sample(); reason = guard_reason(sample, initial)
                    self.update(job, host=sample, elapsed=time.monotonic()-started)
                    if event.is_set() or reason:
                        self.update(job, status='stopping', message=reason or '正在停止所有处理进程…')
                        stop_group(proc); break
                    # A stalled model must not keep the only queue forever.
                    if time.monotonic()-started > max(900, len(job['names'])*300):
                        reason = '本批长时间未结束，已停止。请查看日志后重试。'; stop_group(proc); break
                    event.wait(1)
            if reason or event.is_set():
                self.update(job, status='stopped' if reason else 'cancelled', message=reason or '已停止。已完成的单台结果保留，整批未完成。')
            elif proc.returncode != 0:
                self.update(job, status='failed', message='部分项目处理失败。可查看逐台原因，并重新处理未完成项目。')
            else:
                report = read(Path(job['work'])/'report.json')
                if report.get('status') != 'processed_needs_visual_review' or report.get('processed') != len(job['names']):
                    raise ValueError('完整性检查未通过，未生成整批下载。')
                self.update(job, status='packaging', message='六面处理已完成，正在生成验收总览与下载包')
                output = Path(job['output'])
                files = []
                for row in report['attempts']:
                    for face in FACES:
                        p = output/row['sku']/(face+'.png')
                        if w.sha(p) != row['final_sha256'][face]:
                            raise ValueError('结果校验不一致，请查看过程记录。')
                        files.append(p)
                # Exports contain only this completed batch, never stale files.
                with zipfile.ZipFile(Path(job['root'])/'六面图片.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                    for p in files:
                        z.write(p, str(p.relative_to(output)))
                create_overview(output, job['names'], Path(job['root'])/'验收总览.jpg')
                self.update(job, status='complete', message='六面图片已生成，等待你验收。实际模型 UV 接缝仍需在模型中检查。')
        except Exception as exc:
            if proc is not None:
                stop_group(proc)
            self.update(job, status='failed', message='处理未完成：'+str(exc))
        finally:
            self.update(job, ended=time.time(), elapsed=time.monotonic()-started)

    def snapshot(self, job):
        with self.lock:
            result = dict(job)
        # Older batches keep their original on-disk layout and retry destination.
        result.setdefault('output_base', str(Path(job['root']).parent))
        rows = []; units = 0
        for name in result['names']:
            case = Path(result['work'])/'cases'/name
            done = read(case/'result.json'); live = read(case/'live-progress.json')
            complete = bool(done.get('six_faces_processed'))
            status = 'complete' if complete else 'failed' if done.get('status') == 'failed' else 'running' if case.exists() and result['status'] in ACTIVE else 'waiting' if result['status'] in ACTIVE else 'unfinished'
            units += 15 if complete else live.get('completed_units', 0)
            rows.append(dict(name=name, status=status, stage=live.get('stage', 'loading'), face=live.get('face'),
                             seconds=done.get('seconds'), error=done.get('error'), preview=complete))
        result.update(rows=rows, completed=sum(r['status']=='complete' for r in rows),
                      failed=sum(r['status']=='failed' for r in rows), total=len(rows),
                      progress=min(99, round(100*units/(15*len(rows)))) if result['status'] in ACTIVE else round(100*units/(15*len(rows))),
                      downloadable=result['status']=='complete')
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
            if kind != 'log' and job['status'] != 'complete':
                raise ValueError('整批尚未完成，暂不能下载。')
            return root/{'zip':'六面图片.zip', 'overview':'验收总览.jpg', 'log':'运行日志.txt'}[kind]
        name, face = query.get('sku'), query.get('face')
        if name not in job['names'] or face not in FACES:
            raise ValueError('无效图片')
        case = Path(job['work'])/'cases'/name
        done = read(case/'result.json')
        if kind == 'source':
            manifest = read(case/'work/intake.json')
            return Path(manifest['skus'][name][face]['snapshot'])
        if kind == 'image' and done.get('six_faces_processed'):
            return case/'candidate'/name/(face+'.png')
        raise ValueError('该项目图片尚未完成')


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
                    path = Path(job['output'] if job['status']=='complete' else job['root'])
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
    args = p.parse_args(argv)
    # A home-level lock also prevents multiple queues on different ports.
    home = args.home.expanduser().resolve(); home.mkdir(parents=True, exist_ok=True)
    lockfile = (home/'.service.lock').open('a')
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('此目录的服务已启动，请使用启动程序重新打开网页。')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), BaseHTTPRequestHandler)
    app = App(args.home); server.RequestHandlerClass = handler(app)
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
