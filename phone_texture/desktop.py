"""Desktop shell for the same resumable six-face workflow used by the CLI."""
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

from . import workflow

MATERIALS = {'铝金属': 'aluminum', '不锈钢': 'stainless', '混合材料': 'mixed'}
INPUTS = {'原始六面照片': 'raw', '已抠图，需要摆正': 'cutout', '已完成几何，等待复核': 'existing'}
BASES = {'本机背面（默认）': 'back_image', '正背面两端分别约束': 'front_back_images'}
ATTACHMENTS = {
    'geometry': ('geometry', 'guides', '几何标定资料'),
    'geometry_review': ('geometry', 'review', '几何复核记录'),
    'lighting_scope': ('lighting', 'scope', '六面光照范围'),
    'lighting': ('lighting', 'jobs', '逐面光照任务及复核记录'),
    'structure': ('structure', 'spec', '本机背面与侧面结构标定'),
    'structure_review': ('structure', 'review', '结构复核记录'),
    'model_reference': (None, 'model_review', '固定模型坐标复核记录'),
}


def input_folder(paths):
    paths = [Path(p).expanduser().resolve() for p in paths]
    if not paths:
        return ''
    if len(paths) == 1 and paths[0].is_dir():
        return str(paths[0])
    parents = {p.parent for p in paths}
    if len(parents) == 1 and all(p.is_file() for p in paths):
        return str(next(iter(parents)))
    raise ValueError('请拖入一个六面图目录，或同一目录中的六面图片')


def open_path(path):
    path = str(Path(path).resolve())
    subprocess.Popen(['open', path] if sys.platform == 'darwin' else ['xdg-open', path])


class Window:
    def __init__(self, root, paths=()):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        self.root, self.tk, self.dialog, self.messages = root, tk, filedialog, messagebox
        self.worker = None
        self.events = queue.Queue()
        self.work = None
        self.last_progress = None
        root.title('手机六面图处理')
        root.geometry('900x690')
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='手机六面图处理', font=('', 22)).pack(anchor='w')
        ttk.Label(frame, text='几何 → 材质光照 → 本机背面对齐 → 复核发布').pack(anchor='w', pady=(5, 16))
        self.source = tk.StringVar(value=input_folder(paths))
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        base = Path.home() / 'Documents' / '手机六面处理' / stamp
        self.process = tk.StringVar(value=str(base / '过程'))
        self.output = tk.StringVar(value=str(base / '成品'))
        self.model = tk.StringVar()
        for label, var, file in [('六面图片目录', self.source, False), ('过程目录', self.process, False),
                                 ('成品目录', self.output, False), ('固定模型参考（可选）', self.model, True)]:
            row = ttk.Frame(frame); row.pack(fill='x', pady=4)
            ttk.Label(row, text=label, width=22).pack(side='left')
            ttk.Entry(row, textvariable=var).pack(side='left', fill='x', expand=True)
            ttk.Button(row, text='选择', command=lambda v=var, f=file: self.choose(v, f)).pack(side='left', padx=(8, 0))
        row = ttk.Frame(frame); row.pack(fill='x', pady=10)
        self.material = tk.StringVar(value='铝金属'); self.kind = tk.StringVar(value='原始六面照片')
        ttk.Label(row, text='材质').pack(side='left')
        ttk.Combobox(row, textvariable=self.material, values=list(MATERIALS), state='readonly', width=12).pack(side='left', padx=8)
        ttk.Label(row, text='输入状态').pack(side='left', padx=(16, 0))
        ttk.Combobox(row, textvariable=self.kind, values=list(INPUTS), state='readonly', width=25).pack(side='left', padx=8)
        row = ttk.Frame(frame); row.pack(fill='x', pady=4)
        self.basis = tk.StringVar(value='本机背面（默认）')
        ttk.Label(row, text='天线基准').pack(side='left')
        ttk.Combobox(row, textvariable=self.basis, values=list(BASES), state='readonly', width=26).pack(side='left', padx=8)
        ttk.Label(frame, text='默认：每台使用自己的背面作为天线基准，按钮独立标定。').pack(anchor='w')
        row = ttk.Frame(frame); row.pack(fill='x', pady=14)
        self.buttons = []
        for label, action in [('新建批次', self.create), ('打开批次', self.load), ('继续处理', self.run),
                               ('补充本步资料', self.attach), ('查看检查资料', self.inspect), ('打开成品', self.delivery)]:
            button = ttk.Button(row, text=label, command=action); button.pack(side='left', padx=(0, 7)); self.buttons.append(button)
        self.progress = tk.StringVar(value='请选择六面图片目录，新建批次。')
        ttk.Label(frame, textvariable=self.progress, wraplength=830).pack(anchor='w', pady=8)
        self.log = tk.Text(frame, height=15, state='disabled', wrap='word')
        self.log.pack(fill='both', expand=True)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(150, self.poll)

    def choose(self, var, file=False):
        path = self.dialog.askopenfilename(filetypes=[('复核或标定资料', '*.json')]) if file else self.dialog.askdirectory(mustexist=False)
        if path:
            if var is self.process or var is self.output:
                label = '过程' if var is self.process else '成品'
                path = str(Path(path) / (label + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
            var.set(path)

    def create(self):
        try:
            if not self.source.get().strip():
                raise ValueError('请选择六面图片目录')
            workflow.create(self.source.get(), self.process.get(), self.output.get(),
                material=MATERIALS[self.material.get()], input_kind=INPUTS[self.kind.get()],
                alignment_basis=BASES[self.basis.get()],
                model_reference=self.model.get() or None)
            self.work = Path(self.process.get()).resolve()
            self.refresh()
        except Exception as exc:
            self.messages.showerror('无法新建批次', str(exc))

    def load(self):
        path = self.dialog.askdirectory(title='选择已有批次的过程目录')
        if path:
            try:
                workflow.status(path)
                self.work = Path(path).resolve()
                self.process.set(str(self.work))
                config = workflow.w.read(self.work / 'workflow.json')
                self.output.set(config['output'])
                self.source.set(workflow.w.read(self.work / 'intake.json')['source_root'])
                self.kind.set(next(k for k, v in INPUTS.items() if v == config['input_kind']))
                self.material.set(next(k for k, v in MATERIALS.items() if v == config['material']))
                self.basis.set(next(k for k, v in BASES.items() if v == config['alignment_basis']))
                self.model.set((config.get('model_reference') or {}).get('path', ''))
                self.refresh()
            except Exception as exc:
                self.messages.showerror('无法打开批次', str(exc))

    def run(self):
        if not self.work:
            return self.messages.showinfo('尚未选择批次', '请先新建或打开批次。')
        for button in self.buttons:
            button.configure(state='disabled')
        self.progress.set('处理中，进度会保存在过程目录…')
        self.worker = threading.Thread(target=self.execute, daemon=False)
        self.worker.start()

    def execute(self):
        try:
            process = subprocess.Popen([sys.executable, '-m', 'phone_texture', 'run', str(self.work)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with (self.work / 'desktop-run.log').open('a', encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line)
            process.wait()
        except Exception as exc:
            self.events.put(str(exc) + '\n')
        finally:
            self.events.put(None)

    def poll(self):
        try:
            while True:
                line = self.events.get_nowait()
                if line is None:
                    for button in self.buttons:
                        button.configure(state='normal')
                    self.refresh()
                else:
                    self.log.configure(state='normal'); self.log.insert('end', line); self.log.see('end'); self.log.configure(state='disabled')
        except queue.Empty:
            pass
        if self.worker and self.worker.is_alive():
            self.refresh()
        self.root.after(150, self.poll)

    def refresh(self):
        if self.work:
            value = workflow.status(self.work)
            message = value['stage_label'] + '：' + value['message']
            self.progress.set(message)
            if message != self.last_progress:
                self.last_progress = message
                self.log.configure(state='normal'); self.log.insert('end', message + '\n'); self.log.see('end'); self.log.configure(state='disabled')

    def attach(self):
        if not self.work:
            return
        state = workflow.status(self.work)
        if state['status'] == 'completed':
            return self.messages.showinfo('本批已完成', '需要调整处理资料时，请新建批次版本。')
        group, key, title = ATTACHMENTS.get(state['stage'], ATTACHMENTS['geometry'])
        path = self.dialog.askopenfilename(title='选择' + title, filetypes=[('标定或复核资料', '*.json')])
        if path:
            config = workflow.w.read(self.work / 'workflow.json')
            target = config[group] if group else config
            target[key] = str(Path(path).resolve())
            workflow.save(self.work / 'workflow.json', config)
            self.progress.set('已登记' + title + '，点击继续处理。')

    def inspect(self):
        if self.work:
            open_path(self.work)

    def delivery(self):
        if self.work:
            value = workflow.status(self.work)
            if value['status'] == 'completed':
                open_path(value['output'])
            else:
                self.messages.showinfo('尚未发布', value['message'])

    def close(self):
        if self.worker and self.worker.is_alive():
            self.messages.showinfo('处理中', '当前处理仍在运行，进度会保存。请等待本步结束后关闭窗口。')
            return
        self.root.destroy()


def main(paths=()):
    try:
        import tkinter as tk
    except ImportError:
        # A native macOS folder picker still feeds the same workflow, never the
        # historical batch normalizer. Further steps use its saved configuration.
        if sys.platform != 'darwin':
            raise RuntimeError('此 Python 未安装窗口组件；请使用 phone-texture start/run')
        from .native_desktop import main as native
        return native(paths)
    root = tk.Tk()
    Window(root, paths)
    root.mainloop()
    return 0
