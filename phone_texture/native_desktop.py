"""macOS desktop fallback without Tk; all processing stays in workflow.run."""
from datetime import datetime
from pathlib import Path
import subprocess

from . import workflow
from .desktop import input_folder, MATERIALS, INPUTS, BASES, ATTACHMENTS, open_path


def apple(script, *args):
    result = subprocess.run(['osascript', '-e', 'on run argv\n' + script + '\nend run', *map(str, args)],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return result.stdout.strip()


def select(title, choices):
    return apple('set picked to choose from list (items 2 thru -1 of argv) with prompt (item 1 of argv)\n'
                 'if picked is false then error number -128\nreturn item 1 of picked', title, *choices)


def folder(title):
    return apple('return POSIX path of (choose folder with prompt (item 1 of argv))', title)


def main(paths=()):
    source = input_folder(paths)
    action = select('手机六面图处理', ['新建六面处理批次', '继续已有批次'])
    if action == '继续已有批次':
        work = Path(folder('选择批次过程目录')).resolve()
    else:
        source = source or folder('选择六面图片目录')
        material = MATERIALS[select('选择材质', list(MATERIALS))]
        kind = INPUTS[select('选择输入状态', list(INPUTS))]
        basis = BASES[select('选择天线基准', list(BASES))]
        model = None
        if select('选择模型坐标要求', ['按本机背面对齐图片', '关联固定 UV 模型参考']) == '关联固定 UV 模型参考':
            model = apple('return POSIX path of (choose file with prompt "选择固定模型坐标清单" of type {"public.json"})')
        base = Path(folder('选择保存过程和成品的父目录')) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        work = base / '过程'
        workflow.create(source, work, base / '成品', material=material, input_kind=kind,
                        alignment_basis=basis, model_reference=model)
    while True:
        value = workflow.run(work)
        print(value['stage_label'] + '：' + value['message'], flush=True)
        if value['status'] == 'completed':
            open_path(value['output'])
            return 0
        action = select(value['stage_label'] + '：' + value['message'], ['保存进度并退出', '查看检查资料', '补充本步资料并继续'])
        if action == '保存进度并退出':
            return 0
        if action == '查看检查资料':
            open_path(work)
            continue
        group, key, title = ATTACHMENTS[value['stage']]
        path = apple('return POSIX path of (choose file with prompt (item 1 of argv) of type {"public.json"})', title)
        config = workflow.w.read(work / 'workflow.json')
        (config[group] if group else config)[key] = path
        workflow.save(work / 'workflow.json', config)
