import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from phone_texture.desktop import input_folder


def test_desktop_launcher_dispatches_full_workflow_and_preserves_dropped_path(tmp_path):
    root=Path(__file__).resolve().parents[1]
    command=tmp_path/'手机图片一键处理.command';shutil.copy2(root/command.name,command)
    python=tmp_path/'.venv/bin/python';python.parent.mkdir(parents=True)
    log=tmp_path/'calls.json'
    python.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\n'
                      +'if sys.argv[1] != "-c": Path('+repr(str(log))+').write_text(json.dumps(sys.argv[1:]))\n')
    python.chmod(0o755)
    dropped=tmp_path/'输入 图片';dropped.mkdir()
    p=subprocess.run(['bash',str(command),str(dropped)],env={**os.environ,'PHONE_NORMALIZER_NO_PAUSE':'1'},capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(log.read_text())==['-m','phone_texture','gui',str(dropped)]


def test_dropped_six_faces_resolve_to_folder_without_guessing_across_batches(tmp_path):
    a=tmp_path/'A';a.mkdir();b=tmp_path/'B';b.mkdir()
    for folder in (a,b):
        for face in ('front','back'):(folder/(face+'.png')).touch()
    assert input_folder([a/'front.png',a/'back.png'])==str(a)
    with pytest.raises(ValueError):input_folder([a/'front.png',b/'back.png'])
