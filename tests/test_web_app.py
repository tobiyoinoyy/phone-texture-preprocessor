"""Browser-boundary tests: user input, isolation, completion and cancellation."""
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile

from PIL import Image
import pytest
from phone_texture import web_app as web
from phone_texture import view_recognition as recognition


def six(folder):
    folder.mkdir(parents=True)
    for i,face in enumerate(web.FACES):
        Image.new('RGB',(32,64),(30+i*30,30+i*30,30+i*30)).save(folder/(face+'.JPG'))
    return folder


@pytest.fixture
def identified(monkeypatch):
    """Isolate web orchestration from classifier accuracy (separate regression).

    Pixel-coded faces let web tests swap/rename files without using filenames
    as classifier input. The real inspect_images hash/duplicate logic runs.
    """
    def recognize_pixels(path):
        with Image.open(path) as im:value=im.convert('RGB').getpixel((0,0))[0]
        index=round((value-30)/30)
        return dict(face=web.FACES[index],reason=None,profile='test-pixels')
    monkeypatch.setattr(recognition,'recognize',recognize_pixels)


def test_scan_single_batch_missing_duplicate_and_corrupt(tmp_path,identified):
    source=tmp_path/'原图';six(source/'SKU_01');six(source/'SKU_02')
    (source/'SKU_02/top.JPG').unlink()
    rows=web.scan_folder(source)
    assert rows['valid']==1 and rows['total']==2 and not rows['single']
    assert '顶面' in '；'.join(rows['rows'][1]['errors'])
    assert web.scan_folder(source/'SKU_01')['single']
    Image.new('RGB',(1,1)).save(source/'SKU_01/front.png')
    assert '重复' in '；'.join(web.scan_folder(source/'SKU_01')['rows'][0]['errors'])
    (source/'SKU_01/front.png').unlink();(source/'SKU_01/back.JPG').write_bytes(b'bad')
    assert web.scan_folder(source/'SKU_01')['valid']==0


def test_start_rejects_invalid_concurrency_selection_and_overlapping_paths(tmp_path,identified):
    app=web.App(tmp_path/'app');source=six(tmp_path/'raw');scan=app.scan(source)
    body=dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=5)
    for n in [0,9,10,True,'5']:
        with pytest.raises(ValueError,match='1–8'):app.start(dict(body,concurrency=n))
    for names in [[],['raw','raw'],['../raw'],['other']]:
        with pytest.raises(ValueError,match='SKU'):app.start(dict(body,skus=names))
    for output in [source,source/'out',tmp_path]:
        with pytest.raises(ValueError,match='互相包含'):app.start(dict(body,output=str(output)))
    (source/'top.JPG').unlink()
    with pytest.raises(ValueError,match='Missing'):app.start(body)
    assert not app.jobs


def test_severe_memory_and_swap_guards():
    base=dict(pressure=1,swap=0,available=10*1024**3)
    assert web.guard_reason(base,base) is None
    assert '压力' in web.guard_reason(dict(base,pressure=4),base)
    assert '交换' in web.guard_reason(dict(base,swap=9*1024**3),base)
    assert '不足' in web.guard_reason(dict(base,available=100),base)


def test_restore_interrupted_and_never_expose_incomplete_outputs(tmp_path):
    app=web.App(tmp_path);job=dict(id='old',status='running',root=str(tmp_path/'run'),work=str(tmp_path/'work'),names=['SKU_01'])
    app.jobs['old']=job;app.update(job)
    restored=web.App(tmp_path)
    assert restored.jobs['old']['status']=='interrupted'
    for kind in ['zip','overview','image']:
        with pytest.raises(ValueError):restored.artifact(dict(job='old',kind=kind,sku='SKU_01',face='front'))
    with pytest.raises(ValueError):restored.artifact(dict(job='old',kind='source',sku='../secret',face='front'))


def test_http_rejects_cross_origin_and_untrusted_host(tmp_path):
    app=web.App(tmp_path)
    server=web.ThreadingHTTPServer(('127.0.0.1',0),web.handler(app))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        def request(method,path,headers=None,body=None):
            c=http.client.HTTPConnection('127.0.0.1',server.server_port)
            c.request(method,path,body=body,headers=headers or {});r=c.getresponse();data=r.read();c.close();return r.status,data
        assert request('GET','/')[0]==200
        assert request('GET','/api/state',{'Host':'malicious.example'})[0]==403
        assert request('POST','/api/scan',{'Origin':'https://evil.test','X-App-Token':app.token},'{}')[0]==403
        assert request('POST','/api/scan',{},'{}')[0]==403
        assert request('GET','/../../AGENTS.md')[0]==404
        assert request('POST','/api/scan',{'X-App-Token':app.token},'[]')[0]==400
    finally:server.shutdown();server.server_close();thread.join()


def test_stop_group_reaps_worker_not_just_leader(tmp_path):
    pidfile=tmp_path/'child'
    code='import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)'
    p=subprocess.Popen([sys.executable,'-c',code,str(pidfile)],start_new_session=True)
    try:
        for _ in range(100):
            if pidfile.exists():break
            time.sleep(.02)
        assert pidfile.exists()
        child=int(pidfile.read_text());web.stop_group(p)
        import psutil
        assert not psutil.pid_exists(child) or psutil.Process(child).status()==psutil.STATUS_ZOMBIE
        assert p.poll() is not None
    finally:
        web.stop_group(p)


def test_one_active_batch_and_command_uses_original_pipeline(tmp_path,monkeypatch,identified):
    app=web.App(tmp_path/'app');source=six(tmp_path/'raw');scan=app.scan(source)
    monkeypatch.setattr(threading.Thread,'start',lambda self:None)
    monkeypatch.setattr(web,'host_sample',lambda:dict(pressure=1,available=10*1024**3,swap=0))
    job=app.start(dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=5))
    assert job['concurrency']==1 and job['requested_concurrency']==5
    assert job['status']=='starting'
    with pytest.raises(ValueError,match='已有批次'):app.start(dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=1))
    app.cancel(job['id']);assert app.cancel_events[job['id']].is_set()


def test_completed_batch_exports_only_six_checked_faces(tmp_path,monkeypatch):
    app=web.App(tmp_path/'app');raw=six(tmp_path/'raw');root=tmp_path/'run';root.mkdir()
    work=root/'过程';work.mkdir();out=root/'六面图片';six(out/'raw')
    # Real export validation requires PNG output hashes from the worker report.
    hashes={}
    for face in web.FACES:
        Image.new('RGBA',(32,64),'orange').save(out/'raw'/(face+'.png'))
        (out/'raw'/(face+'.JPG')).unlink()
        hashes[face]=web.w.sha(out/'raw'/(face+'.png'))
    web.save(work/'report.json',dict(status='processed_needs_visual_review',processed=1,attempts=[dict(sku='raw',final_sha256=hashes)]))
    job=dict(id='good',model='iphone17pro',names=['raw'],source=str(raw),single=True,root=str(root),work=str(work),output=str(out),concurrency=1,status='starting')
    app.jobs['good']=job;app.cancel_events['good']=threading.Event()
    captured=[]
    class Process:
        returncode=0
        def __init__(self,cmd,**kwargs):captured.append(cmd)
        def poll(self):return 0
    monkeypatch.setattr(web.subprocess,'Popen',Process)
    monkeypatch.setattr(web,'host_sample',lambda:dict(pressure=1,swap=0,available=10*1024**3))
    app.run(job)
    assert job['status']=='complete'
    assert captured[0][:4]==[sys.executable,'-m','phone_texture','auto']
    assert str(raw) in captured[0] and '--single-name' in captured[0]
    with zipfile.ZipFile(root/'六面图片.zip') as z:
        assert sorted(z.namelist())==sorted('raw/'+f+'.png' for f in web.FACES)
    assert (root/'验收总览.jpg').exists()


def test_browser_folder_picker_lists_only_visible_directories(tmp_path):
    (tmp_path/'一组原图').mkdir();(tmp_path/'输出').mkdir();(tmp_path/'.hidden').mkdir()
    (tmp_path/'photo.jpg').write_bytes(b'not a directory')
    result=web.list_folders(str(tmp_path))
    assert {r['name'] for r in result['directories']}=={'一组原图','输出'}
    assert result['parent']==str(tmp_path.parent)
    with pytest.raises(ValueError):web.list_folders(str(tmp_path/'photo.jpg'))


def test_unnamed_and_swapped_images_create_verified_copies(tmp_path,identified):
    app=web.App(tmp_path/'app');raw=six(tmp_path/'raw')
    original={f:(raw/(f+'.JPG')).read_bytes() for f in web.FACES}
    for i,f in enumerate(web.FACES):(raw/(f+'.JPG')).rename(raw/f'IMG_{i}.jpg')
    scan=app.scan(raw)
    assert scan['renameable']==1 and scan['valid']==0
    with pytest.raises(ValueError,match='SKU'):app.start(dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=1))
    result=app.apply_names(scan['id'])
    assert result['valid']==1 and result['renameable']==0
    assert app.apply_names(scan['id'])['id']==result['id']
    for i,f in enumerate(web.FACES):
        assert (raw/f'IMG_{i}.jpg').read_bytes()==original[f]
        assert (Path(result['source'])/'raw'/(f+'.jpg')).read_bytes()==original[f]
    named=Path(result['source'])/'raw'
    (named/'front.jpg').write_bytes(original['back']);(named/'back.jpg').write_bytes(original['front'])
    swapped=app.scan(named)
    assert swapped['valid']==0 and swapped['renameable']==1
    assert len(swapped['rows'][0]['changes'])==2


def test_naming_rejects_source_changes_and_does_not_fill_duplicate_faces(tmp_path,identified):
    app=web.App(tmp_path/'app');raw=six(tmp_path/'raw');front=(raw/'front.JPG').read_bytes()
    (raw/'front.JPG').rename(raw/'random.jpg');scan=app.scan(raw)
    (raw/'random.jpg').write_bytes((raw/'back.JPG').read_bytes())
    with pytest.raises(ValueError,match='图片已改变'):app.apply_names(scan['id'])
    duplicate=app.scan(raw)
    assert not duplicate['valid'] and not duplicate['renameable']
    assert '同一方向' in ' '.join(duplicate['rows'][0]['errors'])
    assert not (app.home/'命名副本').exists()


def test_naming_keeps_skipped_rows_visible_and_blocks_during_processing(tmp_path,identified):
    app=web.App(tmp_path/'app');raw=tmp_path/'raw';six(raw/'good');six(raw/'incomplete')
    (raw/'good'/'front.JPG').rename(raw/'good'/'random.jpg');(raw/'incomplete'/'top.JPG').unlink()
    scan=app.scan(raw);app.jobs['running']={'status':'running'}
    with pytest.raises(ValueError,match='等待'):app.apply_names(scan['id'])
    app.jobs.clear();result=app.apply_names(scan['id'])
    assert result['valid']==1 and result['skipped']==['incomplete']
