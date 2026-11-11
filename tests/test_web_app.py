"""Browser-boundary tests: user input, isolation, completion and cancellation."""
import http.client
import hashlib
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
    body=dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=1)
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
    job=app.start(dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=1))
    assert job['concurrency']==1 and job['requested_concurrency']==1
    assert job['status']=='starting'
    assert Path(job['root']).parent==app.runs
    assert Path(job['work']).parent==Path(job['root'])
    assert Path(job['output']).parent==app.default_output
    assert job['output_base']==str(app.default_output)
    assert not Path(job['output']).exists()  # Pipeline publishes only finished images.
    with pytest.raises(ValueError,match='已有批次'):app.start(dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=1))
    app.cancel(job['id']);assert app.cancel_events[job['id']].is_set()


@pytest.mark.parametrize('legacy', [False, True])
def test_completed_batch_exports_only_six_checked_faces(tmp_path,monkeypatch,legacy):
    app=web.App(tmp_path/'app');raw=six(tmp_path/'raw')
    root=(tmp_path/'run') if legacy else app.runs/'run';root.mkdir()
    work=root/'过程';work.mkdir()
    out=(root/'六面图片') if legacy else tmp_path/'selected-export'/'run'
    six(out/'raw')
    # Real export validation requires PNG output hashes from the worker report.
    hashes={}
    for face in web.FACES:
        Image.new('RGBA',(32,64),'orange').save(out/'raw'/(face+'.png'))
        (out/'raw'/(face+'.JPG')).unlink()
        hashes[face]=web.w.sha(out/'raw'/(face+'.png'))
    web.save(work/'report.json',dict(status='processed_needs_visual_review',processed=1,attempts=[dict(sku='raw',final_sha256=hashes)]))
    (tmp_path/'previous-work/cases').mkdir(parents=True)
    job=dict(id='good',model='iphone17pro',names=['raw'],source=str(raw),single=True,root=str(root),work=str(work),output=str(out),concurrency=1,status='starting',resume_geometry_from=str(tmp_path/'previous-work'))
    app.jobs['good']=job;app.cancel_events['good']=threading.Event()
    if legacy:job['lighting_profile']='accepted-shadow-lift-v1'
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
    assert captured[0][captured[0].index('--protection-policy')+1]=='none'
    assert captured[0][captured[0].index('--resume-geometry-from')+1]==str(tmp_path/'previous-work')
    assert captured[0][captured[0].index('--lighting-profile')+1]==job.get('lighting_profile','accepted-shadow-lift-v2')
    with zipfile.ZipFile(root/'六面图片.zip') as z:
        assert sorted(z.namelist())==sorted('raw/'+f+'.png' for f in web.FACES)
    assert (root/'验收总览.jpg').exists()
    assert app.artifact(dict(job='good',kind='log'))==root/'运行日志.txt'
    assert app.artifact(dict(job='good',kind='zip'))==root/'六面图片.zip'
    if not legacy:
        assert all(p.suffix=='.png' for p in out.parent.rglob('*') if p.is_file())
        assert len([p for p in out.parent.rglob('*') if p.is_file()])==6
    # Older records still retry to their original export parent, not runtime.
    assert app.snapshot(job)['output_base']==str(root.parent)


def test_partial_batch_exports_complete_sku_and_preserves_failed_reason(tmp_path,monkeypatch):
    app=web.App(tmp_path/'app');raw=six(tmp_path/'raw')
    root=app.runs/'mixed';root.mkdir();work=root/'过程';work.mkdir();out=root/'六面图片'
    hashes={}
    (out/'good').mkdir(parents=True)
    for face in web.FACES:
        p=out/'good'/(face+'.png');Image.new('RGBA',(32,64),'orange').save(p);hashes[face]=web.w.sha(p)
    attempts=[dict(sku='good',six_faces_processed=True,final_sha256=hashes),
              dict(sku='bad',status='failed',six_faces_processed=False,error='antenna endpoint uncertain')]
    web.save(work/'report.json',dict(status='partial_candidates_needs_visual_review',processed=1,total=2,attempts=attempts))
    web.save(work/'partial-candidate-manifest.json',dict(generated=1,total=2,visual_approved=0,items=attempts))
    job=dict(id='mixed',model='iphone17pro',names=['good','bad'],source=str(raw),single=False,root=str(root),work=str(work),output=str(out),concurrency=1,status='starting')
    app.jobs['mixed']=job;app.cancel_events['mixed']=threading.Event()
    class Process:
        returncode=2
        def __init__(self,*args,**kwargs):pass
        def poll(self):return 2
    monkeypatch.setattr(web.subprocess,'Popen',Process)
    monkeypatch.setattr(web,'host_sample',lambda:dict(pressure=1,swap=0,available=10*1024**3))
    app.run(job)
    assert job['status']=='partial'
    assert app.snapshot(job)['downloadable']
    with zipfile.ZipFile(app.artifact(dict(job='mixed',kind='zip'))) as z:
        assert sorted(z.namelist())==sorted(['本批处理说明.json']+['good/'+f+'.png' for f in web.FACES])
        manifest=json.loads(z.read('本批处理说明.json'))
        assert manifest['visual_approved']==0
        assert manifest['items'][1]['error']=='antenna endpoint uncertain'


def test_source_preserved_review_is_downloadable_but_never_program_checked(tmp_path,monkeypatch):
    app=web.App(tmp_path/'app');raw=six(tmp_path/'raw')
    root=app.runs/'review';root.mkdir();work=root/'过程';work.mkdir();out=root/'六面图片'
    candidate=work/'cases/kept/candidate/kept';candidate.mkdir(parents=True)
    (out/'kept').mkdir(parents=True);hashes={}
    for face in web.FACES:
        p=candidate/(face+'.png');Image.new('RGBA',(32,64),'orange').save(p)
        (out/'kept'/(face+'.png')).write_bytes(p.read_bytes());hashes[face]=web.w.sha(p)
    row=dict(sku='kept',status='source_preserved_needs_review',six_faces_generated=True,six_faces_processed=False,final_sha256=hashes,
             preserved_faces={'left':{'stage':'geometry_source'}},review_warnings=[{'message':'天线保留原位置'}])
    web.save(work/'cases/kept/result.json',row)
    web.save(work/'report.json',dict(status='review_candidates_needs_visual_review',processed=0,generated=1,total=1,attempts=[row]))
    web.save(work/'partial-candidate-manifest.json',dict(generated=1,total=1,visual_approved=0,items=[row]))
    job=dict(id='review',model='iphone17pro',names=['kept'],source=str(raw),single=True,root=str(root),work=str(work),output=str(out),concurrency=1,status='starting')
    app.jobs['review']=job;app.cancel_events['review']=threading.Event()
    class Process:
        returncode=2
        def __init__(self,*args,**kwargs):pass
        def poll(self):return 2
    monkeypatch.setattr(web.subprocess,'Popen',Process)
    monkeypatch.setattr(web,'host_sample',lambda:dict(pressure=1,swap=0,available=10*1024**3))
    app.run(job);snapshot=app.snapshot(job)
    assert job['status']=='review' and snapshot['downloadable']
    assert snapshot['completed']==1 and snapshot['program_checked_count']==0 and snapshot['review_count']==1
    assert snapshot['rows'][0]['status']=='review'
    assert app.artifact(dict(job='review',kind='image',sku='kept',face='left')).read_bytes()==(candidate/'left.png').read_bytes()
    with zipfile.ZipFile(app.artifact(dict(job='review',kind='zip'))) as z:
        manifest=json.loads(z.read('本批处理说明.json'))
        assert not manifest['items'][0]['six_faces_processed'] and manifest['visual_approved']==0


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


def test_model_buttons_expose_pro_and_promax_as_one_radio_group():
    from html.parser import HTMLParser
    class Models(HTMLParser):
        def __init__(self):
            super().__init__();self.options={};self.selects=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=='select':self.selects.append(attrs.get('id'))
            if tag=='input' and attrs.get('name')=='model':
                assert attrs['type']=='radio'
                self.options[attrs['value']]=attrs['aria-label']
    models=Models();models.feed((web.ASSETS/'index.html').read_text())
    assert models.options=={'iphone17pro':'iPhone 17 Pro','iphone17promax':'iPhone 17 Pro Max'}
    assert 'model' not in models.selects


def test_export_directory_cannot_contain_internal_records(tmp_path,identified):
    app=web.App(tmp_path/'app');scan=app.scan(six(tmp_path/'raw'))
    body=dict(scan_id=scan['id'],skus=['raw'],model='iphone17pro',concurrency=1)
    for path in [app.home,app.runs,app.runs/'nested']:
        with pytest.raises(ValueError,match='运行记录'):app.start(dict(body,output=str(path)))
    assert not app.jobs


def test_full_stage_effect_warning_is_downloadable_and_not_called_unaligned_or_passed(tmp_path,monkeypatch):
    app=web.App(tmp_path/'app');raw=six(tmp_path/'raw')
    root=app.runs/'review';root.mkdir();work=root/'过程';work.mkdir();out=root/'六面图片'
    candidate=work/'cases/kept/candidate/kept';candidate.mkdir(parents=True)
    (out/'kept').mkdir(parents=True);hashes={}
    for face in web.FACES:
        p=candidate/(face+'.png');Image.new('RGBA',(32,64),'orange').save(p)
        (out/'kept'/(face+'.png')).write_bytes(p.read_bytes());hashes[face]=web.w.sha(p)
    row=dict(sku='kept',status='full_stages_effect_warning_needs_review',six_faces_generated=True,six_faces_processed=False,final_sha256=hashes,
             lighting_effect_warnings=[dict(face='back',original_effect_check_passed=False,error='Perimeter effect failed at top')],review_warnings=[])
    web.save(work/'cases/kept/result.json',row)
    web.save(work/'report.json',dict(status='review_candidates_needs_visual_review',processed=0,generated=1,total=1,attempts=[row]))
    web.save(work/'partial-candidate-manifest.json',dict(generated=1,total=1,visual_approved=0,items=[row]))
    job=dict(id='review',model='iphone17pro',names=['kept'],source=str(raw),single=True,root=str(root),work=str(work),output=str(out),concurrency=1,status='starting')
    app.jobs['review']=job;app.cancel_events['review']=threading.Event()
    class Process:
        returncode=2
        def __init__(self,*args,**kwargs):pass
        def poll(self):return 2
    monkeypatch.setattr(web.subprocess,'Popen',Process)
    monkeypatch.setattr(web,'host_sample',lambda:dict(pressure=1,swap=0,available=10*1024**3))
    app.run(job);snapshot=app.snapshot(job)
    # 2026-11-06: full-stage effect warnings pass green; the stored batch status is unchanged.
    assert job['status']=='review' and snapshot['status']=='complete' and snapshot['downloadable']
    assert snapshot['completed']==1 and snapshot['program_checked_count']==1 and snapshot['review_count']==0
    assert snapshot['rows'][0]['status']=='complete'
    assert app.artifact(dict(job='review',kind='image',sku='kept',face='left')).read_bytes()==(candidate/'left.png').read_bytes()
    with zipfile.ZipFile(app.artifact(dict(job='review',kind='zip'))) as z:
        manifest=json.loads(z.read('本批处理说明.json'))
        assert not manifest['items'][0]['six_faces_processed'] and manifest['visual_approved']==0


    assert snapshot['rows'][0]['review_warnings'][0]['code']=='lighting_effect_target_warning'
    assert '天线保留原位置' not in job['message']
    assert snapshot['rows'][0]['all_stages_complete']
    assert snapshot['rows'][0]['structural_alignment_applied']
    assert snapshot['rows'][0]['face_processing']['left']=='seam'
    assert snapshot['rows'][0]['review_only'] and snapshot['retry_count']==0


def test_stage_evidence_overrides_status_and_unknown_face_is_not_called_lit():
    # A retained candidate may already have structural work, with only seams blocked.
    row=dict(status='source_preserved_needs_review',six_faces_generated=True,
             structural_alignment_applied=True,stage_status=dict(geometry='complete',lighting='complete',structure='complete',seam='blocked'),
             face_processing=dict(left='structure'),
             blocked_checks=[dict(stage='seam',error='UV seam color difference exceeds the delivery limit')],
             review_warnings=[dict(code='source_positions_preserved_alignment_unverified',message='天线保留原位置')])
    state=web.processing_state(row)
    assert state['structural_alignment_applied'] and state['retry_required'] and not state['review_only']
    assert '天线已对齐' in state['stage_summary'] and '接缝未完成' in state['stage_summary']
    assert state['face_processing']['right']=='unknown'
    assert not state['review_warnings']
    assert state['review_reasons'][0]['stage']=='seam'
    assert state['review_reasons'][0]['detail'].startswith('UV seam')
    assert web.processing_state({})['face_processing']['front']=='unknown'
    assert web.failure_hint('Bottom paper material boundary unresolved: bottom')=='纸尾与金属的材料边界待核对；该面保留几何结果。'
    assert web.failure_hint('Ambiguous crossing side hardware contours')=='同一控件存在多个候选材料边界，需核对；这不代表原图有缺陷。'


def test_retry_selects_incomplete_stages_and_creates_separate_checkpoint_batch(tmp_path,monkeypatch,identified):
    app=web.App(tmp_path/'app');source=tmp_path/'raw'
    names=['aligned','effect-warning','pending','failed']
    for name in names:six(source/name)
    old=app.runs/'old';old.mkdir();work=old/'过程';work.mkdir()
    results={
        'aligned':dict(status='processed_needs_visual_review',six_faces_generated=True,six_faces_processed=True),
        'effect-warning':dict(status='full_stages_effect_warning_needs_review',six_faces_generated=True,six_faces_processed=False),
        'pending':dict(status='source_preserved_needs_review',six_faces_generated=True,six_faces_processed=False,
                       stage_status=dict(geometry='complete',lighting='complete',structure='blocked',seam='blocked')),
        'failed':dict(status='failed',six_faces_generated=False)}
    for name,data in results.items():
        case=work/'cases'/name;case.mkdir(parents=True);web.save(case/'result.json',data)
    job=dict(id='old',status='partial',names=names,root=str(old),work=str(work),source=str(source),single=False,
             model='iphone17pro',concurrency=3,output=str(tmp_path/'export'/'old'),output_base=str(tmp_path/'export'))
    app.jobs['old']=job
    monkeypatch.setattr(threading.Thread,'start',lambda self:None)
    monkeypatch.setattr(web,'host_sample',lambda:dict(pressure=1,available=10*1024**3,swap=0))
    assert app.retry_names(job)==['pending','failed']
    with pytest.raises(ValueError,match='复核'):app.retry(dict(id='old',skus=['effect-warning']))
    before=(work/'cases/pending/result.json').read_bytes()
    new=app.retry(dict(id='old'))
    assert new['names']==['pending','failed'] and new['parent_job_id']=='old'
    assert new['resume_geometry_from']==str(work)
    assert new['work']!=job['work'] and new['output']!=job['output']
    assert (work/'cases/pending/result.json').read_bytes()==before


def package_fixture(tmp_path):
    app=web.App(tmp_path/'app');root=app.runs/'batch';root.mkdir();work=root/'过程';work.mkdir();out=tmp_path/'export'
    names=['kept','warning'];rows=[]
    for sku,status in zip(names,['source_preserved_needs_review','full_stages_effect_warning_needs_review']):
        folder=out/sku;folder.mkdir(parents=True);hashes={}
        for i,face in enumerate(web.FACES):
            path=folder/(face+'.png');Image.new('RGBA',(12,20),(30+i*20,60,90,255)).save(path);hashes[face]=web.w.sha(path)
        rows.append(dict(sku=sku,status=status,six_faces_generated=True,six_faces_processed=False,final_sha256=hashes))
    report=dict(status='review_candidates_needs_visual_review',generated=2,processed=0,total=2,attempts=rows,exported_skus=names)
    web.save(work/'partial-candidate-manifest.json',dict(generated=2,total=2,visual_approved=0,items=rows))
    job=dict(id='batch',names=names,root=str(root),work=str(work),output=str(out),status='review')
    app.jobs[job['id']]=job
    return app,job,report


def test_package_warning_candidates_zip_hashes_and_download_integrity(tmp_path):
    app,job,report=package_fixture(tmp_path)
    rows=app.package(job,report)
    assert len(rows)==2
    with zipfile.ZipFile(app.artifact(dict(job='batch',kind='zip'))) as archive:
        expected={r['sku']+'/'+f+'.png':r['final_sha256'][f] for r in rows for f in web.FACES}
        assert set(archive.namelist())==set(expected)|{'本批处理说明.json'}
        for name,digest in expected.items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        note=json.loads(archive.read('本批处理说明.json'))
        assert note['exported_files']==expected and note['visual_approved']==0
    (Path(job['root'])/'六面图片.zip').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='校验不一致'):app.artifact(dict(job='batch',kind='zip'))


@pytest.mark.parametrize('damage',['missing','changed','manifest','exported-set'])
def test_package_does_not_hide_missing_or_changed_warning_exports(tmp_path,damage):
    app,job,report=package_fixture(tmp_path)
    target=Path(job['output'])/'warning/front.png'
    if damage=='missing':target.unlink()
    elif damage=='changed':Image.new('RGBA',(12,20),'red').save(target)
    elif damage=='exported-set':report['exported_skus']=['kept']
    else:web.save(Path(job['work'])/'partial-candidate-manifest.json',dict(generated=1,items=[report['attempts'][0]]))
    with pytest.raises(ValueError):app.package(job,report)
    assert not (Path(job['root'])/'六面图片.zip').exists()
    assert not (Path(job['root'])/'六面图片.zip.tmp').exists()


def test_browser_stage_and_viewer_text_use_actual_evidence():
    # Execute the same pure render functions used by the browser, not copied logic.
    import shutil
    executable=shutil.which('node')
    if not executable:pytest.skip('Node unavailable for browser render check')
    script=(web.ASSETS/'app.js').read_text()
    functions='\n'.join(line for line in script.splitlines() if line.startswith(('function stageText(', 'function faceText(', 'function seconds(')))
    fixture=dict(status='review',seconds=2,stage_summary='调光、天线对齐和接缝处理已完成 · 效果待核对',face_processing=dict(left='seam'))
    code=functions+'\nconst row='+json.dumps(fixture,ensure_ascii=False)+';const partial={face_processing:{left:"structure"},per_face_stages:{left:{lighting:"failed"}}};console.log(JSON.stringify([stageText(row),faceText(row,"left"),faceText(row,"right"),faceText(partial,"left")]));'
    values=json.loads(subprocess.check_output([executable,'-e',code],text=True))
    assert '天线保留原位置' not in values[0] and '已完成' in values[0]
    assert values[1]=='调光与接缝已处理'
    assert values[2]=='该面处理阶段待核对'
    assert values[3]=='天线已对齐，调光未完成'


def test_http_download_preserves_warning_candidates_and_hashes(tmp_path,monkeypatch):
    app,job,report=package_fixture(tmp_path)
    for row in report['attempts']:
        case=Path(job['work'])/'cases'/row['sku'];case.mkdir(parents=True)
        web.save(case/'result.json',row)
    app.package(job,report)
    monkeypatch.setattr(web,'host_sample',lambda:dict(pressure=1,available=10*1024**3,swap=0))
    server=web.ThreadingHTTPServer(('127.0.0.1',0),web.handler(app))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        def get(path):
            connection=http.client.HTTPConnection('127.0.0.1',server.server_port)
            connection.request('GET',path);response=connection.getresponse();data=response.read();connection.close()
            return response.status,data
        status,data=get('/api/state');assert status==200
        state=json.loads(data)['jobs'][0]
        assert state['completed']==2 and state['review_count']==1 and state['status']=='review'
        assert state['retry_count']==1 and state['review_only_count']==0
        assert state['downloadable']
        assert state['rows'][1]['structural_alignment_applied']
        status,data=get('/api/artifact?job=batch&kind=zip&download=1')
        assert status==200 and hashlib.sha256(data).hexdigest()==job['artifact_sha256']['zip']
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert len(archive.namelist())==13
            assert 'warning/front.png' in archive.namelist()
            assert json.loads(archive.read('本批处理说明.json'))['visual_approved']==0
    finally:
        server.shutdown();server.server_close();thread.join()
