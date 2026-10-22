"""Bounded extreme-concurrency experiment; originals-only complete six-face jobs.

One configuration per invocation. Append runs to a fresh experiment root while
freezing implementation and source hashes; severe pressure stops the process group.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark_automatic import host_sample,code_hashes


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def stop(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--model',required=True,choices=('iphone17pro','iphone17promax'));p.add_argument('--records',type=Path,required=True)
    p.add_argument('--jobs',type=int,required=True);p.add_argument('--inference-slots',type=int,default=2)
    p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if not 1<=a.jobs<=10 or not 1<=a.inference_slots<=min(a.jobs,4):p.error('unsupported limits')
    records=a.records.resolve();records.mkdir(parents=True,exist_ok=True)
    baseline=code_hashes(repo)
    baseline['scripts/benchmark_extreme.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(a.input.resolve().glob('SKU_*/*')) if x.is_file() and x.suffix.lower() in ('.png','.jpg','.jpeg')}
    if len(sources)!=60:raise ValueError('This experiment requires ten original six-face SKUs')
    protocol=dict(source=str(a.input.resolve()),implementation_sha256=baseline,source_sha256=sources,
        cache_policy='Fresh masks, geometry, material analysis, lighting, alignment per run. Installed model weights and OS cache may be warm.',
        severe_pressure_stop=4,maximum_swap_growth_bytes=8*1024**3,timeout_seconds=600,
        scope='complete_six_face_candidates_not_final_visual_or_model_uv_acceptance',reference_outputs=str(a.reference.resolve()))
    pp=records/'protocol.json'
    if pp.exists():
        if json.loads(pp.read_text())!=protocol:raise RuntimeError('Frozen experiment implementation/source/protocol changed')
    else:write(pp,protocol)
    name=f'jobs-{a.jobs}-inference-{a.inference_slots}';folder=records/name;folder.mkdir(exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    cmd=[str(repo/'.venv/bin/phone-texture'),'auto',str(a.input.resolve()),'--model',a.model,'--jobs',str(a.jobs),'--inference-slots',str(a.inference_slots),'--experimental-concurrency','--device','mps','--work',str(folder/'process'),'--output',str(folder/'six-face-candidates')]
    samples=[host_sample()];initial=samples[0];start=time.perf_counter();guard=None
    with (folder/'console.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                sample=host_sample(proc);sample['elapsed_seconds']=time.perf_counter()-start;samples.append(sample)
                write(folder/'live-host.json',sample)
                if sample['pressure_level']>=4:guard='severe_memory_pressure'
                elif sample['swap_used_bytes']-initial['swap_used_bytes']>8*1024**3:guard='swap_growth_exceeded_8GiB'
                elif time.perf_counter()-start>600:guard='timeout_600_seconds'
                if guard:stop(proc);break
                time.sleep(1)
        except BaseException:
            stop(proc);raise
    wall=time.perf_counter()-start;samples.append(host_sample());write(folder/'host-samples.json',samples)
    report_path=folder/'process/report.json';report=json.loads(report_path.read_text()) if report_path.exists() else {}
    attempts=report.get('attempts',[])
    if not attempts:
        progress=folder/'process/progress.json'
        attempts=json.loads(progress.read_text()).get('attempts',[]) if progress.exists() else []
    complete=sum(x.get('six_faces_processed',False) for x in attempts)
    faces=list((folder/'six-face-candidates').glob('*/*.png'))
    good=(proc.returncode==0 and complete==report.get('total')==10 and len(faces)==60 and guard is None)
    equality=None;fallbacks=sum(v['fallback_count'] for r in attempts for v in r.get('device_observations',[]))
    if good:
        equality=all(hashlib.sha256(f.read_bytes()).hexdigest()==hashlib.sha256((a.reference/f.relative_to(folder/'six-face-candidates')).read_bytes()).hexdigest() for f in faces)
        good=equality and fallbacks==0
    # Failures still report elapsed time, but never a successful throughput.
    row=dict(configuration=name,jobs=a.jobs,inference_slots=a.inference_slots,status='complete_candidates' if good else 'stopped' if guard else 'failed',
        processed=complete,total=10,output_faces=len(faces),wall_seconds=wall,amortized_seconds_per_sku=wall/10 if good else None,
        mean_execution_seconds=sum(r['seconds'] for r in attempts)/len(attempts) if attempts else None,
        maximum_pressure_level=max(s['pressure_level'] for s in samples),
        initial_swap_bytes=initial['swap_used_bytes'],maximum_swap_growth_bytes=max(s['swap_used_bytes'] for s in samples)-initial['swap_used_bytes'],
        final_swap_bytes=samples[-1]['swap_used_bytes'],minimum_available_bytes=min(s['available_bytes'] for s in samples),
        maximum_process_rss_sum_bytes=max(s['process_rss_sum_bytes'] for s in samples),
        pressure_sample_counts={str(k):sum(s['pressure_level']==k for s in samples) for k in sorted(set(s['pressure_level'] for s in samples))},
        cpu_fallbacks=fallbacks,byte_equal_to_verified_reference=equality,guard=guard,returncode=proc.returncode,
        errors=[dict(sku=r['sku'],error=r['error']) for r in attempts if r.get('error')])
    current=code_hashes(repo);current['scripts/benchmark_extreme.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current!=baseline:raise RuntimeError('Implementation changed during experiment')
    if any(hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h for f,h in sources.items()):raise RuntimeError('Original changed')
    write(folder/'result.json',row)
    summary_path=records/'summary.json';summary=json.loads(summary_path.read_text()) if summary_path.exists() else []
    summary.append(row);write(summary_path,summary)
    print(json.dumps(row,ensure_ascii=False,indent=2),flush=True)
    return 0 if good else 2


if __name__=='__main__':raise SystemExit(main())
