"""Fresh original-only six-face benchmark; failed/incomplete cases never count."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

import psutil


def host_sample(process=None):
    raw=subprocess.check_output(['sysctl','kern.memorystatus_vm_pressure_level','vm.swapusage'],text=True)
    pressure=int(re.search(r'pressure_level: (\d+)',raw).group(1))
    swap=float(re.search(r'used = ([\d.]+)M',raw).group(1))*1024**2
    rows=[]
    if process is not None:
        try:
            root=psutil.Process(process.pid)
            for p in [root,*root.children(recursive=True)]:
                try:rows.append(dict(pid=p.pid,rss=p.memory_info().rss,cpu_seconds=sum(p.cpu_times()[:2])))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        except psutil.NoSuchProcess:pass
    return dict(time=time.time(),pressure_level=pressure,swap_used_bytes=int(swap),
        available_bytes=psutil.virtual_memory().available,processes=rows,
        process_rss_sum_bytes=sum(r['rss'] for r in rows))


def code_hashes(repo):
    paths=[*repo.glob('phone_texture/*.py'),repo/'skills/phone-aluminum-frame-restore/scripts/harmonize_aluminum.py']
    return {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--model',required=True,choices=('iphone17pro','iphone17promax'))
    p.add_argument('--records',type=Path,required=True);p.add_argument('--levels',type=int,nargs='+',default=[1,3,4]);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];records=a.records.resolve();records.mkdir(parents=True,exist_ok=False)
    if any(n not in range(1,5) for n in a.levels):p.error('levels must be 1..4')
    baseline=code_hashes(repo);protocol=dict(source=str(a.input.resolve()),levels=a.levels,
        machine=platform.platform(),machine_memory_bytes=psutil.virtual_memory().total,
        cpu_count=psutil.cpu_count(),implementation_sha256=baseline,
        scope='All six segmentation calls, fresh geometry/material/landmark analysis, lighting, alignment, numeric checks, evidence and output I/O',
        cache_policy='No historical image masks, guides, landmarks, approvals or processed images; installed model weights and OS file cache may be warm',
        final_visual_acceptance_timed=False,model_uv_verified=False)
    (records/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));summary=[]
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    for level in a.levels:
        if code_hashes(repo)!=baseline:raise RuntimeError('Implementation changed during the benchmark')
        folder=records/f'workers-{level}';folder.mkdir();initial=host_sample();samples=[initial];start=time.perf_counter();guard=None
        command=[str(repo/'.venv/bin/phone-texture'),'auto',str(a.input.resolve()),'--model',a.model,'--work',str(folder/'process'),'--output',str(folder/'six-face-candidates'),'--jobs',str(level)]
        with (folder/'console.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                row=host_sample(proc);row['elapsed_seconds']=time.perf_counter()-start;samples.append(row)
                if row['pressure_level']>=4 or row['swap_used_bytes']-initial['swap_used_bytes']>8*1024**3:
                    guard='Critical host memory pressure or swap growth over 8 GiB';os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(2)
        wall=time.perf_counter()-start;samples.append(host_sample());report_path=folder/'process/report.json'
        (folder/'host-samples.json').write_text(json.dumps(samples,indent=2))
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        attempts=report.get('attempts',[]);completed=report.get('processed',0);total=report.get('total',0)
        faces=list((folder/'six-face-candidates').glob('*/*.png'))
        good=(proc.returncode==0 and total>0 and completed==total and len(faces)==total*6 and guard is None)
        forward=[q for r in attempts for q in r.get('post_forward_memory_samples',[])]
        row=dict(workers=level,status='complete_candidates' if good else 'failed',processed=completed,total=total,faces=len(faces),
            wall_seconds=wall,amortized_seconds_per_sku=wall/completed if good else None,
            mean_sku_latency_seconds=sum(r['seconds'] for r in attempts)/len(attempts) if attempts else None,
            candidate_skus_per_hour=3600*completed/wall if good else None,
            maximum_pressure_level=max(s['pressure_level'] for s in samples),
            swap_growth_bytes=samples[-1]['swap_used_bytes']-initial['swap_used_bytes'],
            maximum_process_rss_sum_bytes=max(s['process_rss_sum_bytes'] for s in samples),
            maximum_observed_post_forward_driver_bytes=max([q['mps_driver_bytes'] for q in forward],default=0),
            mps_memory_measurement='Per-process post-forward snapshot, not true peak or aggregate GPU use',
            visual_approved=report.get('visual_approved',0),guard=guard,returncode=proc.returncode)
        summary.append(row);(records/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        with (records/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(summary)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if not good:raise RuntimeError('Full six-face candidate benchmark failed; no success throughput reported')
    if code_hashes(repo)!=baseline:raise RuntimeError('Implementation changed during the benchmark')
    # Repeatability is measured, never supplied as an input/cached result.
    digests={}
    for level in a.levels:
        folder=records/f'workers-{level}/six-face-candidates'
        digests[str(level)]={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*/*.png'))}
    (records/'output-repeatability.json').write_text(json.dumps(dict(sha256=digests,byte_equal=all(v==next(iter(digests.values())) for v in digests.values())),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
