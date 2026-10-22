"""Double-click launcher: reuse this home's server or start one on a free port."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser


def current(home):
    try:
        value=json.loads((home/'.service.json').read_text());url=value['url']
        if not re.fullmatch(r'http://127\.0\.0\.1:\d+',url):return None
        with urlopen(url+'/api/health',timeout=1) as r:state=json.load(r)
        if state.get('app')=='phone-six-view-browser' and state.get('home')==str(home):return url
    except Exception:pass
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True)
    p.add_argument('--no-open',action='store_true');a=p.parse_args()
    home=a.home.expanduser().resolve();home.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1];url=current(home)
    if not url:
        with (home/'服务日志.txt').open('a') as log:
            process=subprocess.Popen([sys.executable,'-m','phone_texture.web_app','--home',str(home),'--port','0','--no-open'],
                                     cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(80):
            url=current(home)
            if url:break
            if process.poll() is not None:
                raise SystemExit('启动失败，请查看 '+str(home/'服务日志.txt'))
            time.sleep(.25)
        else:
            process.terminate()
            raise SystemExit('启动超时，请查看服务日志。')
    if not a.no_open:webbrowser.open(url)
    print(url)


if __name__=='__main__':main()
