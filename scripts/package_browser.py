"""Install a local macOS launcher without copying the maintained engine."""
import argparse
from pathlib import Path
import plistlib
import shlex
import subprocess

from phone_texture import __app_version__


def main():
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];home=a.destination.expanduser().resolve();home.mkdir(parents=True,exist_ok=True)
    command=home/'启动六面工坊.command'
    command.write_text('#!/bin/bash\n'+shlex.join([str(repo/'.venv/bin/python'),str(repo/'scripts/launch_web.py'),'--home',str(home)])+'\n')
    command.chmod(0o755)
    # Dispatch a document to Terminal rather than a new app identity directly
    # reading another app's environment inside macOS-protected Documents.
    shell=shlex.join(['/usr/bin/open','-a','Terminal',str(command)])
    script=home/'.launcher.applescript'
    script.write_text('do shell script "'+shell.replace('\\','\\\\').replace('"','\\"')+'"\n')
    bundle=home/'六面工坊.app'
    subprocess.run(['osacompile','-o',str(bundle),str(script)],check=True,capture_output=True)
    info=bundle/'Contents/Info.plist'
    with info.open('rb') as f:value=plistlib.load(f)
    value.update(CFBundleIdentifier='local.phone-texture.six-view-browser',CFBundleName='六面工坊',
                 CFBundleDisplayName='六面工坊', CFBundleShortVersionString=__app_version__,
                 CFBundleVersion=__app_version__, LSUIElement=True)
    with info.open('wb') as f:plistlib.dump(value,f)
    print(bundle)


if __name__=='__main__':main()
