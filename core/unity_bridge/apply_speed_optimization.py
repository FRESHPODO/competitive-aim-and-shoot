"""Checkpoint and resume the three selected live runs without moving their records."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import checkpoint,save_state


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--optimization',choices=('single_segment_belief_interpolation','cached_motor_speed'),default='single_segment_belief_interpolation')
    parser.add_argument('--recover-report',type=Path,help='Continue from an interrupted maintenance report')
    options=parser.parse_args()
    baseline=read(ROOT/'storage-reports/current-training-speed.json')
    folder=options.recover_report.parent if options.recover_report else ROOT/'storage-reports/maintenance'/f'speed-{time.strftime("%Y%m%d-%H%M%S")}'
    folder.mkdir(parents=True,exist_ok=bool(options.recover_report));items=read(options.recover_report) if options.recover_report else []
    def report():save_state(folder/'result.json',items)
    for entry in ([] if options.recover_report else baseline):
        run=(ROOT/'training-results'/entry['run']).resolve()
        assert run.is_relative_to((ROOT/'training-results').resolve())
        progress=read(run/'progress.json');meta=read(run/'training.json')
        if progress['status']!='running' or (run/'stop.request').exists():continue
        if options.optimization in meta.get('runtimeOptimizations',[]):continue
        process=psutil.Process(progress['pid']);command=process.cmdline()
        if 'unity_bridge.train_competitive' not in command:raise RuntimeError('Unexpected training process')
        assert Path(command[command.index('--output')+1]).resolve()==run
        backup=folder/run.name;backup.mkdir()
        for name in ('training.json','progress.json','settings.json','rewards.json'):
            (backup/name).write_bytes((run/name).read_bytes())
        item=dict(run=str(run),oldPid=process.pid,beforeEpisodes=progress['episodes'],target=progress['target'],
                  command=command,settingsHash=hashlib.sha256((run/'settings.json').read_bytes()).hexdigest(),
                  rewardsHash=hashlib.sha256((run/'rewards.json').read_bytes()).hexdigest(),state='stop_requested')
        items.append(item);(run/'stop.request').touch();report()
        print('Checkpoint requested:',run.name,flush=True)
    deadline=time.monotonic()+120
    while any(i['state']=='stop_requested' for i in items):
        for item in items:
            if item['state']!='stop_requested':continue
            run=Path(item['run']);p=read(run/'progress.json')
            if p['status'] not in ('stopped','completed'):continue
            if psutil.pid_exists(item['oldPid']):continue
            source,marker=checkpoint(run)
            assert marker['episodes']==p['episodes'] and p['episodes']>=item['beforeEpisodes']
            item['savedEpisodes']=p['episodes'];item['state']='checkpointed'
            print('Checkpoint complete:',run.name,p['episodes'],flush=True);report()
        if time.monotonic()>deadline:raise TimeoutError('Training checkpoint did not complete; do not force kill')
        time.sleep(.2)
    for item in items:
        if item['state']!='checkpointed':continue
        run=Path(item['run']);meta=read(run/'training.json')
        current=read(run/'progress.json')
        if current['status'] not in ('stopped','completed') or psutil.pid_exists(item['oldPid']):
            raise RuntimeError('Recovery requires a saved, stopped process')
        _,marker=checkpoint(run)
        assert marker['episodes']==current['episodes']==item['savedEpisodes']
        if item['savedEpisodes']>=item['target']:item['state']='already_completed';continue
        log=run/'episodes.jsonl';item['episodePrefixBytes']=log.stat().st_size
        item['episodePrefixHash']=hashlib.sha256(log.read_bytes()).hexdigest()
        command=item['command'];module=command.index('unity_bridge.train_competitive')
        args=[str(ROOT.parent/'.venv/Scripts/python.exe'),'-u','-m','unity_bridge.train_competitive',
              *command[module+1:],'--resume',str(run),'--model-output',meta['modelDirectory'],'--continue-in-place']
        (run/'stop.request').unlink(missing_ok=True)
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
        with (run/'training.log').open('a',encoding='utf-8') as stream:
            stream.write(f'\nApplying verified runtime optimization {options.optimization}; continuing saved training.\n');stream.flush()
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        item.update(state='restarting',launcherPid=process.pid);report()
    deadline=time.monotonic()+90
    while any(i['state']=='restarting' for i in items):
        for item in items:
            if item['state']!='restarting':continue
            run=Path(item['run']);p=read(run/'progress.json')
            if p['status']=='running' and p['pid']!=item['oldPid']:
                assert p['episodes']>=item['savedEpisodes'] and p['target']==item['target']
                assert hashlib.sha256((run/'settings.json').read_bytes()).hexdigest()==item['settingsHash']
                assert hashlib.sha256((run/'rewards.json').read_bytes()).hexdigest()==item['rewardsHash']
                with (run/'episodes.jsonl').open('rb') as f:
                    assert hashlib.sha256(f.read(item['episodePrefixBytes'])).hexdigest()==item['episodePrefixHash']
                assert options.optimization in read(run/'training.json')['runtimeOptimizations']
                item.update(state='running',newPid=p['pid'],readyEpisodes=p['episodes']);report()
                print('Resumed:',run.name,p['episodes'],'/',p['target'],flush=True)
        if time.monotonic()>deadline:raise TimeoutError('Resume not ready; inspect training.log')
        time.sleep(.25)
    report();print('REPORT',folder/'result.json',flush=True)


if __name__=='__main__':main()
