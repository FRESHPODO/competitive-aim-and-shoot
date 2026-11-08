"""Move legacy records/models to distinct project roots and resume active jobs."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import psutil
from unity_bridge.storage import ROOT,TRAINING_ROOT,MODEL_ROOT,MATCH_ROOT,REPORT_ROOT,MIGRATIONS,resolve_path,read_json
from unity_bridge.trainer_state import save_state,checkpoint

def checked_move(source,target,moves):
    source=source.resolve();target=target.resolve()
    if not source.is_relative_to(ROOT) or not target.is_relative_to(ROOT):raise ValueError('Outside workspace')
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(target));moves[str(source)]=str(target)

def active_legacy():
    jobs=[]
    for p in psutil.process_iter(['cmdline','create_time']):
        try:
            args=p.info['cmdline'] or []
            if 'unity_bridge.train_competitive' not in args or '--output' not in args:continue
            folder=Path(args[args.index('--output')+1]).resolve()
            if folder.parent!=ROOT/'local-results'/'competitive':continue
            jobs.append(dict(pid=p.pid,created=p.create_time(),directory=str(folder),args=args))
        except psutil.Error:pass
    return jobs

def rewrite_references(moves):
    replacements=[]
    for old,new in sorted(moves.items(),key=lambda pair:len(pair[0]),reverse=True):
        replacements.extend([(old,new),(old.replace('\\','/'),new.replace('\\','/'))])
    def rewrite(value):
        if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
        if isinstance(value,list):return [rewrite(v) for v in value]
        if isinstance(value,str):
            for old,new in replacements:value=value.replace(old,new)
        return value
    count=0
    for base in [TRAINING_ROOT,MODEL_ROOT,MATCH_ROOT,ROOT/'local-results',ROOT/'configs']:
        for path in base.rglob('*.json'):
            if path.stat().st_size>8_000_000 or 'snapshots' in path.parts or 'live' in path.parts:continue
            data=read_json(path)
            if not data:continue
            new=rewrite(data)
            if new!=data:save_state(path,new);count+=1
    for path in MATCH_ROOT.rglob('*.jsonl'):
        lines=[];dirty=False
        for line in path.read_text(encoding='utf-8').splitlines():
            data=json.loads(line);new=rewrite(data);dirty|=new!=data;lines.append(json.dumps(new,ensure_ascii=False))
        if dirty:path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return count

def migrate(restart_active=False):
    REPORT_ROOT.mkdir(parents=True,exist_ok=True);jobs=active_legacy()
    if jobs and not restart_active:raise RuntimeError('Active training detected; --restart-active is needed to checkpoint and resume it')
    save_state(REPORT_ROOT/'active-training-maintenance.json',dict(jobs=jobs,status='prepared'))
    if jobs:
        for job in jobs:(Path(job['directory'])/'stop.request').touch()
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            waiting=[]
            for job in jobs:
                try:
                    p=psutil.Process(job['pid'])
                    if p.create_time()==job['created'] and p.is_running():waiting.append(job['pid'])
                except psutil.Error:pass
            if not waiting:break
            time.sleep(.5)
        else:raise RuntimeError(f'Workers have not checkpointed/exited; nothing has been moved: {waiting}')
    moves=read_json(MIGRATIONS).get('moves',{});verified=[]
    legacy=ROOT/'local-results'/'competitive'
    if legacy.exists():
        for old in sorted(legacy.iterdir()):
            if not old.is_dir():continue
            records=TRAINING_ROOT/old.name;models=MODEL_ROOT/old.name
            metadata=read_json(old/'training.json');models.mkdir(parents=True,exist_ok=True)
            hashes={name:hashlib.sha256((old/name).read_bytes()).hexdigest() for name in ['agent_a.zip','agent_b.zip'] if (old/name).exists()}
            for name in ['agent_a.zip','agent_b.zip','snapshots','torch_rng.pt','cuda_rng.pt']:
                if (old/name).exists():checked_move(old/name,models/name,moves)
            for replay in list(old.glob('replay-*')):
                if replay.is_dir():checked_move(replay,MATCH_ROOT/(old.name+'-'+replay.name),moves)
            checked_move(old,records,moves)
            if metadata:
                metadata['modelDirectory']=str(models);save_state(records/'training.json',metadata)
                save_state(models/'model-manifest.json',dict(trainingDirectory=str(records),completedEpisodes=metadata.get('completedEpisodes'),targetEpisodes=metadata.get('targetEpisodes'),status=metadata.get('status'),legacySnapshots=True))
            for name,digest in hashes.items():
                if hashlib.sha256((models/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Model checksum changed')
            verified.append(dict(run=old.name,sha256=hashes))
    legacy_matches=ROOT/'local-results'/'custom-duels'
    if legacy_matches.exists():
        for old in sorted(legacy_matches.iterdir()):
            if old.is_dir():checked_move(old,MATCH_ROOT/old.name,moves)
    save_state(MIGRATIONS,dict(moves=moves,updated=datetime.now().isoformat()))
    changed=rewrite_references(moves)
    restarted=[]
    for job in jobs:
        prior=resolve_path(job['directory']);metadata=read_json(prior/'training.json');source,marker=checkpoint(prior)
        if marker['episodes']>=metadata['targetEpisodes']:continue
        output=TRAINING_ROOT/(prior.name+'-continued-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        output.mkdir(parents=True)
        command=[sys.executable,'-u','-m','unity_bridge.train_competitive','--settings',str(prior/'settings.json'),'--output',str(output),
            '--resume',str(prior),'--episodes',str(metadata['targetEpisodes']),'--timeout',str(metadata['timeoutSeconds']),
            '--seed',str(metadata['seed']),'--batch-episodes',str(metadata['batchEpisodes']),'--device',metadata.get('device','cpu'),'--snapshot-interval','10000']
        if (prior/'rewards.json').exists():command+=['--rewards',str(prior/'rewards.json')]
        with (output/'training.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        restarted.append(dict(previous=str(prior),output=str(output),pid=process.pid,resumedEpisode=marker['episodes'],target=metadata['targetEpisodes']))
    state_path=ROOT/'local-results'/'trainer-state.json';state=read_json(state_path)
    if state and restarted:
        selected=next((j for j in restarted if j['previous']==state.get('lastRun')),restarted[-1])
        state['lastRun']=selected['output'];state['resume']=selected['output'];save_state(state_path,state)
    report=dict(status='migrated',moves=len(moves),updatedReferences=changed,verifiedModelHashes=verified,restarted=restarted)
    save_state(REPORT_ROOT/'storage-migration-result.json',report)
    save_state(REPORT_ROOT/'active-training-maintenance.json',dict(jobs=jobs,status='migrated-and-resumed',restarted=restarted))
    print(json.dumps(report,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--restart-active',action='store_true');a=p.parse_args()
    if a.apply:migrate(a.restart_active)
    else:print(json.dumps(dict(active=active_legacy(),trainingRoot=str(TRAINING_ROOT),modelRoot=str(MODEL_ROOT),matchRoot=str(MATCH_ROOT)),ensure_ascii=False))
if __name__=='__main__':main()
