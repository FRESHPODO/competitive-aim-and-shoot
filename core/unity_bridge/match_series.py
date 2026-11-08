"""Frozen-model repeated duels under a custom scenario; no training occurs."""
import argparse,copy,hashlib,json,time,shutil
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from unity_bridge.server import DualSession
from unity_bridge.duel_policy import DuelPolicy, bind_duel_policies
from unity_bridge.training_settings import migrate,validate,sample
from unity_bridge.reward_generation.rewards import normalize,validate as validate_rewards,sample as sample_rewards,breakdown
from unity_bridge.trainer_state import save_state
from unity_bridge.replay import save_replay
from unity_bridge.storage import resolve_path,checkpoint_directory,model_lock

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scenario',required=True);p.add_argument('--model-a',required=True);p.add_argument('--model-b',required=True)
    p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=20);p.add_argument('--seed',type=int,default=900123)
    p.add_argument('--timeout',type=float,default=30);p.add_argument('--replays',type=int,default=0,help='Optional HTML exports; native JSON replays are always saved for ALL rounds')
    p.add_argument('--resample',action='store_true',help='Draw ranged conditions again per round; otherwise freeze one draw')
    a=p.parse_args()
    if a.rounds<1 or a.replays<0 or a.seed<0 or not np.isfinite(a.timeout) or a.timeout<=0:p.error('Invalid rounds, replay count, seed or timeout')
    document=json.loads(Path(a.scenario).read_text(encoding='utf-8-sig'))
    settings=migrate(document.get('parameters',document));validate(settings)
    reward_config=normalize(document.get('rewardConfig',{}));validate_rewards(reward_config)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise ValueError('Choose an unused result folder')
    models=[];sources=[]
    inputs=[('a',resolve_path(a.model_a)),('b',resolve_path(a.model_b))]
    with ExitStack() as locks:
        runs={run for _,path in inputs if (run:=checkpoint_directory(path)) is not None}
        for run in sorted(runs):locks.enter_context(model_lock(run))
        for side,source in inputs:
            frozen=output/f'agent_{side}.zip';shutil.copyfile(source,frozen)
            sources.append(dict(source=str(source),sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()))
            models.append(frozen)
    manifest=dict(status='running',rounds=a.rounds,seed=a.seed,timeoutSeconds=a.timeout,resample=a.resample,
        models=sources,parameters=settings,rewardConfig=reward_config,playerPresets=document.get('playerPresets',{}))
    save_state(output/'manifest.json',manifest)
    torch.set_num_threads(1);started=time.monotonic();rows=[]
    rng=np.random.default_rng(a.seed);fixed=sample(settings,rng);fixed['rewardValues']=sample_rewards(reward_config,rng)
    def report(status):
        summary=dict(status=status,completed=len(rows),requested=a.rounds,
            winsA=sum(r['winner']=='A' for r in rows),winsB=sum(r['winner']=='B' for r in rows),
            draws=sum(r['winner'] is None for r in rows),timeouts=sum(r['outcome']=='time_limit_draw' for r in rows),
            elapsedSeconds=time.monotonic()-started,pid=__import__('os').getpid(),seed=a.seed)
        count=summary['completed']
        summary.update(winRateA=summary['winsA']/count if count else None,winRateB=summary['winsB']/count if count else None,drawRate=summary['draws']/count if count else None,rateDenominator='all completed matches including draws')
        save_state(output/'progress.json',summary);save_state(output/'summary.json',summary)
    report('running')
    try:
        policies=[DuelPolicy(path) for path in models];dual=DualSession(recording='replay')
        for index in range(a.rounds):
            if (output/'stop.request').exists():manifest['status']='stopped';break
            seed=a.seed+index
            if a.resample:
                rng=np.random.default_rng(seed);request=sample(settings,rng);request['rewardValues']=sample_rewards(reward_config,rng)
            else:request=copy.deepcopy(fixed)
            request['playerPresets']=document.get('playerPresets',{})
            dual.round=seed;dual.reset(request);dual.inactivity_limit_ms=a.timeout*1000
            bind_duel_policies(dual,policies,request)
            while dual.time<a.timeout*1000 and dual.combat.hp>0 and not dual.draw:dual.step(deadline_ms=a.timeout*1000)
            killers={e['agent'] for e in dual.combat.events if e.get('killed')}
            winner='AB'[next(iter(killers))] if len(killers)==1 else None
            outcome='kill' if winner else 'shared_kill' if killers else 'time_limit_draw'
            components=breakdown(dual.combat.events,dual.combat.maximum_hp,request['rewardValues'])
            row=dict(round=index+1,seed=seed,winner=winner,outcome=outcome,timeMs=dual.time,remainingHp=dual.combat.hp,
                shots=[sum(e['agent']==i for e in dual.combat.events) for i in range(2)],
                hits=[sum(e['agent']==i and e['hit'] for e in dual.combat.events) for i in range(2)],
                damage=[part['actualDamage'] for part in components],
                criticalHits=[sum(e['agent']==i and e['critical'] for e in dual.combat.events) for i in range(2)],
                blocked=[sum(not d['approved'] for d in s.strategy_log) for s in dual.shooters],
                fireDecisions=[sum(bool(d['approved']) for d in s.strategy_log) for s in dual.shooters],
                cooldownBlocked=[sum(bool(d.get('cooldownBlocked')) for d in s.decision_log) for s in dual.shooters],
                rewards=[part['total'] for part in components],rewardComponents=components,settings=request,replay=None)
            path=output/'replays'/f'round-{index+1:05d}.html'
            save_replay(dual,path,request,f'Custom duel {index+1} / seed {seed}',html=index<a.replays)
            row['replay']=str(path.with_suffix('.json').resolve())
            rows.append(row)
            with (output/'rounds.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            report('running');print(f'Round {index+1}/{a.rounds}: {winner or outcome}',flush=True)
        else:manifest['status']='completed'
    except BaseException:
        manifest['status']='failed';raise
    finally:
        save_state(output/'manifest.json',manifest);report(manifest['status'])

if __name__=='__main__':main()
