"""Paired full-round evaluation against ungated A, using held-out seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from unity_bridge.server import DualSession
from unity_bridge.decision_env import sample_scenario
from unity_bridge.fire_decider import FireDecider, MODEL_DIR


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default=str(MODEL_DIR/'candidate.zip'))
    parser.add_argument('--rounds',type=int,default=40)
    parser.add_argument('--scenario', help='Same fixed JSON overrides used for condition training')
    parser.add_argument('--replays',type=int,default=1,help='Save first N paired rounds as standalone 3D HTML')
    parser.add_argument('--seed',type=int,default=90123)
    parser.add_argument('--output',default=str(MODEL_DIR/'evaluation.json'))
    args=parser.parse_args()
    if args.rounds < 1: parser.error('rounds must be positive')
    scenario=json.loads(Path(args.scenario).read_text(encoding='utf-8')) if args.scenario else None
    torch.set_num_threads(1)
    policy=FireDecider(args.model)
    dual=DualSession()
    rows=[]
    for index in range(args.rounds):
        seed=args.seed+index
        rng=np.random.default_rng(seed)
        if index%2==0:
            request=dict(decisionAgent=None,targetSpeed=float(rng.uniform(0,40)),
                         randomTargetSpawn=True)
            group='defaults'
        else:
            request=sample_scenario(rng)
            group='randomized'
        if scenario is not None:
            request.update(scenario)
            group='condition'
        request.update(decisionAgent=None,decisionEnabled=[False,False])
        row=dict(seed=seed,group=group,settings=request)
        for name in ('baseline','ppo'):
            dual.round=seed
            dual.reset(request)
            if name=='ppo': dual.shooters[0].decider=policy
            for _ in range(100):
                result=dual.step()
                if result['done']: break
            log=dual.shooters[0].strategy_log
            if index < args.replays:
                from unity_bridge.replay import save_replay
                save_replay(dual,Path(args.output).parent/'replays'/f'{index:04d}-{name}.html',request,f'{name} / seed {seed}')
            row[name]=dict(scoreA=dual.combat.scores[0],scoreB=dual.combat.scores[1],
                           done=result['done'],timeMs=result['timeMs'],
                           candidates=len(log),rejected=sum(not d['approved'] for d in log),
                           criticalA=sum(e['agent']==0 and e['critical'] for e in dual.combat.events))
        rows.append(row)
        print(f"Pair {index+1}/{args.rounds}: A score {row['baseline']['scoreA']} -> {row['ppo']['scoreA']}",flush=True)
    summary={}
    for group in ('all','defaults','randomized','condition'):
        selected=[r for r in rows if group=='all' or r['group']==group]
        if not selected: continue
        differences=np.array([r['ppo']['scoreA']-r['baseline']['scoreA'] for r in selected])
        summary[group]=dict(rounds=len(selected),
            baselineMeanScoreA=float(np.mean([r['baseline']['scoreA'] for r in selected])),
            ppoMeanScoreA=float(np.mean([r['ppo']['scoreA'] for r in selected])),
            pairedDifference=float(differences.mean()),
            pairedStandardError=float(differences.std(ddof=1)/np.sqrt(len(selected))) if len(selected)>1 else None,
            ppoRejections=sum(r['ppo']['rejected'] for r in selected),
            baselineTimeouts=sum(not r['baseline']['done'] for r in selected),
            ppoTimeouts=sum(not r['ppo']['done'] for r in selected))
    report=dict(model=args.model,summary=summary,rounds=rows,
                note='Small held-out paired evaluation; not a guarantee across scenarios or training seeds.')
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
