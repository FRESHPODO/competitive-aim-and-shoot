"""Paired-seed frozen-policy reward-input comparisons, with native replays."""
import argparse
import json
import statistics
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from unity_bridge.server import DualSession
from unity_bridge.fire_decider import FireDecider
from unity_bridge.training_settings import sample
from unity_bridge.competitive_rewards import from_kill_events, normalize
from unity_bridge.replay import save_replay
from unity_bridge.storage import ROOT, MATCH_ROOT, MODEL_COPIES_ROOT


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def run_matches(document, model, run, rounds, label):
    started=time.monotonic(); torch.set_num_threads(1)
    policies=[FireDecider(model/f'agent_{s}.zip') for s in 'ab']
    sources=[]
    for side in 'ab':
        source=model/f'agent_{side}.zip';shutil.copyfile(source,run/source.name)
        sources.append(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    manifest=dict(status='running',rounds=rounds,seed=421,timeoutSeconds=30,inactivityTimeoutSeconds=15,resample=True,
                  models=sources,parameters=document['parameters'],rewardConfig=normalize(document['rewardConfig']),
                  playerPresets=document['playerPresets'])
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]; traces=[]; dual=DualSession(recording='replay')
    for index in range(rounds):
        seed=421+index;request=sample(document['parameters'],np.random.default_rng(seed))
        request.update(rewardValues={s:dict(document['rewardConfig'][s],draw=0.) for s in 'AB'},
                       playerPresets=document['playerPresets'])
        dual.round=seed;dual.reset(request)
        decisions=[[],[]]
        class Gate:
            def __init__(self,i):self.i=i
            def decide(self,now,context):
                answer=policies[self.i].decide(now,context)
                decisions[self.i].append(dict(timeMs=now,**answer))
                return answer
        for i,shooter in enumerate(dual.shooters):shooter.decider=Gate(i);shooter.decision_mode='bump'
        while dual.time<30000 and dual.combat.hp>0 and not dual.draw:dual.step(deadline_ms=30000)
        decisions=[[d for d in stream if d['timeMs']<=dual.time] for stream in decisions]
        events=dual.combat.events;killers={e['agent'] for e in events if e.get('killed')}
        winner='AB'[next(iter(killers))] if len(killers)==1 else None
        path=run/'replays'/f'round-{index+1:05d}.json'
        save_replay(dual,path,request,f'{label} / seed {seed}',html=False)
        row=dict(round=index+1,seed=seed,winner=winner,outcome='kill' if winner else 'time_limit_draw',
                 timeMs=dual.time,remainingHp=dual.combat.hp,
                 shots=[sum(e['agent']==i for e in events) for i in range(2)],
                 hits=[sum(e['agent']==i and e['hit'] for e in events) for i in range(2)],
                 damage=[sum(e['damage'] for e in events if e['agent']==i) for i in range(2)],
                 blocked=[sum(not d['approved'] for d in stream if d['timeMs']<=dual.time) for stream in decisions],
                 fireDecisions=[sum(d['approved'] for d in stream if d['timeMs']<=dual.time) for stream in decisions],
                 rewards=from_kill_events(events,request['rewardValues']),settings=request,
                 replay=str(path))
        row['outcome']='kill' if winner else 'shared_kill' if killers else 'no_shots_draw' if dual.draw else 'time_limit_draw'
        rows.append(row); traces.append(dict(events=events,decisions=decisions))
        with (run/'rounds.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        count=len(rows);wins=[sum(r['winner']==s for r in rows) for s in 'AB'];draws=count-sum(wins)
        summary=dict(status='completed' if count==rounds else 'running',completed=count,requested=rounds,
                     winsA=wins[0],winsB=wins[1],draws=draws,timeouts=sum(r['outcome']=='time_limit_draw' for r in rows),
                     winRateA=wins[0]/count,winRateB=wins[1]/count,drawRate=draws/count,
                     elapsedSeconds=time.monotonic()-started)
        for filename in ('summary.json','progress.json'):(run/filename).write_text(json.dumps(summary),encoding='utf-8')
    manifest['status']='completed';(run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return rows,traces,summary


def probe_same_context(output, rounds=20):
    """Change only the evaluated agent's reward inputs on identical baseline states."""
    torch.set_num_threads(1);report=read(output/'comparison.json');results=[]
    for result in report['results']:
        if result['variant']!='baseline':continue
        run=Path(result['directory']);document=read(run/'scenario-input.json')
        policies=[FireDecider(run/f'agent_{s}.zip') for s in 'ab'];dual=DualSession(recording=False)
        stats={s:{v:dict(count=0,flips=0,absoluteProbabilityChange=0.,signedProbabilityChange=0.)
                  for v in ('win5','loss5')} for s in 'AB'}
        expected=[json.loads(line) for line in (run/'rounds.jsonl').read_text().splitlines()]
        for index in range(min(rounds,len(expected))):
            seed=421+index;request=sample(document['parameters'],np.random.default_rng(seed))
            request['rewardValues']={s:dict(win=1.,loss=1.,draw=0.) for s in 'AB'}
            dual.round=seed;dual.reset(request)
            class Gate:
                def __init__(self,i):self.i=i
                def decide(self,now,context):
                    baseline=policies[self.i].decide(now,context)
                    for variant,key in (('win5','win'),('loss5','loss')):
                        changed=dict(context,own_reward=dict(context['own_reward'],**{key:5.}))
                        alternative=policies[self.i].decide(now,changed)
                        s=stats['AB'[self.i]][variant];s['count']+=1
                        s['flips']+=baseline['approved']!=alternative['approved']
                        delta=alternative['approveProbability']-baseline['approveProbability']
                        s['absoluteProbabilityChange']+=abs(delta);s['signedProbabilityChange']+=delta
                    return baseline
            for i,s in enumerate(dual.shooters):s.decider=Gate(i);s.decision_mode='bump'
            while dual.time<30000 and dual.combat.hp>0 and not dual.draw:dual.step(deadline_ms=30000)
            row=expected[index]
            assert abs(dual.time-row['timeMs'])<1e-6
            assert [sum(e['agent']==i for e in dual.combat.events) for i in range(2)]==row['shots']
        for conditions in stats.values():
            for s in conditions.values():
                n=s['count'];s['flipRate']=s['flips']/n
                for key in ('absoluteProbabilityChange','signedProbabilityChange'):s[key]/=n
        results.append(dict(model=result['model'],rounds=min(rounds,len(expected)),stats=stats,
                            baselineReproducible=True))
        (output/'same-context-probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print('PROBE',result['model'],stats,flush=True)
    return results


def write_report(output):
    report=read(output/'comparison.json');probes=read(output/'same-context-probe.json')
    labels={'20261101_pro_pro_150000':'프로–프로 15만 경기',
            '20261101_ama_ama_120000':'아마추어–아마추어 12만 경기',
            '20261101_pro_ama_200000':'프로–아마추어 20만 경기'}
    profiles=[(1,1),(5,1),(1,5)]
    lines=['# 보상 입력 교차 비교', '',
           f'세 모델 쌍 × A 보상 3종 × B 보상 3종 × {report["rounds"]}경기. 모델 가중치는 고정했다.',
           '보상은 (내 처치 보상, 상대 처치 시 비용)이며, 모든 조합에 같은 시드 421부터의 경기 조건을 사용했다.',
           '공통 종료 조건은 경기 30초 제한과 15초 무격발 종료다. 어느 무승부에도 보상/비용은 0이다.',
           '각 모델 쌍은 자신의 학습 설정을 사용했다. 서로 다른 모델 쌍 사이의 성능 비교 실험은 아니다.',
           '상대 조건을 고정한 열 안에서 내 보상 조건을 비교해야 한다. 상대도 상태 변화에 반응하는 것은 대결의 일부다.',
           '아래 승률은 무승부를 포함한 전체 경기 기준이다. 50경기 표본으로 최적 전략이나 일반적인 우열을 확정하지 않는다.', '']
    for model,label in labels.items():
        rows=[r for r in report['results'] if r['model']==model]
        grid={tuple(r['rewards']):r for r in rows}
        lines += ['## '+label,'','A 승률. 행은 A 보상, 열은 B 보상.','',
                  '| A \\ B | 1/1 | 5/1 | 1/5 |','|---|---:|---:|---:|']
        for a in profiles:
            lines.append('| '+f'{a[0]}/{a[1]}'+' | '+' | '.join(f'{grid[a+b]["wins"][0]/report["rounds"]:.0%}' for b in profiles)+' |')
        lines += ['', 'A의 보상을 1/1에서 바꿨을 때, 상대 조건별 A 승률 변화:', '',
                  '| A 변경 | B 1/1 | B 5/1 | B 1/5 |','|---|---:|---:|---:|']
        for a in profiles[1:]:
            lines.append(f'| {a[0]}/{a[1]} | '+' | '.join(f'{100*(grid[a+b]["wins"][0]-grid[(1,1)+b]["wins"][0])/report["rounds"]:+.0f}%p' for b in profiles)+' |')
        lines += ['', 'B 승률·무승부·발사 횟수·첫 발사 시간·행동 변화 수치는 comparison.json에 포함했다.', '']
    lines += ['## 동일 관측에서 보상 입력만 변경', '',
              '모델별 기본 보상 20경기를 진행하면서, 실제 상태를 바꾸지 않고 자기 보상 입력만 바꿔 PPO를 추가 평가했다.',
              '아래는 승인/보류 판단이 뒤집힌 비율이다. 메인 발사 요청이 없는 시점과 쿨다운 중 판단도 포함한다.', '',
              '| 모델 | 에이전트 | 승리 보상 1→5 | 패배 비용 1→5 |','|---|---|---:|---:|']
    for item in probes:
        for side,stats in item['stats'].items():
            lines.append(f'| {labels[item["model"]]} | {side} | {stats["win5"]["flipRate"]:.1%} | {stats["loss5"]["flipRate"]:.1%} |')
    lines += ['', '모든 경기의 3D 행동 기록은 각 조건의 replays 폴더에 직접 저장했다. 재생 시 모델을 다시 실행하지 않는다.',
              '보상 입력이 행동을 바꾼다는 사실과, 보상 의미를 일관되게 학습했다는 판단은 구별해야 한다.']
    (output/'결과.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=50)
    args = parser.parse_args()
    output = MATCH_ROOT / ('reward-comparison-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True)
    pairs = [
        ('20261101_pro_pro_150000', '20261031-141040-629041-dd149c29'),
        ('20261101_ama_ama_120000', '20261031-141109-442191-17cd79ff'),
        ('20261101_pro_ama_200000', '20261031-141140-323458-c4f440dc'),
    ]
    variants = {'baseline': (1, 1, 1, 1), 'A_win5': (5, 1, 1, 1),
                'A_loss5': (1, 5, 1, 1), 'B_win5': (1, 1, 5, 1),
                'B_loss5': (1, 1, 1, 5), 'A_win5_B_win5': (5,1,5,1),
                'A_win5_B_loss5': (5,1,1,5), 'A_loss5_B_win5': (1,5,5,1),
                'A_loss5_B_loss5': (1,5,1,5)}
    report = dict(rounds=args.rounds, seed=421, timeout=30, results=[])
    for name, training in pairs:
        metadata = read(ROOT / 'training-results' / training / 'training.json')
        settings = read(ROOT / 'training-results' / training / 'settings.json')
        baseline_rows = baseline_replays = None
        for variant, rewards in variants.items():
            run = output / name / variant
            run.mkdir(parents=True)
            document = dict(parameters=settings, playerPresets=metadata.get('playerPresets', {}),
                            rewardConfig={s: dict(win=rewards[i*2], loss=rewards[i*2+1])
                                          for i, s in enumerate('AB')})
            scenario = run / 'scenario-input.json'
            scenario.write_text(json.dumps(document, indent=2), encoding='utf-8')
            model = MODEL_COPIES_ROOT / name
            rows,replays,summary=run_matches(document,model,run,args.rounds,f'{name} / {variant}')
            if variant == 'baseline':
                baseline_rows, baseline_replays = rows, replays
            result = dict(model=name, variant=variant, rewards=rewards, directory=str(run),
                          wins=[summary['winsA'], summary['winsB']], draws=summary['draws'],
                          winnerChanged=sum(a['winner'] != b['winner'] for a,b in zip(rows,baseline_rows)),
                          shotSequenceChanged=[0,0], decisionChanged=[0,0], commonDecisions=[0,0],
                          meanShots=[], meanFirstShotMs=[], ppoApprovalRate=[])
            for i in range(2):
                first=[]; approved=requested=0
                for replay, base in zip(replays, baseline_replays):
                    shots=[e['timeMs'] for e in replay['events'] if e['agent']==i]
                    old=[e['timeMs'] for e in base['events'] if e['agent']==i]
                    result['shotSequenceChanged'][i] += shots != old
                    if shots:first.append(shots[0])
                    old_decisions={d['timeMs']:d['approved'] for d in base['decisions'][i]}
                    for d in replay['decisions'][i]:
                        if d['timeMs'] in old_decisions:
                            result['commonDecisions'][i]+=1
                            result['decisionChanged'][i]+=d['approved'] != old_decisions[d['timeMs']]
                        requested+=1; approved+=bool(d['approved'])
                result['meanShots'].append(statistics.mean(r['shots'][i] for r in rows))
                result['meanFirstShotMs'].append(statistics.mean(first) if first else None)
                result['ppoApprovalRate'].append(approved/requested if requested else None)
            # Later decisions may differ because earlier decisions changed the game.
            result['decisionComparisonNote']='Paired trajectories; downstream state changes included.'
            report['results'].append(result)
            (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('COMPARISON',name,variant,result['wins'],result['shotSequenceChanged'],flush=True)
    probe_same_context(output)
    write_report(output)
    print('OUTPUT',output,flush=True)


if __name__ == '__main__':
    main()
