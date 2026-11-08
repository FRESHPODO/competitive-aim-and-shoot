"""Matched-seed 5-condition x PPO on/off experiments on the real simulator."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from unity_bridge.atomic_io import replace_with_retry

CASES={'neutral':(1.,1.),'a_worse':(1.5,1.),'b_better':(1.,.5),
       'a_better':(.5,1.),'b_worse':(1.,1.5)}
LABELS={'neutral':'동일 중립','a_worse':'A 불리','b_better':'B 유리',
        'a_better':'A 유리','b_worse':'B 불리'}


def initialize(model):
    import torch
    torch.set_num_threads(1)
    from unity_bridge.server import DualSession
    from unity_bridge.fire_decider import FireDecider
    global DUAL, POLICY
    DUAL=DualSession()
    POLICY=FireDecider(model)


def profiles(case):
    from unity_bridge.server import DEFAULTS
    result=[]
    for scale in CASES[case]:
        profile=DEFAULTS.copy()
        profile[:4]=[value*scale for value in profile[:4]]
        result.append(profile)
    return result


def run_batch(job):
    case,mode,indices,base_seed,max_ms=job
    rows=[]
    for index in indices:
        seed=base_seed+index
        DUAL.round=seed
        DUAL.reset(dict(decisionAgent=None,profiles=profiles(case),targetHp=1000,
                        minDamage=100,maxDamage=300,criticalRadius=.01,targetRadius=5,
                        randomTargetSpeed=True,targetSpeedMin=0,targetSpeedMax=40,
                        randomTargetSpawn=True,shotIntervalsMs=[150,150],shotSpreadDeg=[0,0]))
        if mode=='ppo': DUAL.shooters[0].decider=POLICY
        while DUAL.time<max_ms and DUAL.combat.hp>0 and not DUAL.draw:
            DUAL.step()
        events=DUAL.combat.events
        scores=DUAL.combat.scores
        done=DUAL.combat.hp<=0
        row=dict(case=case,mode=mode,index=index,seed=seed,completed=done,
                 time_ms=DUAL.time,target_speed=DUAL.motion.speed,
                 target_initial=DUAL.motion.initial.tolist(),target_axis=DUAL.motion.axis.tolist(),
                 score_a=scores[0],score_b=scores[1],
                 win_a=int(done and scores[0]>scores[1]),win_b=int(done and scores[1]>scores[0]),
                 shared_kill=int(done and scores[0]==scores[1]),timeout=int(not done))
        for agent,side in enumerate(('a','b')):
            shots=[e for e in events if e['agent']==agent]
            row['shots_'+side]=len(shots)
            row['damage_'+side]=sum(e['damage'] for e in shots)
            row['hits_'+side]=sum(e['hit'] for e in shots)
            row['critical_'+side]=sum(e['critical'] for e in shots)
        row['a_candidates']=len(DUAL.shooters[0].strategy_log)
        row['a_rejected']=sum(not d['approved'] for d in DUAL.shooters[0].strategy_log)
        rows.append(row)
    return rows


def write_json(path,data):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    replace_with_retry(temporary,path)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows,output):
    summaries=[]
    for case in CASES:
        for mode in ('baseline','ppo'):
            group=[r for r in rows if r['case']==case and r['mode']==mode]
            for side in ('a','b'):
                summaries.append(dict(case=case,label=LABELS[case],mode=mode,agent=side.upper(),
                    n=len(group),completed=sum(r['completed'] for r in group),
                    final_score=sum(r['score_'+side] for r in group),
                    win_rate=np.mean([r['win_'+side] for r in group]),
                    score_rate=np.mean([r['score_'+side] for r in group]),
                    mean_shots=np.mean([r['shots_'+side] for r in group]),
                    sd_shots=np.std([r['shots_'+side] for r in group],ddof=1),
                    mean_damage=np.mean([r['damage_'+side] for r in group]),
                    sd_damage=np.std([r['damage_'+side] for r in group],ddof=1),
                    shared_kills=sum(r['shared_kill'] for r in group),timeouts=sum(r['timeout'] for r in group)))
    write_csv(output/'summary.csv',summaries)
    comparisons=[]
    rng=np.random.default_rng(60021)
    for case in CASES:
        baseline=sorted((r for r in rows if r['case']==case and r['mode']=='baseline'),key=lambda r:r['index'])
        ppo=sorted((r for r in rows if r['case']==case and r['mode']=='ppo'),key=lambda r:r['index'])
        assert [r['seed'] for r in baseline]==[r['seed'] for r in ppo]
        assert all(a['target_initial']==b['target_initial'] and a['target_axis']==b['target_axis'] and
                   a['target_speed']==b['target_speed'] for a,b in zip(baseline,ppo))
        for side in ('a','b'):
            for metric in ('score','win','shots','damage'):
                delta=np.array([b[metric+'_'+side]-a[metric+'_'+side] for a,b in zip(baseline,ppo)])
                means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
                comparisons.append(dict(case=case,agent=side.upper(),metric=metric,
                    ppo_minus_baseline=float(delta.mean()),
                    paired_bootstrap_low=float(np.quantile(means,.025)),
                    paired_bootstrap_high=float(np.quantile(means,.975))))
    write_csv(output/'paired_comparison.csv',comparisons)
    write_json(output/'summary.json',dict(summary=summaries,paired_comparison=comparisons))
    metadata=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((output/'model.zip').read_bytes()).hexdigest()==metadata['sha256']
    for case in CASES:
        for mode in ('baseline','ppo'):
            group=[r for r in rows if r['case']==case and r['mode']==mode]
            assert len(group)==metadata['rounds_per_cell']
            assert sum(r['win_a']+r['win_b']+r['shared_kill']+r['timeout'] for r in group)==len(group)
            assert np.isclose(sum(r['score_a']+r['score_b'] for r in group),sum(r['completed'] for r in group))
    lines=['# A/B Noise 조건별 반복 실험', '',
        f"각 조건·모드당 {metadata['rounds_per_cell']}회, 총 {len(rows)}회. PPO는 A에만 적용하고 비교군은 양측 모두 기존 Aim-and-Shoot를 사용했습니다.", '',
        '## 결과', '',
        '수치는 A / B 순서입니다. 승률은 단독 처치 수 / 전체 시행 수이며 동시 처치와 시간 초과는 별도 기록합니다. 평균은 타겟 한 개(시행 한 번) 기준입니다.', '',
        '| 조건 | PPO(A) | 최종 점수 A/B | 승률 A/B | 평균 발사 수 A/B | 평균 누적 피해 A/B |',
        '|---|---|---:|---:|---:|---:|']
    for case in CASES:
        for mode in ('baseline','ppo'):
            a=next(s for s in summaries if s['case']==case and s['mode']==mode and s['agent']=='A')
            b=next(s for s in summaries if s['case']==case and s['mode']==mode and s['agent']=='B')
            lines.append(f"| {LABELS[case]} | {'적용' if mode=='ppo' else '미적용'} | {a['final_score']:g} / {b['final_score']:g} | {a['win_rate']:.1%} / {b['win_rate']:.1%} | {a['mean_shots']:.3f} / {b['mean_shots']:.3f} | {a['mean_damage']:.2f} / {b['mean_damage']:.2f} |")
    lines+=['','## PPO 적용으로 인한 A 변화','','승률 차이는 %p, 나머지는 원래 단위입니다. 구간은 동일 seed의 차이에 대한 10,000회 paired bootstrap 95% 구간입니다. 단일 학습 seed 모델과 아래 환경에 한정된 결과입니다.','',
            '| 조건 | A 승률 변화 (%p, 95% 구간) | A 평균 발사 수 변화 | A 평균 피해 변화 |',
            '|---|---:|---:|---:|']
    for case in CASES:
        values={r['metric']:r for r in comparisons if r['case']==case and r['agent']=='A'}
        win=values['win']
        lines.append(f"| {LABELS[case]} | {win['ppo_minus_baseline']*100:+.1f} [{win['paired_bootstrap_low']*100:+.1f}, {win['paired_bootstrap_high']*100:+.1f}] | {values['shots']['ppo_minus_baseline']:+.3f} | {values['damage']['ppo_minus_baseline']:+.2f} |")
    lines+=['','## 설정과 해석 범위','',
        '- 중립 Noise: motor=0.20, position=0.08, speed=0.04, clock=0.04. 불리는 해당 에이전트만 네 값 모두 1.5배, 유리는 해당 에이전트만 모두 0.5배입니다. 상대는 중립을 유지합니다.',
        '- A 불리(0.30, 0.12, 0.06, 0.06)와 B 유리(0.10, 0.04, 0.02, 0.02)는 변경한 에이전트가 다릅니다. 이 정의로 다섯 조건을 구분했습니다.',
        '- HP=1000, 반지름=5°, 최소/최대 피해=100/300, 즉사 반지름=타겟의 1%, 양측 탄퍼짐=0°, 쿨다운=150ms. 표적 속도는 시행마다 0–40°/s 균등 추출, 시작 위치와 방향도 무작위입니다.',
        f"- seed {metadata['base_seed']}–{metadata['base_seed']+metadata['rounds_per_cell']-1}을 모든 조건과 두 모드에서 공통으로 사용했습니다. 양측 사격 난수는 서로 다르며, 각 모드 간 동일 에이전트 seed는 같습니다. PPO 거부는 이후 운동 경로와 난수 소비를 바꿀 수 있습니다.",
        '- 누적 피해는 기존 UI와 같은 판정 피해 합계로, 남은 체력을 초과한 피해(overkill)를 포함합니다. 이 때문에 양측 합이 항상 1000은 아닙니다.',
        '- 시간 초과 기준은 시뮬레이션 30초입니다. 미완료 시행을 버리지 않고 평균과 승률 분모에 포함합니다.',
        '- 여러 Noise를 함께 조정했으므로 개별 Noise의 인과 효과를 분리한 실험은 아닙니다. 전체 설정·조건 조합으로 일반화하거나 여러 PPO 학습 seed의 성능으로 해석할 수 없습니다.',
        '- 현재 시뮬레이터의 100ms 피해 정산·관측 스냅샷 구조를 유지했습니다. 정책 파라미터 및 원본 사격 모델은 실험 중 고정했습니다.',
        f"- PPO 체크포인트 SHA-256: `{metadata['sha256']}`",'',
        '## 동시 처치·시간 초과','']
    for case in CASES:
        for mode in ('baseline','ppo'):
            s=next(s for s in summaries if s['case']==case and s['mode']==mode)
            lines.append(f"- {LABELS[case]} / {mode}: 완료 {s['completed']}, 동시 처치 {s['shared_kills']}, 시간 초과 {s['timeouts']}.")
    lines+=['','## 원본 데이터','']
    for name in ('trials.csv','summary.csv','paired_comparison.csv','metadata.json'):
        lines.append(f'- [{name}]({(output/name).as_posix()})')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--rounds',type=int,default=200)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=210000)
    parser.add_argument('--max-ms',type=float,default=30000.)
    args=parser.parse_args()
    if args.rounds<200: parser.error('At least 200 trials per condition required')
    output=Path(args.output).absolute()
    output.mkdir(parents=True,exist_ok=True)
    (output/'batches').mkdir(exist_ok=True)
    model=Path(args.model).absolute()
    frozen=output/'model.zip'
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    metadata=dict(model=str(model),sha256=digest,rounds_per_cell=args.rounds,base_seed=args.seed,
                  max_ms=args.max_ms,cases={case:profiles(case) for case in CASES},
                  fixed=dict(hp=1000,radius_deg=5,min_damage=100,max_damage=300,critical_radius=.01,
                             cooldown_ms=[150,150],spread_deg=[0,0],speed_uniform=[0,40]),
                  modes=dict(baseline='Both original Aim-and-Shoot; no PPO',ppo='A PPO, B original'),
                  definitions=dict(win='strict score greater than opponent; denominator all trials',
                      damage='sum of authoritative damage, including overkill, per target',
                      score='sum of per-target awarded points; simultaneous kill can split',
                      timeout='no kill by 30s simulation; retained in every mean and denominator'))
    for source in ('unity_bridge/server.py','unity_bridge/fire_decider.py','src/agent/vplayer.py',
                   'data/ans_agent_models/ijhcs_default/checkpoints/rl_model_20000000_steps.zip'):
        metadata.setdefault('source_sha256',{})[source]=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if (output/'metadata.json').exists():
        if json.loads((output/'metadata.json').read_text(encoding='utf-8'))!=metadata:
            raise ValueError('Existing output belongs to a different experiment')
    else:
        shutil.copyfile(model,frozen)
        write_json(output/'metadata.json',metadata)
    jobs=[]
    rows=[]
    for case in CASES:
        for mode in ('baseline','ppo'):
            for start in range(0,args.rounds,20):
                path=output/'batches'/f'{case}-{mode}-{start:04d}.json'
                if path.exists(): rows.extend(json.loads(path.read_text(encoding='utf-8')))
                else: jobs.append(((case,mode,list(range(start,min(start+20,args.rounds))),args.seed,args.max_ms),path))
    started=time.monotonic()
    def progress(status):
        write_json(output/'progress.json',dict(status=status,completed=len(rows),total=args.rounds*10,
            elapsed_seconds=time.monotonic()-started,counts={f'{case}/{mode}':sum(r['case']==case and r['mode']==mode for r in rows)
                 for case in CASES for mode in ('baseline','ppo')}))
    progress('running')
    try:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(frozen),)) as pool:
            pending={pool.submit(run_batch,job):path for job,path in jobs}
            for future in as_completed(pending):
                batch=future.result()
                write_json(pending[future],batch)
                rows.extend(batch)
                progress('running')
                print(f'{len(rows)}/{args.rounds*10} completed | {batch[0]["case"]}/{batch[0]["mode"]}',flush=True)
        rows.sort(key=lambda r:(list(CASES).index(r['case']),r['mode'],r['index']))
        assert len(rows)==args.rounds*10
        write_csv(output/'trials.csv',rows)
        summary=summarize(rows,output)
        progress('completed')
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except BaseException:
        progress('failed')
        raise


if __name__=='__main__': main()
