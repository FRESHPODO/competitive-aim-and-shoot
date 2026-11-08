"""Balanced, frozen-policy generation evaluation; compact summaries, no replay recording."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import load_state, save_state
from unity_bridge.player_presets import apply
from unity_bridge.training_settings import sample
from unity_bridge.competitive_rewards import values_from_alpha
from unity_bridge.fire_decider import FireDecider
from unity_bridge.server import DualSession

_dual = _policies = _config = None


def initialize(models, config):
    global _dual, _policies, _config
    os.environ['ANS_TORCH_DEVICE'] = 'cpu'
    torch.set_num_threads(1)
    _config = config
    _policies = {int(g): FireDecider(path) for g, path in models.items()}
    for decider in _policies.values():
        decider.model.policy.requires_grad_(False)
    _dual = DualSession(recording=False)


def play(job):
    ga, gb, pa, pb, seed, pair = job
    settings = apply(apply(_config['settings'], 'A', pa), 'B', pb)
    request = sample(settings, np.random.default_rng(seed))
    request['rewardValues'] = {s: values_from_alpha(.5) for s in 'AB'}
    _dual.round = seed
    _dual.reset(request)
    _dual.inactivity_limit_ms = _config['timeout'] * 1000
    for side, generation in enumerate((ga, gb)):
        _dual.shooters[side].decider = _policies[generation]
        _dual.shooters[side].decision_mode = 'bump'
    while _dual.time < _config['timeout'] * 1000 and _dual.combat.hp > 0 and not _dual.draw:
        _dual.step(deadline_ms=_config['timeout'] * 1000)
    killers = {e['agent'] for e in _dual.combat.events if e.get('killed')}
    winner = (ga, gb)[next(iter(killers))] if len(killers) == 1 else None
    return dict(pair=pair, generationA=ga, generationB=gb, presetA=pa, presetB=pb,
                seed=seed, winner=winner, timeMs=_dual.time,
                shots=[sum(e['agent'] == i for e in _dual.combat.events) for i in range(2)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=50, help='Seeds per preset combination and side assignment')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=Path, help='Finish the missing games in an interrupted evaluation folder')
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.workers <= 8:
        parser.error('Invalid repeats or workers')
    if args.resume:
        output = args.resume.resolve()
        manifest = load_state(output / 'manifest.json')
        if not manifest or not manifest.get('sideBalanced') or not manifest.get('deterministicBoth'):
            raise ValueError('Not a compatible generation evaluation')
        extraction = dict(models=manifest['models'])
        config = manifest['config']
        args.repeats = manifest['repeats']
    else:
        preview = load_state(ROOT / 'storage-reports/latest-generation-preview.json')
        extraction = load_state(Path(preview['preservedModels']) / 'extraction.json')
        config = load_state(Path(preview['trainingRun']) / 'config.json')
    models = {e['generation']: e['preservedModel'] for e in extraction['models']}
    generations = sorted(models)
    pairs = [(generations[2], generations[1]), (generations[2], generations[0]), (generations[1], generations[0])]
    if not args.resume:
        stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S')
        output = ROOT / 'match-results' / ('generation-evaluation-' + stamp)
        output.mkdir(parents=True, exist_ok=False)
    jobs = []
    presets = ['professional_mean', 'amateur_mean']
    for newer, older in pairs:
        for ai, pa in enumerate(presets):
            for bi, pb in enumerate(presets):
                for index in range(args.repeats):
                    seed = 832000 + (ai * 2 + bi) * args.repeats + index
                    jobs.append((newer, older, pa, pb, seed, f'{newer}/{older}'))
                    jobs.append((older, newer, pb, pa, seed, f'{newer}/{older}'))
    manifest = dict(models=extraction['models'], config=config, alphaA=.5, alphaB=.5,
                    deterministicBoth=True, sideBalanced=True, seedsSharedAcrossPairs=True,
                    repeats=args.repeats, requested=len(jobs), replayRecording=False)
    save_state(output / 'manifest.json', manifest)
    result = {f'{a}/{b}': dict(newer=a, older=b, n=0, winsNewer=0, winsOlder=0, draws=0) for a, b in pairs}
    by_generation = {g: dict(generation=g, n=0, wins=0, losses=0, draws=0) for g in generations}
    def tally(row):
        s = result[row['pair']]
        s['n'] += 1
        s['winsNewer' if row['winner'] == s['newer'] else 'winsOlder' if row['winner'] == s['older'] else 'draws'] += 1
        for generation in (row['generationA'], row['generationB']):
            g = by_generation[generation]
            g['n'] += 1
            g['draws' if row['winner'] is None else 'wins' if row['winner'] == generation else 'losses'] += 1
    def key(row):
        return (row['generationA'], row['generationB'], row['presetA'], row['presetB'], row['seed'], row['pair'])
    done = set()
    valid_end = 0
    if args.resume:
        with (output / 'rounds.jsonl').open('rb') as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                except ValueError:
                    break
                identity = key(row)
                if identity in done:
                    raise ValueError('Duplicate saved evaluation game')
                done.add(identity)
                tally(row)
                valid_end = existing.tell()
        with (output / 'rounds.jsonl').open('r+b') as existing:
            existing.truncate(valid_end)
    pending_jobs = [job for job in jobs if tuple(job) not in done]
    prior_elapsed = (load_state(output / 'summary.json') or {}).get('elapsedSeconds', 0.) if args.resume else 0.
    started = time.monotonic()
    def report(completed, status):
        data = dict(status=status, completed=completed, requested=len(jobs),
                    elapsedSeconds=prior_elapsed + time.monotonic() - started, results=result,
                    byGeneration=by_generation, output=str(output))
        save_state(output / 'summary.json', data)
        save_state(ROOT / 'storage-reports/recent-generation-evaluation.json', data)
    report(len(done), 'running')
    print(f'Resuming {len(done)} saved games; {len(pending_jobs)} remaining', flush=True)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(models, config)) as pool, (output / 'rounds.jsonl').open('a', encoding='utf-8') as log:
        futures = [pool.submit(play, job) for job in pending_jobs]
        for completed, future in enumerate(as_completed(futures), len(done) + 1):
            row = future.result()
            log.write(json.dumps(row) + '\n')
            tally(row)
            if completed % 50 == 0:
                log.flush()
                report(completed, 'running')
                print(f'{completed}/{len(jobs)} completed', flush=True)
    for s in result.values():
        n = s['n']
        s.update(winRateNewer=s['winsNewer']/n, winRateOlder=s['winsOlder']/n, drawRate=s['draws']/n,
                 scoreNewer=(s['winsNewer']+.5*s['draws'])/n, scoreOlder=(s['winsOlder']+.5*s['draws'])/n)
        assert n == args.repeats * 8 and n == s['winsNewer'] + s['winsOlder'] + s['draws']
    for s in by_generation.values():
        n = s['n']
        assert n == args.repeats * 16
        s.update(winRate=s['wins']/n, lossRate=s['losses']/n, drawRate=s['draws']/n,
                 meanScore=(s['wins']+.5*s['draws'])/n)
    report(len(jobs), 'completed')
    print(json.dumps(result), flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
