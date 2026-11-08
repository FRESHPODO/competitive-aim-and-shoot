"""Preserve three recent generations and record labelled deterministic demo matches."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from unity_bridge.storage import ROOT, EVALUATION_MODEL_ROOT, model_lock, resolve_path
from unity_bridge.trainer_state import load_state, save_state
from unity_bridge.player_presets import apply, provenance
from unity_bridge.training_settings import sample
from unity_bridge.competitive_rewards import values_from_alpha, from_kill_events
from unity_bridge.fire_decider import FireDecider
from unity_bridge.server import DualSession
from unity_bridge.replay import save_replay


def main():
    torch.set_num_threads(1)
    runs = [p.parent for p in (ROOT / 'generational-results').glob('*/training.json')]
    run = max(runs, key=lambda p: (p / 'progress.json').stat().st_mtime_ns)
    meta = load_state(run / 'training.json')
    config = meta['config']
    model_root = resolve_path(meta['modelDirectory'])
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S-%f')
    preserved = EVALUATION_MODEL_ROOT / stamp
    output = ROOT / 'match-results' / ('recent-generations-' + stamp)
    preserved.mkdir(parents=True)
    output.mkdir(parents=True)
    extracted = []
    with model_lock(model_root):
        entries = sorted(load_state(model_root / 'opponents.json') or [], key=lambda e: e['generation'])[-3:]
        if len(entries) != 3:
            raise ValueError('Three completed generations are not available')
        for entry in entries:
            source = model_root / entry['relativePath']
            target = preserved / f'generation-{entry["generation"]:03d}.zip'
            shutil.copyfile(source, target)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256']
            extracted.append(dict(entry, preservedModel=str(target)))
    save_state(preserved / 'extraction.json', dict(trainingRun=str(run), models=extracted))
    policies = {e['generation']: FireDecider(e['preservedModel']) for e in extracted}
    pairs = [(extracted[2], extracted[1]), (extracted[2], extracted[0]), (extracted[1], extracted[0])]
    cases = [(alpha, pa, pb, relation) for alpha in (.9, .1) for pa, pb, relation in (
        ('professional_mean', 'professional_mean', '프로 / 프로'),
        ('professional_mean', 'amateur_mean', '프로 / 아마추어'),
        ('amateur_mean', 'professional_mean', '아마추어 / 프로'))]
    playlist = []
    pair_runs = []
    dual = DualSession(recording='replay')
    for ea, eb in pairs:
        ga, gb = ea['generation'], eb['generation']
        folder = output / f'g{ga:03d}-vs-g{gb:03d}'
        folder.mkdir()
        for side, entry in [('a', ea), ('b', eb)]:
            shutil.copyfile(entry['preservedModel'], folder / f'agent_{side}.zip')
        rows = []
        began = time.monotonic()
        first_settings = first_presets = first_rewards = None
        for index, (alpha, pa, pb, relation) in enumerate(cases, 1):
            settings = apply(apply(config['settings'], 'A', pa), 'B', pb)
            values = dict(A=values_from_alpha(alpha), B=values_from_alpha(.5))
            presets = dict(A=provenance(pa), B=provenance(pb))
            seed = 885100 + index
            request = sample(settings, np.random.default_rng(seed))
            request.update(rewardValues=values, playerPresets=presets)
            dual.round = seed
            dual.reset(request)
            dual.inactivity_limit_ms = config['timeout'] * 1000
            for i, generation in enumerate((ga, gb)):
                dual.shooters[i].decider = policies[generation]
                dual.shooters[i].decision_mode = 'bump'
            while dual.time < config['timeout'] * 1000 and dual.combat.hp > 0 and not dual.draw:
                dual.step(deadline_ms=config['timeout'] * 1000)
            label = f'A {ga}세대 (α {alpha:g}) / B {gb}세대 (α 0.5) · {relation}'
            path = folder / 'replays' / f'round-{index:05d}.json'
            save_replay(dual, path, request, label, html=False)
            replay = load_state(path)
            replay['modelGenerations'] = [ga, gb]
            replay['modelHashes'] = [ea['sha256'], eb['sha256']]
            save_state(path, replay)
            playlist.append(str(path))
            killers = {e['agent'] for e in dual.combat.events if e.get('killed')}
            winner = 'AB'[next(iter(killers))] if len(killers) == 1 else None
            row = dict(round=index, seed=seed, label=label, winner=winner,
                outcome='kill' if winner else 'shared_kill' if killers else 'time_limit_draw',
                timeMs=dual.time, remainingHp=dual.combat.hp,
                shots=[sum(e['agent'] == i for e in dual.combat.events) for i in range(2)],
                hits=[sum(e['agent'] == i and e['hit'] for e in dual.combat.events) for i in range(2)],
                damage=[sum(e['damage'] for e in dual.combat.events if e['agent'] == i) for i in range(2)],
                blocked=[sum(not d['approved'] for d in s.strategy_log) for s in dual.shooters],
                rewards=from_kill_events(dual.combat.events, values), settings=request, replay=str(path))
            rows.append(row)
            with (folder / 'rounds.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
            if first_settings is None:
                first_settings, first_presets = settings, presets
                first_rewards = {s: dict(alpha=dict(mode='fixed', value=values[s]['alpha'], min=0., max=1., step=.05)) for s in 'AB'}
            print(label, 'winner', winner or 'draw', flush=True)
        manifest = dict(status='completed', rounds=len(rows), seed=885101, timeoutSeconds=config['timeout'],
            resample=False, parameters=first_settings, playerPresets=first_presets, rewardConfig=first_rewards,
            models=[dict(source=e['preservedModel'], sha256=e['sha256'], generation=e['generation']) for e in (ea, eb)],
            demonstration=True, varyingConditionsPerRound=True)
        summary = dict(status='completed', completed=len(rows), requested=len(rows),
            winsA=sum(r['winner'] == 'A' for r in rows), winsB=sum(r['winner'] == 'B' for r in rows),
            draws=sum(r['winner'] is None for r in rows), elapsedSeconds=time.monotonic() - began, seed=885101)
        save_state(folder / 'manifest.json', manifest)
        save_state(folder / 'progress.json', summary)
        save_state(folder / 'summary.json', summary)
        pair_runs.append(str(folder))
    save_state(output / 'playlist.json', playlist)
    result = dict(trainingRun=str(run), preservedModels=str(preserved), generations=[e['generation'] for e in extracted],
                  pairRuns=pair_runs, playlist=str(output / 'playlist.json'), totalReplays=len(playlist),
                  conditions='A alpha .9/.1; B alpha .5; professional/professional, professional/amateur, amateur/professional; both policies deterministic')
    save_state(output / 'preview.json', result)
    save_state(ROOT / 'storage-reports/latest-generation-preview.json', result)
    print('READY', output, flush=True)


if __name__ == '__main__':
    main()
