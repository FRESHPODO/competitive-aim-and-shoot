"""Reward edge cases, UI round-trip, actual training, generation rotation and resume."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from unity_bridge.reward_generation import rewards
from unity_bridge.reward_generation.config import defaults, validate, KIND
from unity_bridge.reward_generation.decider import FireDecider, OBS_SIZE
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import load_state, save_state


def main():
    values = {s: dict(alpha=.5, win=1., loss=1., damageFactor=1., criticalFactor=.75) for s in 'AB'}
    events = [dict(agent=0, timeMs=1, damage=700., killed=False, critical=False),
              dict(agent=1, timeMs=2, damage=600., killed=True, critical=True)]
    parts = rewards.breakdown(events, 1000., values)
    assert np.isclose(parts[0]['total'], -.3)
    assert np.isclose(parts[1]['total'], 2.05)
    assert parts[1]['actualDamage'] == 300.
    simultaneous = [dict(agent=i, timeMs=1, damage=1000., killed=True, critical=True) for i in range(2)]
    tied = rewards.breakdown(simultaneous, 1000., values)
    assert all(p['actualDamage'] == 500. and p['total'] == 1.25 for p in tied)
    nonfatal = rewards.breakdown([dict(agent=0, timeMs=1, damage=10., critical=True, killed=False)], 1000., values)
    assert nonfatal[0]['criticalReward'] == 0
    zero = deepcopy(values)
    for side in 'AB':
        zero[side].update(damageFactor=0., criticalFactor=0.)
    assert [p['total'] for p in rewards.breakdown(events, 1000., zero)] == [-1., 1.]
    # Independent bonus survives alpha=0.
    alpha_zero = deepcopy(values)
    alpha_zero['B'].update(alpha=0., win=0.)
    assert rewards.breakdown(events, 1000., alpha_zero)[1]['criticalReward'] == .75
    config = defaults()
    validate(config)
    invalid = deepcopy(config)
    invalid['rewardConfig']['A']['criticalFactor'].update(mode='fixed', value=1.1)
    try:
        validate(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Out-of-range critical coefficient accepted')
    original_settings = (ROOT / 'app-settings/generational-trainer.json').read_bytes()
    ROOT.joinpath('storage-reports').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='reward-generation-check-', dir=ROOT / 'storage-reports') as temp:
        folder = Path(temp)
        # Verify both additional coefficient editors, persistence and isolated UI paths.
        import tkinter as tk
        from unity_bridge.reward_generation.ui import GenerationTrainer
        for name, subfolder in [('TCL_LIBRARY', 'tcl8.6'), ('TK_LIBRARY', 'tk8.6')]:
            path = Path(sys.base_prefix) / 'tcl' / subfolder
            if path.is_dir():
                os.environ[name] = str(path)
        root = tk.Tk()
        root.withdraw()
        ui = GenerationTrainer(root, state_path=folder / 'draft.json')
        ui.apply_config(config)
        assert ui.configuration() == validate(config)
        assert all(side+'.'+factor in ui.inputs for side in 'AB' for factor in rewards.FACTORS)
        ui.persist()
        ui.close()
        root = tk.Tk()
        root.withdraw()
        ui = GenerationTrainer(root, state_path=folder / 'draft.json')
        assert ui.configuration() == validate(config)
        ui.close()
        assert (ROOT / 'app-settings/generational-trainer.json').read_bytes() == original_settings
        config.update(episodes=6, timeout=1.5, generationGames=2, batchEpisodes=2,
                      retainedGenerations=2, workers=1, seed=42718)
        for side in 'AB':
            for key in rewards.FACTORS:
                config['rewardConfig'][side][key].update(mode='fixed', value=.5)
        run, models = folder / 'run', folder / 'models'
        save_state(folder / 'config.json', config)
        with (folder / 'engine.log').open('w', encoding='utf-8') as log:
            cmd = [sys.executable, '-m', 'unity_bridge.reward_generation.train', '--config', str(folder / 'config.json'),
                   '--output', str(run), '--model-output', str(models), '--record-episodes']
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=150)
            if result.returncode:
                raise AssertionError((folder / 'engine.log').read_text(encoding='utf-8'))
            rows = [json.loads(line) for line in (run / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            (run / 'episodes.jsonl').unlink()
            result = subprocess.run([sys.executable, '-m', 'unity_bridge.reward_generation.train', '--resume', str(run),
                                     '--episodes', '8'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=150)
            if result.returncode:
                raise AssertionError((folder / 'engine.log').read_text(encoding='utf-8'))
        state = load_state(run / 'progress.json')
        assert state['status'] == 'completed' and state['episodes'] == 8 and state['kind'] == KIND
        assert [p['generation'] for p in state['opponents']] == [3, 4]
        model = FireDecider(models / 'learner.zip').model
        assert model.observation_space.shape == (OBS_SIZE,) == (24,)
        resumed_rows = [json.loads(line) for line in (run / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['episode'] for r in resumed_rows] == [7, 8]
        assert all('playerPresets' not in r['settings'] and 'rewardComponents' in r for r in resumed_rows)
        rows += resumed_rows
        assert len(rows) == 8
        assert any(r['rewardComponents'][0]['damageReward'] > 0 for r in rows)
        for row in rows:
            for i in range(2):
                assert np.isclose(row['rewards'][i], row['rewardComponents'][i]['total'])
                assert row['rewardValues']['AB'[i]]['criticalFactor'] == .5
            assert sum(c['damageFraction'] for c in row['rewardComponents']) <= 1.0000001
            if row['episode'] > 2:
                assert row['opponent']['kind'] == 'frozen_policy'
        # A saved v8 opponent can actually consume the new coefficients and make decisions.
        assert any(row['decisions'][1] > 0 for row in rows if row['episode'] > 2)
    print(json.dumps(dict(passed=True, rewardEdgeCases=True, uiRoundTrip=True, originalSettingsUnchanged=True,
                          trainedGames=8, resumed=True, retainedGenerations=[3, 4], observationSize=24)))


if __name__ == '__main__':
    main()
