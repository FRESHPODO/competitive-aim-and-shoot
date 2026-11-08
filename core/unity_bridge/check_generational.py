"""Integration checks: bootstrap removal, exact boundaries, bounded retention, resume."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import cloudpickle
import numpy as np
import torch

from unity_bridge.generation_config import defaults, validate, sample_scenario, CUSTOM
from unity_bridge.generation_pool import materialize, choose, digest
from unity_bridge.player_presets import PRESETS, KEYS
from unity_bridge.trainer_state import load_state, save_state
from unity_bridge.storage import ROOT
from unity_bridge.benchmark_simulation_speed import equal


def main():
    config = defaults()
    config.update(episodes=17, timeout=3., generationGames=4, retainedGenerations=2,
                  workers=2, batchEpisodes=3, seed=88571)
    config['presetChoices'] = {side: ['professional_mean', 'amateur_mean'] for side in 'AB'}
    for side in 'AB':
        config['rewardConfig'][side]['alpha'].update(mode='uniform', min=.1, max=.9, step=.2)
    validate(config)
    seen = {side: set() for side in 'AB'}
    for seed in range(100):
        settings, presets = sample_scenario(config, np.random.default_rng(seed))
        settings2, presets2 = sample_scenario(config, np.random.default_rng(seed))
        assert equal((settings, presets), (settings2, presets2))
        for side in 'AB':
            pid = presets[side]['id']
            seen[side].add(pid)
            for key in KEYS:
                assert settings[f'{side}.{key}']['value'] == PRESETS[pid]['parameters'][key]
    assert all(len(ids) == 2 for ids in seen.values())
    assert choose([], np.random.default_rng(0))['kind'] == 'always_approve'
    with tempfile.TemporaryDirectory(prefix='generational-check-', dir=ROOT / 'storage-reports') as temp:
        folder = Path(temp)
        run, models = folder / 'records', folder / 'models'
        save_state(folder / 'config.json', config)
        command = [sys.executable, '-m', 'unity_bridge.train_generational', '--config', str(folder / 'config.json'),
                   '--output', str(run), '--model-output', str(models)]
        with (folder / 'engine.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 100
                requested = False
                while process.poll() is None:
                    p = load_state(run / 'progress.json') or {}
                    if p.get('episodes', 0) >= 4 and not requested:
                        (run / 'stop.request').touch()
                        requested = True
                    if time.monotonic() > deadline:
                        (run / 'stop.request').touch()
                        raise TimeoutError('Integration training timed out')
                    time.sleep(.02)
                if process.returncode:
                    raise AssertionError((folder / 'engine.log').read_text(encoding='utf-8'))
                stopped = load_state(run / 'progress.json')
                assert requested and stopped['status'] == 'stopped' and 4 <= stopped['episodes'] < 17, stopped
                prefix = (run / 'episodes.jsonl').read_bytes()
                # Simulate a crash after writing a summary, before saving PPO.
                with (run / 'episodes.jsonl').open('ab') as summaries:
                    summaries.write(json.dumps(dict(episode=stopped['episodes'] + 1)).encode() + b'\n{"partial":')
                subprocess.run([sys.executable, '-m', 'unity_bridge.train_generational', '--resume', str(run)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=100)
            finally:
                if process.poll() is None:
                    (run / 'stop.request').touch()
                    process.wait(timeout=30)
        assert (run / 'episodes.jsonl').read_bytes().startswith(prefix)
        rows = [json.loads(line) for line in (run / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['episode'] for r in rows] == list(range(1, 18))
        for row in rows:
            assert 'winner' in row and 'playerPresets' not in row['settings']
            assert set(row['playerPresets']['A']) == {'id'}
            assert row['learningEnabled'] == dict(A=True, B=False)
            if row['episode'] <= 4:
                assert row['opponent']['kind'] == 'always_approve'
                assert row['decisions'][1] == row['approvals'][1]
            else:
                available = (row['episode'] - 1) // 4
                assert row['opponent']['kind'] == 'frozen_policy'
                assert max(1, available - 1) <= row['opponent']['generation'] <= available
        history = [json.loads(line) for line in (run / 'generations.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['episode'] for r in history] == [4, 8, 12, 16], history
        p = load_state(run / 'progress.json')
        assert p['status'] == 'completed' and p['target'] == p['episodes'] == 17
        assert [e['generation'] for e in p['opponents']] == [3, 4]
        assert sorted(x.name for x in (models / 'generations').iterdir()) == ['000003', '000004']
        assert not list(run.rglob('*replay*'))
        assert not list(models.rglob('agent_b.zip'))
        from unity_bridge.train_generational import recovery
        recovered, recovered_state = recovery(models)
        assert recovered_state['episodes'] == 17
        # A damaged latest payload must still leave a valid retained-pool slot.
        payload = (recovered / 'learner.zip').read_bytes()
        (recovered / 'learner.zip').write_bytes(b'incomplete')
        fallback, fallback_state = recovery(models)
        assert fallback != recovered and fallback_state['episodes'] <= 17
        (recovered / 'learner.zip').write_bytes(payload)
        from unity_bridge import generation_workers as worker
        from unity_bridge.fire_decider import FireDecider
        policy = FireDecider(models / 'learner.zip').model.policy
        worker.initialize(cloudpickle.dumps(policy), folder / 'never-stop')
        weights = cloudpickle.dumps(policy.state_dict())
        opponents = materialize(p['opponents'], models)
        original = {e['generation']: digest(e['model']) for e in opponents}
        for seed in range(4):
            before = worker.collect((seed + 711, config, opponents, weights))
            after = worker.collect((seed + 711, config, opponents, weights))
            for result in (before, after):
                assert result is not None
                result[1].pop('simulationSeconds')
            assert equal(before, after), 'Worker assignment/RNG changed a fixed-seed episode'
        worker.collect((999, config, opponents[-1:], weights))
        assert set(worker._opponents).issubset({opponents[-1]['sha256']})
        assert all(digest(e['model']) == original[e['generation']] for e in opponents)
        assert not worker._dual.recording and all(len(s.frames) <= 2 for s in worker._dual.shooters)
        # Independently exercise the three-generation option with real policy files.
        from unity_bridge.generation_pool import publish, prune
        recent = []
        three_root = folder / 'three-generation-models'
        trained_model = FireDecider(models / 'learner.zip').model
        for generation in range(1, 6):
            recent = publish(trained_model, three_root, generation,
                             generation * 100, recent, 3)
            prune(three_root, recent)
        assert [e['generation'] for e in recent] == [3, 4, 5]
        assert len(materialize(recent, three_root)) == 3
        assert sorted(p.name for p in (three_root / 'generations').iterdir()) == ['000003', '000004', '000005']
        summary = dict(passed=True, totalGames=17, resumedFrom=stopped['episodes'], goalPreserved=True,
            generationBoundaries=[4, 8, 12, 16], retained=[3, 4], bootstrapRemoved=True,
            frozenOpponentHashesUnchanged=True, deterministicEpisodes=4, workerCacheEvicted=True,
            crashTailReconciled=True, fallbackRecovery=True, threeGenerationRetention=True,
            replayRecording=False, presetSelections={side: sorted(ids) for side, ids in seen.items()})
        save_state(ROOT / 'storage-reports/generational-verification.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
