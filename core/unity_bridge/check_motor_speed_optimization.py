"""Verify exact RNG, trajectory and policy-transition parity for cached speed."""
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from src.agent.module_aim import Aim
from unity_bridge import rollout_workers
from unity_bridge.benchmark_simulation_speed import equal
from unity_bridge.fire_decider import FireDecider
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import copy_checkpoint


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('reference', ROOT / 'storage-reports/verification/motor_noise_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    fast = Aim.add_motor_noise
    cases = 0
    for dtype in (np.float32, np.float64):
        for size in (1, 3, 101, 150):
            for scale in (0., 1e-8, 1., 1000.):
                velocity = (np.random.default_rng(size).normal(size=(size, 2)) * scale).astype(dtype)
                for noise in (0., .2, .5):
                    args = (np.array([3., -4.], dtype=dtype), velocity, noise, 50)
                    np.random.seed(123)
                    before = reference.add_motor_noise(*args)
                    old_rng = np.random.get_state()
                    np.random.seed(123)
                    after = fast(*args)
                    assert equal(before, after) and equal(old_rng, np.random.get_state())
                    cases += 1
    results = []
    try:
        for run in sorted((ROOT / 'training-results').glob('20261104-*')):
            meta = json.loads((run / 'training.json').read_text(encoding='utf-8-sig'))
            with tempfile.TemporaryDirectory(prefix='motor-parity-', dir=ROOT / 'storage-reports') as temp:
                models, saved = copy_checkpoint(run, temp)
                policies = [FireDecider(models / f'agent_{s}.zip').model.policy for s in 'ab']
                weights = cloudpickle.dumps([p.state_dict() for p in policies])
                rollout_workers.initialize(cloudpickle.dumps(policies), Path(temp) / 'stop.request')
                def collect(seed, implementation):
                    Aim.add_motor_noise = implementation
                    start = time.perf_counter()
                    records, row = rollout_workers.collect((seed, meta['settings'], meta['rewardConfig'],
                        meta.get('playerPresets', {}), meta['learningEnabled'], meta['timeoutSeconds'], weights))
                    elapsed = time.perf_counter() - start
                    row = {k: v for k, v in row.items() if k not in
                           ('workerPid', 'simulationSeconds', 'simulationStarted', 'simulationFinished')}
                    return elapsed, records, row
                base = meta['seed'] + saved['episodes']
                collect(base - 1, reference.add_motor_noise)
                collect(base - 1, fast)
                samples = []
                for index in range(12):
                    seed = base + index
                    if index % 2:
                        new = collect(seed, fast); old = collect(seed, reference.add_motor_noise)
                    else:
                        old = collect(seed, reference.add_motor_noise); new = collect(seed, fast)
                    assert equal(old[1:], new[1:]), (run.name, seed)
                    assert not rollout_workers._dual.recording
                    assert all(not s.session.recording and len(s.frames) <= 2
                               and len(s.strategy_log) <= 1 and len(s.decision_log) <= 1
                               for s in rollout_workers._dual.shooters)
                    samples.append(dict(seed=seed, beforeSeconds=old[0], afterSeconds=new[0]))
                result = dict(run=run.name, exactTransitions=True, replayRecording=False,
                    speedup=sum(s['beforeSeconds'] for s in samples) / sum(s['afterSeconds'] for s in samples), samples=samples)
                results.append(result)
                print('PASS', run.name, 'speedup', round(result['speedup'], 3), flush=True)
    finally:
        Aim.add_motor_noise = fast
    (ROOT / 'storage-reports/motor-speed-verification.json').write_text(
        json.dumps(dict(exactMotorCases=cases, runs=results), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
