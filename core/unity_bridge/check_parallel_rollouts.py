"""Regression: recording parity, process isolation, deterministic ordering and stop."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import tempfile
import time

import cloudpickle
import numpy as np
import torch
from stable_baselines3 import PPO

from unity_bridge import rollout_workers as workers
from unity_bridge.train_competitive import PolicySpace
from unity_bridge.training_settings import defaults
from unity_bridge.competitive_rewards import defaults as reward_defaults


def assert_same(left, right):
    for a, b in zip(left[0], right[0]):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert np.array_equal(x[0], y[0]), 'observation changed'
            assert x[1:] == y[1:], 'action/reward/value/logprob changed'
    for key in ('seed', 'outcome', 'timeMs', 'rewards', 'shots', 'decisions', 'settings'):
        assert left[1][key] == right[1][key], key


def main():
    torch.set_num_threads(1)
    policies = [PPO('MlpPolicy', PolicySpace(), seed=i, device='cpu',
                    n_steps=4, batch_size=4).policy for i in range(2)]
    blob = cloudpickle.dumps(policies)
    weights = cloudpickle.dumps([p.state_dict() for p in policies])
    scratch=Path(__file__).resolve().parents[1]/'local-results'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='parallel-rollout-check-',dir=scratch) as folder:
        stop = Path(folder) / 'stop.request'
        settings = defaults()
        for side in ('A', 'B'):
            for key in ('minDamage', 'maxDamage', 'instantKill'):
                settings[f'{side}.{key}'].update(mode='fixed', value=0, min=0, max=0)
        def job(seed, timeout=4.):
            return seed, settings, reward_defaults(), {}, dict(A=True, B=True), timeout, weights

        for seed in (421, 422):
            results = []
            for recording in (True, False):
                workers.initialize(blob, stop, recording=recording)
                began = time.perf_counter()
                results.append(workers.collect(job(seed)))
                print(f'recording={recording}: {time.perf_counter()-began:.3f}s', flush=True)
                if not recording:
                    assert all(len(s.frames) <= 2 and len(s.strategy_log) <= 1
                               for s in workers._dual.shooters)
            assert_same(*results)
        print('PASS: full and lightweight paths produce identical trajectories', flush=True)

        # Enough work to verify actual overlap after Windows process startup.
        jobs = [job(500+i, 10.) for i in range(16)]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=workers.initialize, initargs=(blob, str(stop))) as pool:
            results = list(pool.map(workers.collect, jobs))
            assert len({r[1]['workerPid'] for r in results}) == 2
            assert any(a[1]['workerPid'] != b[1]['workerPid'] and
                       max(a[1]['simulationStarted'], b[1]['simulationStarted']) <
                       min(a[1]['simulationFinished'], b[1]['simulationFinished'])
                       for a in results for b in results)
            workers.initialize(blob, stop)
            for index in (0, 7, 15):
                assert_same(results[index], workers.collect(jobs[index]))
            stop.touch()
            assert pool.submit(workers.collect, jobs[0]).result() is None
        print('PASS: two overlapping processes, deterministic episodes, cooperative stop', flush=True)


if __name__ == '__main__':
    main()
