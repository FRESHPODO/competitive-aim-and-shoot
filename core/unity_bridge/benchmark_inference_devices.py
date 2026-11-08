"""Isolated CPU/GPU inference timing with the configured aggregate GPU governor."""
import copy
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from unity_bridge.fire_decider import FireDecider
from unity_bridge.gpu_budget import configure, gpu_work, budget_status
from unity_bridge.server import Session
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import copy_checkpoint


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    configure('cuda:0')
    source = ROOT / 'training-results/20261104-162112-100729-408bbdd1'
    results = []
    with tempfile.TemporaryDirectory(prefix='device-benchmark-', dir=ROOT / 'storage-reports') as temp:
        models, _ = copy_checkpoint(source, temp)
        aim = Session(recording=False).sim.model
        ppo = FireDecider(models / 'agent_a.zip').model
        for name, model in [('aim_SAC', aim), ('additional_PPO', ppo)]:
            cpu = model.policy.to('cpu')
            with gpu_work():
                gpu = copy.deepcopy(cpu).to('cuda:0')
            for batch in (1, 16, 64, 256):
                obs = np.zeros((batch, *model.observation_space.shape), dtype=np.float32)
                measurements = {}
                for device, policy in [('cpu', cpu), ('cuda:0', gpu)]:
                    samples = []
                    with gpu_work():
                        for _ in range(10):
                            policy.predict(obs, deterministic=True)
                    for _ in range(5):
                        # Excludes deliberate budget waiting; includes transfers
                        # since predict returns a NumPy action on the CPU.
                        with gpu_work():
                            began = time.perf_counter()
                            for _ in range(50):
                                policy.predict(obs, deterministic=True)
                            if device != 'cpu':
                                torch.cuda.synchronize()
                            samples.append((time.perf_counter() - began) / 50 * 1000)
                    measurements[device] = statistics.median(samples)
                result = dict(model=name, batch=batch, milliseconds=measurements,
                              rawGpuSpeedup=measurements['cpu'] / measurements['cuda:0'])
                results.append(result)
                print(name, batch, result, flush=True)
    report = dict(gpu=torch.cuda.get_device_name(), results=results, governor=budget_status(),
                  note='Inference only, host/device transfers included, budget waiting excluded; no live model changed.')
    (ROOT / 'storage-reports/inference-device-benchmark.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
