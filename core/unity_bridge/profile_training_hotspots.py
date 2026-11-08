"""Read-only frozen-policy profiling, isolated from the live trainers."""
import cProfile
import io
import json
import pstats
import tempfile
from pathlib import Path

import cloudpickle
import torch
from unity_bridge import rollout_workers
from unity_bridge.fire_decider import FireDecider
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import copy_checkpoint


def main():
    torch.set_num_threads(1)
    report = ROOT / 'storage-reports' / 'training-hotspots.txt'
    with report.open('w', encoding='utf-8') as output:
        for run in sorted((ROOT / 'training-results').glob('20261104-*')):
            meta = json.loads((run / 'training.json').read_text(encoding='utf-8-sig'))
            progress = json.loads((run / 'progress.json').read_text(encoding='utf-8-sig'))
            with tempfile.TemporaryDirectory(prefix='hotspot-', dir=ROOT / 'storage-reports') as temp:
                models, saved = copy_checkpoint(run, temp)
                policies = [FireDecider(models / f'agent_{s}.zip').model.policy for s in 'ab']
                weights = cloudpickle.dumps([p.state_dict() for p in policies])
                rollout_workers.initialize(cloudpickle.dumps(policies), Path(temp) / 'stop.request')
                def collect(seed):
                    return rollout_workers.collect((seed, meta['settings'], meta['rewardConfig'],
                        meta.get('playerPresets', {}), meta['learningEnabled'], meta['timeoutSeconds'], weights))
                seed = meta['seed'] + saved['episodes']
                collect(seed - 1)
                profiler = cProfile.Profile()
                profiler.enable()
                for index in range(8):
                    collect(seed + index)
                profiler.disable()
                output.write(f'\nRUN {run.name} {meta.get("playerPresets")}\n')
                pstats.Stats(profiler, stream=output).strip_dirs().sort_stats('cumulative').print_stats(40)
                pstats.Stats(profiler, stream=output).strip_dirs().sort_stats('tottime').print_stats(25)
                output.flush()
                print('Profiled', run.name, flush=True)
    print(report, flush=True)


if __name__ == '__main__':
    main()
