"""Compare frozen-policy episodes, including every PPO transition, before/after."""
import copy
import importlib.util
import json
import statistics
import time
from pathlib import Path
import cloudpickle
import numpy as np
import torch
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import copy_checkpoint
from unity_bridge.fire_decider import FireDecider
from unity_bridge import rollout_workers,server,aim_belief


def equal(a,b):
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(1);output=ROOT/'storage-reports/verification/simulation-speed'
    spec=importlib.util.spec_from_file_location('belief_reference',output/'aim_belief_reference.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    fast=aim_belief.estimate_candidate;results=[]
    for run in sorted((ROOT/'training-results').glob('20261104-*')):
        meta=json.loads((run/'training.json').read_text(encoding='utf-8-sig'))
        progress=json.loads((run/'progress.json').read_text(encoding='utf-8-sig'))
        models,_=copy_checkpoint(run,output/run.name)
        policies=[FireDecider(models/f'agent_{s}.zip').model.policy for s in 'ab']
        weights=cloudpickle.dumps([{k:v.detach().cpu().clone() for k,v in p.state_dict().items()} for p in policies])
        rollout_workers.initialize(cloudpickle.dumps(policies),output/'never-stop.request')
        def collect(seed,implementation):
            server.estimate_candidate=implementation
            job=(seed,meta['settings'],meta['rewardConfig'],meta.get('playerPresets',{}),
                 meta['learningEnabled'],meta['timeoutSeconds'],weights)
            start=time.perf_counter();records,row=rollout_workers.collect(job)
            elapsed=time.perf_counter()-start
            semantic={k:v for k,v in row.items() if k not in ('workerPid','simulationSeconds','simulationStarted','simulationFinished')}
            return elapsed,records,semantic
        base=meta['seed']+progress['episodes']
        collect(base-1,reference.estimate_candidate);collect(base-1,fast)
        samples=[]
        for index in range(3):
            seed=base+index
            if index%2:
                new=collect(seed,fast);old=collect(seed,reference.estimate_candidate)
            else:
                old=collect(seed,reference.estimate_candidate);new=collect(seed,fast)
            assert equal(old[1:],new[1:]),(run.name,seed,'trajectory changed')
            samples.append(dict(seed=seed,beforeSeconds=old[0],afterSeconds=new[0],
                                gameSeconds=old[2]['timeMs']/1000,decisions=old[2]['decisions']))
        result=dict(run=run.name,profiles=meta.get('playerPresets'),samples=samples,
            speedup=sum(s['beforeSeconds'] for s in samples)/sum(s['afterSeconds'] for s in samples),exactTransitions=True)
        results.append(result)
        (ROOT/'storage-reports/simulation-speed-optimization.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS',run.name,'speedup',round(result['speedup'],2),'all observations/actions/values/logprobs/rewards identical',flush=True)
    server.estimate_candidate=fast


if __name__=='__main__':main()
