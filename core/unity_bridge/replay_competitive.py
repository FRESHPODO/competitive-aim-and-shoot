"""Observe held-out matches between the two learned agents in the 3D viewer."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from unity_bridge.fire_decider import FireDecider
from unity_bridge.server import DualSession
from unity_bridge.training_settings import sample,validate,migrate
from unity_bridge.replay import save_replay
from unity_bridge.competitive_rewards import defaults as default_rewards,from_kill_events,sample as sample_rewards
from unity_bridge.storage import training_directory
from unity_bridge.trainer_state import copy_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--seed',type=int,default=900123)
    args=parser.parse_args()
    if args.rounds<1: parser.error('rounds must be positive')
    root=training_directory(args.run);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    settings=json.loads((root/'settings.json').read_text(encoding='utf-8-sig'));settings=migrate(settings);validate(settings)
    metadata=json.loads((root/'training.json').read_text(encoding='utf-8-sig'))
    timeout=metadata.get('timeoutSeconds',30.)
    state=json.loads((root/'progress.json').read_text(encoding='utf-8-sig'))
    source,_=copy_checkpoint(root,output/'model-inputs')
    torch.set_num_threads(1)
    policies=[FireDecider(source/f'agent_{s}.zip') for s in ('a','b')]
    dual=DualSession();rows=[]
    for i in range(args.rounds):
        seed=args.seed+i
        rng=np.random.default_rng(seed)
        request=sample(settings,rng);reward_values=sample_rewards(metadata.get('rewardConfig',default_rewards()),rng)
        request['rewardValues']=reward_values
        request['playerPresets']=metadata.get('playerPresets',{})
        dual.round=seed;dual.reset(request)
        dual.inactivity_limit_ms=timeout*1000
        for j in range(2):
            dual.shooters[j].decider=policies[j]
            dual.shooters[j].decision_mode=getattr(policies[j].model,'decision_mode','candidate')
        while dual.time<timeout*1000 and dual.combat.hp>0 and not dual.draw:dual.step(deadline_ms=timeout*1000)
        outcome='kill' if dual.combat.hp<=0 else ('no_shots_draw' if dual.draw else 'time_limit_draw')
        rewards=from_kill_events(dual.combat.events,reward_values)
        rows.append(dict(seed=seed,outcome=outcome,scores=dual.combat.scores,rewards=rewards))
        save_replay(dual,output/f'round-{i:03d}.html',request,f'A/B competition / seed {seed}')
    (output/'evaluation.json').write_text(json.dumps(dict(source=str(source),rounds=rows),indent=2),encoding='utf-8')
    print(f'Saved {len(rows)} competitive replays to {output}',flush=True)


if __name__=='__main__':main()
