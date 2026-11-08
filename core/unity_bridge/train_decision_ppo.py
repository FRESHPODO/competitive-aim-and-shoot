"""Train a universal, parameter-conditioned fire approval policy on the real bridge."""
import argparse
import json
import time
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from unity_bridge.decision_env import CandidateEnv
from unity_bridge.fire_decider import encode_context, OBS_SIZE, FEATURE_NAMES, FEATURE_VERSION, MODEL_DIR
from unity_bridge.atomic_io import replace_with_retry
from stable_baselines3.common.vec_env import SubprocVecEnv
from functools import partial


def make_env(seed, scenario, filename):
    torch.set_num_threads(1)
    return Monitor(CandidateEnv(encode_context, OBS_SIZE, seed=seed, scenario=scenario), filename)


class SaveProgress(BaseCallback):
    def __init__(self, directory, requested):
        super().__init__()
        self.directory = directory
        self.requested = requested
        self.started = time.monotonic()
        self.stopped = False
        self.target = requested

    def write_status(self, status):
        episodes=list(self.model.ep_info_buffer or [])
        metrics={k:float(v) for k,v in self.logger.name_to_value.items()
                 if isinstance(v,(int,float,np.number)) and np.isfinite(v)}
        data=dict(status=status,decisions=self.num_timesteps,requested=self.target,
                  pid=os.getpid(),updatedAt=time.time(),
                  elapsedSeconds=time.monotonic()-self.started,
                  meanReward=float(np.mean([e['r'] for e in episodes])) if episodes else None,
                  meanEpisodeLength=float(np.mean([e['l'] for e in episodes])) if episodes else None,
                  metrics=metrics)
        path=self.directory/'progress.tmp.json'
        try:
            path.write_text(json.dumps(data,indent=2),encoding='utf-8')
            replace_with_retry(path,self.directory/'progress.json',attempts=6)
        except PermissionError:
            # Monitoring is best-effort: a reader lock must never stop PPO.
            print('Status file temporarily locked; training continues.',flush=True)

    def save(self, name):
        temporary=self.directory/f'{name}.tmp.zip'
        self.model.save(temporary)
        replace_with_retry(temporary,self.directory/f'{name}.zip',attempts=60)

    def _on_training_start(self):
        self.target=self.model.num_timesteps+self.requested
        self.write_status('running')

    def _on_step(self):
        if self.num_timesteps%16==0: self.write_status('running')
        if (self.directory/'stop.request').exists():
            self.stopped=True
            return False
        return True

    def _on_rollout_end(self):
        self.save('latest')
        self.save(f'step_{self.num_timesteps:08d}')
        self.write_status('running')
        print(f'Saved rollout at {self.num_timesteps} decisions', flush=True)

    def _on_training_end(self):
        self.save('latest')
        self.write_status('stopped' if self.stopped else 'completed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=8192)
    parser.add_argument('--seed',type=int,default=421)
    parser.add_argument('--resume',type=str)
    parser.add_argument('--scenario', help='JSON file containing fixed game/profile overrides')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--output',type=str,default=str(MODEL_DIR))
    args=parser.parse_args()
    if args.steps < 1 or args.workers < 1:
        parser.error('steps and workers must be positive')
    from pathlib import Path
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'stop.request').unlink(missing_ok=True)
    torch.set_num_threads(1)
    scenario=json.loads(Path(args.scenario).read_text(encoding='utf-8')) if args.scenario else None
    factories=[partial(make_env,args.seed+i*1000000,scenario,str(output/('episodes' if args.workers==1 else f'episodes-{i}')))
               for i in range(args.workers)]
    env=factories[0]() if args.workers==1 else SubprocVecEnv(factories, start_method='spawn')
    started=time.monotonic()
    progress=None
    try:
        if args.resume:
            model=PPO.load(args.resume,env=env,device='cpu')
            model.tensorboard_log=str(output/'tensorboard')
            if tuple(getattr(model,'decision_feature_names',()))!=FEATURE_NAMES:
                raise ValueError('Cannot resume a different observation schema')
        else:
            model=PPO('MlpPolicy',env,device='cpu',seed=args.seed,verbose=1,
                      n_steps=256,batch_size=64,n_epochs=10,learning_rate=3e-4,
                      gamma=1.,gae_lambda=.95,ent_coef=.01,target_kl=.03,
                      tensorboard_log=str(output/'tensorboard'),
                      policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])))
            model.decision_feature_names=FEATURE_NAMES
            model.decision_feature_version=FEATURE_VERSION
        progress=SaveProgress(output,args.steps)
        model.learn(total_timesteps=args.steps,callback=progress,
                    reset_num_timesteps=not bool(args.resume))
        progress.save('candidate')
        metadata=dict(algorithm='PPO',kind='parameter-conditioned universal policy',
                      seed=args.seed,total_decisions=model.num_timesteps,
                      scenario=scenario,workers=args.workers,
                      seconds=time.monotonic()-started,features=FEATURE_NAMES,
                      reward='terminal own score minus opponent score; time cost 0.001/second',
                      training='randomized scenarios with fixed JSON overrides' if scenario else '40% asymmetric default noise with random speed; 60% randomized settings and both profiles',
                      opponent='frozen original Aim-and-Shoot, no approval gate',
                      limitation='100ms shared-state snapshots; no guarantee outside trained distribution')
        (output/'training.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(f'Trained candidate saved to {output / "candidate.zip"}',flush=True)
    except BaseException:
        if progress is not None:
            # Preserve the latest trained weights even on an environment error.
            try:
                progress.save(f'recovery_{model.num_timesteps:08d}')
            except Exception as save_error:
                print(f'Recovery save failed: {save_error}',flush=True)
            progress.write_status('failed')
        raise
    finally:
        env.close()


if __name__=='__main__': main()
