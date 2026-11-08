"""Collect both agents' trajectories in the same games, then update both PPOs."""
import argparse
import json
import time
import shutil
import tempfile
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
import cloudpickle
import psutil
from unity_bridge.opponent_pool import prepare as prepare_opponents, select as select_opponent
from unity_bridge.curriculum_workers import initialize as initialize_worker, collect as collect_episode
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from unity_bridge.fire_decider import encode_context, OBS_SIZE, FEATURE_NAMES, FEATURE_VERSION,validate_model_observation
from unity_bridge.training_settings import validate,sample,migrate
from unity_bridge.server import DualSession
from unity_bridge.atomic_io import replace_with_retry
from unity_bridge.trainer_state import checkpoint,copy_checkpoint,save_state
from unity_bridge.episode_results import write_episode_result
from unity_bridge.storage import ROOT,CURRICULUM_MODEL_ROOT,training_directory,model_lock,resolve_path
import os
from unity_bridge.competitive_rewards import defaults as default_rewards,validate as validate_rewards,from_kill_events,normalize as normalize_rewards,sample as sample_rewards
from unity_bridge.gpu_budget import configure as configure_gpu,gpu_work,budget_status,TrainingStopped

class BudgetRolloutBuffer(RolloutBuffer):
    def get(self,batch_size=None):
        batches=super().get(batch_size)
        while True:
            with gpu_work():
                try:batch=next(batches)
                except StopIteration:return
                yield batch

def gpu_call(function,*args,**kwargs):
    with gpu_work():return function(*args,**kwargs)

def inference_policy(model):
    if model.device.type=='cpu':return model.policy
    # GPU is useful for minibatch updates; single decisions stay on the CPU.
    with torch.random.fork_rng(devices=[]),gpu_work():
        policy=model.policy_class(model.observation_space,model.action_space,lambda _:0.,**model.policy_kwargs)
        policy.load_state_dict({key:value.detach().cpu() for key,value in model.policy.state_dict().items()})
    policy.set_training_mode(False);policy.requires_grad_(False);return policy

class PolicySpace(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(OBS_SIZE,),np.float32)
    action_space=spaces.Discrete(2)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);return np.zeros(OBS_SIZE,np.float32),{}
    def step(self,action):raise RuntimeError('Use the joint game collector')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--learner-model',help='A PPO checkpoint to keep as the learning subject; required for a new curriculum')
    p.add_argument('--settings',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--batch-episodes',type=int,default=16)
    p.add_argument('--workers',type=int,default=0,help='Parallel CPU simulation processes; 0 selects up to 4 based on cores and available RAM')
    p.add_argument('--checkpoint-seconds',type=float,default=60,help='Seconds between rolling saves; snapshots and final/stop saves are always kept')
    p.add_argument('--seed',type=int,default=421);p.add_argument('--resume')
    p.add_argument('--opponent-pool',help='JSON frozen-opponent list; A trains, B cycles by episode')
    p.add_argument('--rewards',help='JSON containing A/B win reward, loss cost and draw reward')
    p.add_argument('--freeze-a',action='store_true',help='Keep A playing without updating its PPO')
    p.add_argument('--freeze-b',action='store_true',help='Keep B playing without updating its PPO')
    p.add_argument('--device',default='cpu',help='PPO update device. Simulation and single-decision inference always use CPU; cuda:0 optionally accelerates minibatch updates')
    p.add_argument('--model-output',help='Model directory, separate from training records')
    p.add_argument('--snapshot-interval',type=int,default=10000,help='Games between durable snapshots; two rolling paired recovery slots are always maintained')
    p.add_argument('--record-episodes',action='store_true',help='Include full settings in per-game analysis records')
    args=p.parse_args()
    if args.episodes<1 or args.timeout<=0 or not np.isfinite(args.timeout) or args.batch_episodes<1 or args.seed<0 or args.snapshot_interval<1 or args.workers<0 or not np.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds<=0:p.error('Invalid episodes, timeout, batch, seed or snapshot interval')
    if args.workers==0:
        args.workers=max(1,min(4,psutil.cpu_count(logical=False) or 1,int(psutil.virtual_memory().available/(768*1024**2))))
    args.workers=min(args.workers,args.batch_episodes,args.episodes)
    if args.resume:args.resume=str(training_directory(args.resume))
    document=json.loads(Path(args.settings).read_text(encoding='utf-8-sig'))
    settings=migrate(document.get('parameters',document));validate(settings)
    reward_config=default_rewards()
    if args.resume:
        prior_rewards=json.loads((Path(args.resume)/'training.json').read_text(encoding='utf-8-sig'))
        reward_config=prior_rewards.get('rewardConfig',reward_config)
    prior_elapsed=0.
    if args.resume:
        prior_progress=Path(args.resume)/'progress.json'
        if prior_progress.exists():prior_elapsed=json.loads(prior_progress.read_text(encoding='utf-8-sig')).get('elapsedSeconds',0.)
    reward_config=document.get('rewardConfig',reward_config)
    if args.rewards:reward_config=json.loads(Path(args.rewards).read_text(encoding='utf-8-sig'))
    reward_config=normalize_rewards(reward_config);validate_rewards(reward_config)
    learning_enabled=dict(A=True,B=True)
    if args.resume:learning_enabled.update(prior_rewards.get('learningEnabled',{}))
    learning_enabled.update(document.get('learningEnabled',{}))
    if args.freeze_a:learning_enabled['A']=False
    if args.freeze_b:learning_enabled['B']=False
    pool_config=document.get('opponentPool',prior_rewards.get('opponentPool',{}) if args.resume else {})
    if args.opponent_pool:pool_config=json.loads(Path(args.opponent_pool).read_text(encoding='utf-8-sig'))
    if not pool_config.get('enabled'):p.error('This trainer requires an enabled B opponent list')
    if not args.resume and not args.learner_model:p.error('Select the fixed A learner model with --learner-model')
    if args.freeze_a:p.error('A is the learner in this trainer; only B opponents are frozen')
    learning_enabled=dict(A=True,B=False)
    if set(learning_enabled)!={'A','B'} or any(type(v) is not bool for v in learning_enabled.values()):p.error('learningEnabled must contain boolean A/B values')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'training.json').exists():raise ValueError('Choose an unused output folder')
    model_output=resolve_path(args.model_output) if args.model_output else CURRICULUM_MODEL_ROOT/output.name
    if (model_output/'model-manifest.json').exists():raise ValueError('Choose an unused model folder')
    model_output.mkdir(parents=True,exist_ok=True)
    pool_config=prepare_opponents(pool_config,output)
    learner_input=prior_rewards.get('learnerInput',{}) if args.resume else {}
    learner_path=None
    if not args.resume:
        learner_path=output/'learner-input.zip'
        learner_source=resolve_path(args.learner_model)
        shutil.copyfile(learner_source,learner_path)
        from unity_bridge.fire_decider import FireDecider
        FireDecider(learner_path)
        learner_input=dict(source=str(learner_source),model=str(learner_path.resolve()),sha256=hashlib.sha256(learner_path.read_bytes()).hexdigest())
    resume_copy=tempfile.TemporaryDirectory(prefix='paired-resume-',dir=output)
    save_sequence=0;last_archive_bucket=None;last_save=0.;pool=None
    timings=dict(collectionSeconds=0.,updateSeconds=0.,checkpointSeconds=0.)
    torch.set_num_threads(1);models=[];episodes=0;started=time.monotonic();status='running'
    metadata=dict(mode='joint-game independent PPO self-play',settings=settings,targetEpisodes=args.episodes,
        timeoutSeconds=args.timeout,batchEpisodes=args.batch_episodes,workers=args.workers,checkpointSeconds=args.checkpoint_seconds,seed=args.seed,resume=args.resume,
        reward='own kill +2*alpha; opponent kill -2*(1-alpha); draw 0; opponent alpha hidden',rewardConfig=reward_config,
        scope='Request-only aim-belief fire approval PPO; aiming SAC frozen',decisionMode='bump',decisionTrigger='source_fire_request',damageEstimate='aim_perception_and_ideal_plan',firePlanRule='commit_until_shot',cooldownRule='reject_early_proposal_replan',observationVersion=FEATURE_VERSION,status=status,
        playerPresets=document.get('playerPresets',prior_rewards.get('playerPresets',{}) if args.resume else {}),learningEnabled=learning_enabled,device=args.device,gpuTargetPercent=65 if str(args.device).startswith('cuda') else None,
        modelDirectory=str(model_output),snapshotInterval=args.snapshot_interval,opponentPool=pool_config,learnerInput=learner_input,trainerKind="fixed-learner-curriculum")
    (output/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    (output/'training.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def progress():
        data=dict(status=status,episodes=episodes,target=args.episodes,pid=os.getpid(),decisions=[m.num_timesteps for m in models],elapsedSeconds=prior_elapsed+time.monotonic()-started,learningEnabled=learning_enabled,seed=args.seed,device=args.device,gpuBudget=budget_status(),workers=args.workers,timings=timings,opponentCount=len(pool_config['opponents']) if pool_config['enabled'] else 0)
        tmp=output/'progress.tmp.json';tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');replace_with_retry(tmp,output/'progress.json')
    def save_models(*,final=False):
        nonlocal save_sequence,last_archive_bucket,last_save
        save_started=time.monotonic()
        if len(models)==2:
            with model_lock(model_output):
                live=model_output/'live'/f'slot-{save_sequence%2}';live.mkdir(parents=True,exist_ok=True)
                (live/'complete.json').unlink(missing_ok=True)
                for i,m in enumerate(models):
                    name=f'agent_{"a" if i==0 else "b"}.zip';tmp=live/(name+'.tmp.zip');m.save(tmp);replace_with_retry(tmp,live/name)
                    shutil.copy2(live/name,model_output/(name+'.tmp.zip'));replace_with_retry(model_output/(name+'.tmp.zip'),model_output/name)
                torch.save(torch.get_rng_state(),live/'torch_rng.pt')
                if str(args.device).startswith('cuda'):torch.save(torch.cuda.get_rng_state(args.device),live/'cuda_rng.pt')
                save_state(live/'complete.json',dict(episodes=episodes,targetEpisodes=args.episodes))
                bucket=episodes//args.snapshot_interval
                if last_archive_bucket is None or bucket!=last_archive_bucket or final:
                    snapshot=model_output/'snapshots'/f'{episodes:08d}'
                    if not (snapshot/'complete.json').exists():
                        snapshot.mkdir(parents=True,exist_ok=True)
                        for file in live.iterdir():
                            if file.name!='complete.json':shutil.copy2(file,snapshot/file.name)
                        shutil.copy2(live/'complete.json',snapshot/'complete.json')
                    last_archive_bucket=bucket
                save_sequence+=1
                save_state(model_output/'model-manifest.json',dict(trainingDirectory=str(output.resolve()),completedEpisodes=episodes,targetEpisodes=args.episodes,status=status,
                    rollingRecoverySlots=2,snapshotInterval=args.snapshot_interval))
        metadata.update(completedEpisodes=episodes,status=status)
        save_state(output/'training.json',metadata)
        last_save=time.monotonic()
        timings['checkpointSeconds']+=last_save-save_started
    try:
        configure_gpu(args.device,output/'stop.request')
        os.environ['ANS_TORCH_DEVICE']='cpu'
        print(f'PPO update device: {args.device}; simulation and action inference: CPU'+('; aggregate GPU target: 65% (cooperative)' if str(args.device).startswith('cuda') else ''),flush=True)
        if args.resume:
            prior=json.loads((Path(args.resume)/'training.json').read_text(encoding='utf-8-sig'))
            if prior.get('mode')!='joint-game independent PPO self-play':raise ValueError('Resume requires a joint-game training run')
            if prior.get('decisionMode')!='bump':raise ValueError('Resume requires a binary BUMP policy')
            source,saved=copy_checkpoint(args.resume,resume_copy.name)
            episodes=saved['episodes']
            if episodes>=args.episodes:raise ValueError('Total game target must exceed the resumed game count')
        for i,side in enumerate(('a','b')):
            if args.resume or side=='a':
                initial=source/f'agent_{side}.zip' if args.resume else learner_path
                old=gpu_call(PPO.load,initial,device=args.device)
                validate_model_observation(old)
                m=old;m.set_env(PolicySpace())
            elif pool_config['enabled'] and side=='b':
                m=gpu_call(PPO.load,pool_config['opponents'][0]['model'],device=args.device)
                validate_model_observation(m)
            else:
                m=gpu_call(PPO,'MlpPolicy',PolicySpace(),seed=args.seed+i,device=args.device,n_steps=256,batch_size=64,
                    gamma=1.,gae_lambda=.95,ent_coef=.01,target_kl=.03,
                    policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])))
                m.decision_feature_names=FEATURE_NAMES;m.decision_feature_version=FEATURE_VERSION
            if not(pool_config['enabled'] and side=='b'):m.decision_mode='bump'
            m.decision_fire_plan_rule='commit_until_shot'
            m.decision_cooldown_rule='reject_early_proposal_replan'
            if not learning_enabled[side.upper()]:m.policy.requires_grad_(False)
            m.set_logger(configure(str(output/side),['stdout','csv']));models.append(m)
        if args.resume and (source/'torch_rng.pt').exists():torch.set_rng_state(torch.load(source/'torch_rng.pt',weights_only=True))
        if args.resume and str(args.device).startswith('cuda') and (source/'cuda_rng.pt').exists():torch.cuda.set_rng_state(torch.load(source/'cuda_rng.pt',weights_only=True),args.device)
        resume_copy.cleanup()
        policies=[inference_policy(m) for m in models]
        metadata['actualDevices']=dict(ppo=[str(next(m.policy.parameters()).device) for m in models],
            aim=['cpu','cpu'],inference=['cpu','cpu'])
        metadata['collector']='persistent spawned processes; frozen policy per batch; lightweight simulation'
        print(f'Parallel simulation workers: {args.workers}; actual model devices: {metadata["actualDevices"]}',flush=True)
        # Serialize CPU policies as bytes: never pickle CUDA tensors into a child process.
        from copy import deepcopy
        cpu_policies=[deepcopy(policy).to('cpu') for policy in policies]
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            os.environ[name]='1'
        pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(cloudpickle.dumps(cpu_policies),str(output.resolve()/'stop.request')))
        save_models();progress()
        while episodes<args.episodes:
            if (output/'stop.request').exists():status='stopped';break
            records=[[],[]]
            for model in models:model.policy.set_training_mode(False)
            weights=cloudpickle.dumps([{k:v.detach().cpu().clone() for k,v in m.policy.state_dict().items()} for m in models])
            count=min(args.batch_episodes,args.episodes-episodes)
            jobs=[(args.seed+episodes+j,settings,reward_config,metadata['playerPresets'],learning_enabled,args.timeout,weights,select_opponent(pool_config,episodes+j)) for j in range(count)]
            collection_started=time.monotonic()
            futures=[pool.submit(collect_episode,job) for job in jobs]
            # Consume in seed order so completion order cannot change PPO trajectories.
            for future in futures:
                while True:
                    try:result=future.result(timeout=1.);break
                    except FutureTimeout:progress()
                if result is None:
                    status='stopped'
                    for pending in futures:pending.cancel()
                    break
                episode_records,row=result
                for i in range(2):records[i].extend(episode_records[i])
                episodes+=1;row['episode']=episodes
                with (output/'episodes.jsonl').open('a',encoding='utf-8') as f:
                    write_episode_result(f,row,detailed=args.record_episodes)
                progress()
                opponent_text=f' / B opponent: {row["opponentPool"]["name"]}' if row.get('opponentPool') else ''
                print(f'Game {episodes}/{args.episodes}: {row["outcome"]}, reward A/B {row["rewards"][0]:+g}/{row["rewards"][1]:+g}'+opponent_text,flush=True)
            timings['collectionSeconds']+=time.monotonic()-collection_started
            update_started=time.monotonic()
            # Policy weights stayed fixed while collecting this common batch.
            for i,m in enumerate(models):
                if not learning_enabled['AB'[i]]:continue
                if not records[i]:continue
                if len(records[i])<2:
                    print(f'Agent {i}: fewer than 2 decisions; recorded without PPO update',flush=True);continue
                buffer=BudgetRolloutBuffer(len(records[i]),m.observation_space,m.action_space,device=args.device,gamma=1.,gae_lambda=.95,n_envs=1)
                for obs,action,reward,first,value,logprob in records[i]:
                    buffer.add(obs,np.array([action]),np.array([reward]),np.array([first]),torch.tensor([value]),torch.tensor([logprob]))
                buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]))
                m.rollout_buffer=buffer;m.num_timesteps+=len(records[i]);m._current_progress_remaining=1-episodes/args.episodes
                m.train();m.logger.dump(step=episodes)
                policies[i]=inference_policy(m)
            timings['updateSeconds']+=time.monotonic()-update_started
            if time.monotonic()-last_save>=args.checkpoint_seconds or episodes//args.snapshot_interval!=last_archive_bucket:
                save_models()
            progress()
            if status=='stopped':break
        if status=='running':status='completed'
    except (TrainingStopped,KeyboardInterrupt):
        status='stopped'
        (output/'stop.request').touch()
    except BaseException:
        status='failed';raise
    finally:
        if pool is not None:
            if status!='completed':(output/'stop.request').touch()
            pool.shutdown(wait=True,cancel_futures=True)
        save_models(final=True);resume_copy.cleanup();metadata.update(status=status,completedEpisodes=episodes,decisions=[m.num_timesteps for m in models])
        (output/'training.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');progress()

if __name__=='__main__':main()
