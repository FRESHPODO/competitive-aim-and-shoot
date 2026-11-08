"""A single conditional PPO learner against automatically refreshed recent generations."""
import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import cloudpickle
import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from unity_bridge.atomic_io import replace_with_retry
from unity_bridge.reward_generation.decider import FEATURE_NAMES, FEATURE_VERSION, validate_model_observation
from unity_bridge.reward_generation.config import KIND, validate
from unity_bridge.generation_pool import digest, materialize, publish, prune
from unity_bridge.reward_generation.workers import initialize, collect
from unity_bridge.storage import ROOT, model_lock, resolve_path
from unity_bridge.trainer_state import load_state, save_state
from unity_bridge.episode_results import trim_episode_results, write_episode_result
from unity_bridge.train_competitive import BudgetRolloutBuffer
from unity_bridge.reward_generation.decider import PolicySpace

from unity_bridge.reward_generation.paths import RESULTS, MODELS


def recovery(model_root):
    candidates = []
    for marker in (Path(model_root) / 'live').glob('*/state.json'):
        state = load_state(marker)
        if state and state.get('kind') == KIND:
            candidates.append((state['episodes'], marker.stat().st_mtime_ns, marker.parent, state))
    for _, _, folder, state in sorted(candidates, key=lambda x: x[:2], reverse=True):
        try:
            if digest(folder / 'learner.zip') != state['modelSha256'] or not (folder / 'torch_rng.pt').is_file():
                continue
            materialize(state['opponents'], model_root)
            return folder, state
        except (OSError, ValueError):
            continue
    raise ValueError('유효한 단일 모델 복구 체크포인트가 없습니다.')


def is_running(run):
    p = load_state(Path(run) / 'progress.json') or {}
    try:
        process = psutil.Process(p['pid'])
        command = process.cmdline()
        if 'unity_bridge.reward_generation.train' not in command:
            return False
        for option in ('--output', '--resume'):
            if option in command:
                path = Path(command[command.index(option) + 1])
                if not path.is_absolute():
                    path = Path(process.cwd()) / path
                if path.resolve() == Path(run).resolve():
                    return True
    except (psutil.Error, KeyError, ValueError):
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--model-output')
    parser.add_argument('--resume')
    parser.add_argument('--record-episodes', action='store_true', help='Include full settings in per-game analysis records')
    parser.add_argument('--episodes', type=int, help='Optional extended total target on resume')
    args = parser.parse_args()
    run = Path(args.resume or args.output or '').resolve()
    if not args.resume and (not args.config or not args.output):
        parser.error('새 학습에는 --config와 --output이 필요합니다.')
    if args.resume and args.output and Path(args.output).resolve() != run:
        parser.error('이어 학습은 기존 기록 폴더에서 수행합니다.')
    saved = None
    if args.resume:
        if is_running(run):
            raise ValueError('이미 실행 중인 학습입니다. 먼저 중단·저장하세요.')
        metadata = load_state(run / 'training.json') or {}
        if metadata.get('trainerKind') != KIND:
            raise ValueError('이 학습기의 이어 학습 폴더가 아닙니다.')
        model_root = resolve_path(metadata['modelDirectory'])
        source, saved = recovery(model_root)
        config = validate(saved['config'])
        if args.episodes is not None:
            config['episodes'] = args.episodes
            config = validate(config)
        if config['episodes'] <= saved['episodes']:
            raise ValueError('목표 총 경기 수는 저장된 경기 수보다 커야 합니다.')
    else:
        config = validate(json.loads(Path(args.config).read_text(encoding='utf-8-sig')), resolve_seed=True)
        model_root = Path(args.model_output).resolve() if args.model_output else MODELS / run.name
        if run.exists() and any(run.iterdir()):
            raise ValueError('비어 있는 새 학습 기록 폴더를 지정하세요.')
        if model_root.exists() and any(model_root.iterdir()):
            raise ValueError('비어 있는 새 모델 폴더를 지정하세요.')
    if args.model_output and Path(args.model_output).resolve() != model_root:
        raise ValueError('이어 학습에서는 모델 폴더를 변경할 수 없습니다.')
    run.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    # Exclude two launchers racing for the same run. Different runs are independent.
    from filelock import FileLock, Timeout
    try:
        run_lock = FileLock(str(run / '.training.lock'), timeout=0)
        run_lock.acquire()
    except Timeout:
        raise ValueError('같은 폴더에서 이미 학습 중입니다.')
    try:
        train(run, model_root, config, saved, source if saved else None, record_episodes=args.record_episodes)
    finally:
        run_lock.release()


def train(run, model_root, config, saved=None, source=None, *, record_episodes=False):
    torch.set_num_threads(1)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['ANS_TORCH_DEVICE'] = 'cpu'
    stop = run / 'stop.request'
    stop.unlink(missing_ok=True)
    state = deepcopy(saved) if saved else dict(kind=KIND, episodes=0, generation=0, opponents=[],
        updates=0, elapsedSeconds=0., wins=0, losses=0, draws=0,
        timings=dict(collectionSeconds=0., updateSeconds=0., checkpointSeconds=0.))
    state['config'] = config
    state.pop('modelSha256', None)
    prior_elapsed = state['elapsedSeconds']
    began = time.monotonic()
    last_progress = last_save = 0.
    save_sequence = 0
    status = 'running'
    pool = model = None
    log = None
    workers = config['workers'] or max(1, min(4, psutil.cpu_count(logical=False) or 1,
        int(psutil.virtual_memory().available / (768 * 1024**2))))
    workers = min(workers, config['batchEpisodes'], config['episodes'] - state['episodes'])
    metadata = dict(trainerKind=KIND, mode='single conditional PPO against recent frozen opponents',
        modelDirectory=str(model_root), trainingDirectory=str(run), decisionMode='bump',
        decisionTrigger='source_fire_request', observationVersion=FEATURE_VERSION,
        firePlanRule='commit_until_shot', cooldownRule='reject_early_proposal_replan',
        learningEnabled=dict(A=True, B=False), device='cpu', workers=workers,
        replayRecording=False, rewardDefinition='outcome + damageFactor * actualDamage/maxHp + criticalFactor * criticalKill', config=config)
    save_state(run / 'config.json', config)

    def progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < .5:
            return
        state['elapsedSeconds'] = prior_elapsed + now - began
        data = dict(state, status=status, target=config['episodes'], pid=os.getpid(), workers=workers,
            decisions=model.num_timesteps if model else 0,
            nextGenerationAt=(state['episodes'] // config['generationGames'] + 1) * config['generationGames'],
            opponentCount=len(state['opponents']) or 1, bootstrap=not bool(state['opponents']))
        save_state(run / 'progress.json', data)
        last_progress = now

    def checkpoint(*, pool_changed=False):
        nonlocal save_sequence, last_save
        if model is None:
            return
        start = time.monotonic()
        progress(force=True)
        with model_lock(model_root):
            folder = model_root / 'live' / f'slot-{save_sequence % 2}'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'state.json').unlink(missing_ok=True)
            temp = folder / 'learner.tmp.zip'
            model.save(temp)
            replace_with_retry(temp, folder / 'learner.zip')
            torch.save(torch.get_rng_state(), folder / 'torch_rng.pt')
            marker = dict(state, modelSha256=digest(folder / 'learner.zip'))
            save_state(folder / 'state.json', marker)
            if pool_changed:
                # Both recovery slots must refer to retained opponents before
                # deleting a generation used by the previous recovery slot.
                standby = model_root / 'live' / f'slot-{1 - save_sequence % 2}'
                standby.mkdir(parents=True, exist_ok=True)
                (standby / 'state.json').unlink(missing_ok=True)
                shutil.copy2(folder / 'learner.zip', standby / 'learner.tmp.zip')
                replace_with_retry(standby / 'learner.tmp.zip', standby / 'learner.zip')
                shutil.copy2(folder / 'torch_rng.pt', standby / 'torch_rng.pt')
                save_state(standby / 'state.json', marker)
            shutil.copy2(folder / 'learner.zip', model_root / 'learner.tmp.zip')
            replace_with_retry(model_root / 'learner.tmp.zip', model_root / 'learner.zip')
            save_state(model_root / 'model-manifest.json', dict(trainingDirectory=str(run),
                trainerKind=KIND, completedEpisodes=state['episodes'], targetEpisodes=config['episodes'],
                status=status, singleModel='learner.zip', rollingRecoverySlots=2))
            save_state(model_root / 'opponents.json', state['opponents'])
            # Retired files may be removed only once the new window is recoverable.
            prune(model_root, state['opponents'])
        save_state(run / 'training.json', dict(metadata, status=status, completedEpisodes=state['episodes']))
        state['timings']['checkpointSeconds'] += time.monotonic() - start
        last_save = time.monotonic()
        save_sequence += 1

    try:
        if saved:
            # Load before rotating either slot: source can be slot-0 itself.
            model = PPO.load(source / 'learner.zip', device='cpu')
            validate_model_observation(model)
            model.set_env(PolicySpace())
            torch.set_rng_state(torch.load(source / 'torch_rng.pt', weights_only=True))
        else:
            model = PPO('MlpPolicy', PolicySpace(), seed=config['seed'], device='cpu', n_steps=256,
                batch_size=64, gamma=1., gae_lambda=.95, ent_coef=.01, target_kl=.03,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
            model.decision_feature_names = FEATURE_NAMES
            model.decision_feature_version = FEATURE_VERSION
        model.decision_mode = 'bump'
        model.decision_fire_plan_rule = 'commit_until_shot'
        model.decision_cooldown_rule = 'reject_early_proposal_replan'
        model.set_logger(configure(str(run / 'metrics' / str(time.time_ns())), ['csv']))
        policy = deepcopy(model.policy).to('cpu')
        policy.set_training_mode(False)
        policy.requires_grad_(False)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(cloudpickle.dumps(policy), str(stop)))
        log_path = run / 'episodes.jsonl'
        if saved:
            trim_episode_results(log_path, saved['episodes'])
        log = log_path.open('a', encoding='utf-8')
        checkpoint(pool_changed=True)
        opponent_views = materialize(state['opponents'], model_root)
        print(f'Single learner / CPU {workers} workers / generation every {config["generationGames"]} games / recent {config["retainedGenerations"]}', flush=True)
        while state['episodes'] < config['episodes']:
            if stop.exists():
                status = 'stopped'
                break
            model.policy.set_training_mode(False)
            weights = cloudpickle.dumps({k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()})
            # A batch cannot cross a generation boundary with stale opponents.
            boundary = config['generationGames'] - state['episodes'] % config['generationGames']
            count = min(config['batchEpisodes'], config['episodes'] - state['episodes'], boundary)
            started = time.monotonic()
            futures = [pool.submit(collect, (config['seed'] + state['episodes'] + j, config, opponent_views, weights)) for j in range(count)]
            records = []
            for future in futures:
                while True:
                    try:
                        result = future.result(timeout=1.)
                        break
                    except FutureTimeout:
                        progress()
                if result is None:
                    status = 'stopped'
                    for pending in futures:
                        pending.cancel()
                    break
                transitions, row = result
                records.extend(transitions)
                state['episodes'] += 1
                row['episode'] = state['episodes']
                row['learnerGeneration'] = state['generation']
                write_episode_result(log, row, detailed=record_episodes)
                state['wins' if row['winner'] == 0 else 'losses' if row['winner'] == 1 else 'draws'] += 1
                progress()
            log.flush()
            state['timings']['collectionSeconds'] += time.monotonic() - started
            started = time.monotonic()
            if records:
                buffer = BudgetRolloutBuffer(len(records), model.observation_space, model.action_space,
                    device='cpu', gamma=1., gae_lambda=.95, n_envs=1)
                for obs, action, reward, first, value, logprob in records:
                    buffer.add(obs, np.array([action]), np.array([reward]), np.array([first]),
                               torch.tensor([value]), torch.tensor([logprob]))
                buffer.compute_returns_and_advantage(torch.zeros(1), np.array([True]))
                model.rollout_buffer = buffer
                model.num_timesteps += len(records)
                model._current_progress_remaining = 1 - state['episodes'] / config['episodes']
                normalize_advantage = model.normalize_advantage
                if len(records) == 1:
                    model.normalize_advantage = False
                try:
                    model.train()
                finally:
                    model.normalize_advantage = normalize_advantage
                model.logger.dump(step=state['episodes'])
                state['updates'] += 1
            state['timings']['updateSeconds'] += time.monotonic() - started
            changed = state['episodes'] > 0 and state['episodes'] % config['generationGames'] == 0 and state['updates'] > 0
            if changed:
                number = state['generation'] + 1
                state['opponents'] = publish(model, model_root, number, state['episodes'],
                    state['opponents'], config['retainedGenerations'])
                state['generation'] = number
                opponent_views = materialize(state['opponents'], model_root)
                with (run / 'generations.jsonl').open('a', encoding='utf-8') as history:
                    history.write(json.dumps(dict(episode=state['episodes'], generation=number,
                        opponents=state['opponents'], bootstrapRemoved=True), ensure_ascii=False) + '\n')
                print(f'Generation {number} @ {state["episodes"]} games; opponents {[e["generation"] for e in state["opponents"]]}', flush=True)
            if changed or time.monotonic() - last_save >= config['checkpointSeconds']:
                checkpoint(pool_changed=changed)
            progress(force=True)
            if status == 'stopped':
                break
        if status == 'running':
            status = 'completed'
    except KeyboardInterrupt:
        status = 'stopped'
        stop.touch()
    except BaseException:
        status = 'failed'
        raise
    finally:
        if log is not None:
            log.close()
        if pool is not None:
            if status != 'completed':
                stop.touch()
            pool.shutdown(wait=True, cancel_futures=True)
        checkpoint()
        progress(force=True)


if __name__ == '__main__':
    main()
