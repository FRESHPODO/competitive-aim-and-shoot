"""Persistent CPU simulation processes; one frozen policy version per PPO batch."""
import os
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch

from unity_bridge.server import DualSession
from unity_bridge.fire_decider import encode_context, FireDecider
from unity_bridge.opponent_pool import scenario as opponent_scenario
from unity_bridge.training_settings import sample
from unity_bridge.competitive_rewards import sample as sample_rewards, from_kill_events

_dual = _policies = _stop = _weights = None
_opponents = {}


def initialize(policy_blob, stop_path, recording=False):
    global _dual, _policies, _stop, _weights, _opponents
    os.environ['ANS_TORCH_DEVICE'] = 'cpu'
    torch.set_num_threads(1)
    _policies = cloudpickle.loads(policy_blob)
    for policy in _policies:
        policy.set_training_mode(False)
        policy.requires_grad_(False)
    _dual = DualSession(recording=recording)
    _stop = Path(stop_path)
    _weights = None
    _opponents = {}


def collect(job):
    """Episode RNG depends on its seed, never on worker assignment or completion order."""
    global _weights
    seed, settings, rewards_config, presets, learning_enabled, timeout, weights = job[:7]
    opponent = job[7] if len(job)>7 else None
    if _stop.exists():
        return None
    began = time.monotonic()
    fixed = None
    if opponent is not None:
        key = (opponent['model'], opponent['sha256'])
        if key not in _opponents:
            _opponents[key] = FireDecider(opponent['model'])
            _opponents[key].model.policy.requires_grad_(False)
        fixed = _opponents[key]
        learning_enabled = dict(learning_enabled, B=False)
        settings, presets = opponent_scenario(settings, presets, opponent)
    # Loading a model must not perturb the learner's episode-specific RNG.
    torch.manual_seed(seed % (2**63))
    if weights != _weights:
        for policy, state in zip(_policies, cloudpickle.loads(weights)):
            policy.load_state_dict(state)
        _weights = weights
    rng = np.random.default_rng(seed)
    request = sample(settings, rng)
    reward_values = sample_rewards(rewards_config, rng)
    request.update(playerPresets=presets, rewardValues=reward_values)
    _dual.round = seed
    _dual.reset(request)
    _dual.inactivity_limit_ms = timeout * 1000
    records = [[], []]
    counts = [0, 0]

    class Gate:
        def __init__(self, index):
            self.index = index

        def decide(self, now, context):
            i = self.index
            if i==1 and fixed is not None:
                counts[i]+=1
                return fixed.decide(now,context)
            obs = encode_context(now, context)
            with torch.inference_mode():
                tensor, _ = _policies[i].obs_to_tensor(obs)
                action, value, logprob = _policies[i](tensor, deterministic=False)
            a = int(action.item())
            counts[i] += 1
            if learning_enabled['AB'[i]]:
                records[i].append([obs, a, 0., len(records[i]) == 0,
                                   float(value.item()), float(logprob.item())])
            return dict(approved=bool(a), reason='joint_training', snapshotTimeMs=context['time'])

    for i, shooter in enumerate(_dual.shooters):
        shooter.decider = Gate(i)
        shooter.decision_mode = getattr(fixed.model,'decision_mode','candidate') if i==1 and fixed is not None else 'bump'
    while _dual.time < timeout * 1000 and _dual.combat.hp > 0 and not _dual.draw:
        if _stop.exists():
            return None
        _dual.step(deadline_ms=timeout * 1000)
    rewards = from_kill_events(_dual.combat.events, reward_values)
    for i in range(2):
        if records[i]:
            records[i][-1][2] = rewards[i]
    killers = {e['agent'] for e in _dual.combat.events if e.get('killed')}
    winner = next(iter(killers)) if len(killers) == 1 else None
    outcome = 'kill' if winner is not None else ('simultaneous_draw' if _dual.combat.hp <= 0 else
                                               'no_shots_draw' if _dual.draw else 'time_limit_draw')
    row = dict(seed=seed, outcome=outcome, winner=winner, timeMs=_dual.time, rewards=rewards,
               rewardValues=reward_values, decisions=counts,
               shots=[sum(e['agent'] == i for e in _dual.combat.events) for i in range(2)],
               settings=request, learningEnabled=learning_enabled, workerPid=os.getpid(),
               simulationSeconds=time.monotonic() - began,
               simulationStarted=began,simulationFinished=time.monotonic())
    if opponent is not None:
        row['opponentPool']={k:opponent.get(k) for k in ('index','name','side','presetId','weaponPreset','sha256')}
    return records, row
