"""One trainable request-only policy versus a bounded cache of frozen opponents."""
import os
import time
from pathlib import Path
import cloudpickle
import numpy as np
import torch
from unity_bridge.server import DualSession
from unity_bridge.reward_generation.decider import encode_context, FireDecider
from unity_bridge.reward_generation.config import sample_scenario
from unity_bridge.generation_pool import choose
from unity_bridge.training_settings import sample
from unity_bridge.reward_generation.rewards import sample as sample_rewards, breakdown

_dual = _policy = _stop = _weights = None
_opponents = {}


def initialize(policy_blob, stop_path):
    global _dual, _policy, _stop, _weights, _opponents
    os.environ['ANS_TORCH_DEVICE'] = 'cpu'
    torch.set_num_threads(1)
    _policy = cloudpickle.loads(policy_blob)
    _policy.set_training_mode(False)
    _policy.requires_grad_(False)
    _dual = DualSession(recording=False)
    _stop = Path(stop_path)
    _weights = None
    _opponents = {}


def collect(job):
    global _weights
    seed, config, opponents, weights = job
    if _stop.exists():
        return None
    began = time.monotonic()
    # An expired opponent must not survive in a persistent worker's cache.
    active = {entry['sha256'] for entry in opponents}
    for key in list(_opponents):
        if key not in active:
            del _opponents[key]
    rng = np.random.default_rng(seed)
    opponent = choose(opponents, rng)
    fixed = None
    if opponent['kind'] == 'frozen_policy':
        key = opponent['sha256']
        if key not in _opponents:
            _opponents[key] = FireDecider(opponent['model'])
            _opponents[key].model.policy.requires_grad_(False)
        fixed = _opponents[key]
    # Model construction/loading cannot perturb this game's policy RNG.
    torch.manual_seed(seed % (2**63))
    if weights != _weights:
        _policy.load_state_dict(cloudpickle.loads(weights))
        _weights = weights
    settings, presets = sample_scenario(config, rng)
    request = sample(settings, rng)
    reward_values = sample_rewards(config['rewardConfig'], rng)
    request.update(playerPresets=presets, rewardValues=reward_values)
    _dual.round = seed
    _dual.reset(request)
    _dual.inactivity_limit_ms = config['timeout'] * 1000
    records = []
    counts = [0, 0]
    approvals = [0, 0]

    class Gate:
        def __init__(self, index):
            self.index = index

        def decide(self, now, context):
            i = self.index
            counts[i] += 1
            context = dict(context, own_reward=reward_values['AB'[i]])
            if i == 1:
                decision = fixed.decide(now, context) if fixed else dict(
                    approved=True, reason='bootstrap_always_approve', snapshotTimeMs=context['time'])
            else:
                obs = encode_context(now, context)
                with torch.inference_mode():
                    tensor, _ = _policy.obs_to_tensor(obs)
                    action, value, logprob = _policy(tensor, deterministic=False)
                a = int(action.item())
                records.append([obs, a, 0., len(records) == 0, float(value.item()), float(logprob.item())])
                decision = dict(approved=bool(a), reason='generational_learner', snapshotTimeMs=context['time'])
            approvals[i] += int(decision['approved'])
            return decision

    for i, shooter in enumerate(_dual.shooters):
        shooter.decider = Gate(i)
        shooter.decision_mode = 'bump'
    while _dual.time < config['timeout'] * 1000 and _dual.combat.hp > 0 and not _dual.draw:
        if _stop.exists():
            return None
        _dual.step(deadline_ms=config['timeout'] * 1000)
    components = breakdown(_dual.combat.events, _dual.combat.maximum_hp, reward_values)
    rewards = [item['total'] for item in components]
    if records:
        records[-1][2] = rewards[0]
    killers = {e['agent'] for e in _dual.combat.events if e.get('killed')}
    killer = next(iter(killers)) if len(killers) == 1 else None
    row = dict(seed=seed, outcome='kill' if killer is not None else 'draw', winner=killer,
               timeMs=_dual.time, rewards=rewards, rewardValues=reward_values, rewardComponents=components,
               decisions=counts, approvals=approvals,
               shots=[sum(e['agent'] == i for e in _dual.combat.events) for i in range(2)],
               settings=request, playerPresets=presets,
               opponent={k: v for k, v in opponent.items() if k != 'model'},
               learningEnabled=dict(A=True, B=False), workerPid=os.getpid(),
               simulationSeconds=time.monotonic() - began)
    return records, row
