"""Gym adapter that pauses the real simulator at each A fire candidate.

All Aim-and-Shoot calls stay on a single worker thread. Queues hand off only
observations/actions; no simulator copy, speculative RNG, or synthetic reward.
"""
import queue
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class _Closed(Exception):
    pass


def sample_scenario(rng):
    """Default-condition mixture plus domain randomization of every supplied input."""
    from unity_bridge.server import DEFAULTS
    if rng.random() < .4:
        profiles=[DEFAULTS.copy(),DEFAULTS.copy()]
        side=int(rng.integers(2))
        scale=float(rng.choice([.5,1.5]))
        profiles[side][:4]=[v*scale for v in profiles[side][:4]]
        return dict(decisionAgent=None, profiles=profiles,
                    targetHp=1000., minDamage=100., maxDamage=300., targetRadius=5.,
                    targetSpeed=float(rng.uniform(0,40)), criticalRadius=.01,
                    shotIntervalsMs=[150,150], shotSpreadDeg=[0.,0.], randomTargetSpawn=True)
    profiles=[]
    for _ in range(2):
        profiles.append(list(rng.uniform(0,.5,4)) + list(rng.uniform(1,64,2)) +
                        list(rng.uniform(5,95,2)))
    minimum=float(np.exp(rng.uniform(np.log(1),np.log(1000))))
    maximum=minimum+float(np.exp(rng.uniform(np.log(1),np.log(10000))))
    return dict(decisionAgent=None, profiles=profiles,
                targetHp=float(rng.uniform(1,1000)), minDamage=minimum, maxDamage=maximum,
                targetRadius=float(rng.uniform(.3,20)), targetSpeed=float(rng.uniform(0,40)),
                shotIntervalsMs=list(rng.integers(50,2001,2).astype(float)),
                shotSpreadDeg=[float(rng.choice([0.,.2,1.,5.,20.,180.])) for _ in range(2)],
                criticalRadius=float(rng.choice([0.,.01,.05,.1,.25,.5])), randomTargetSpawn=True)


class CandidateEnv(gym.Env):
    def __init__(self, encode, observation_size, *, seed=1234, max_ms=10000., scenario=None,
                 settings=None, learner_index=0):
        self.encode = encode
        self.base_seed = seed
        self.max_ms = max_ms
        self.scenario = scenario
        self.settings = settings
        self.learner_index = learner_index
        self.opponent_path = None
        self.opponent = None

        if settings is not None:
            from unity_bridge.training_settings import validate
            validate(settings)
        self.observation_space = spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = reject, 1 = approve
        self.worker = None
        self.last_time = 0.
        self.episode = 0
        self.dual = None

    def _run(self):
        try:
            from unity_bridge.server import DualSession
            if self.dual is None:
                self.dual = DualSession()
            dual = self.dual
            rng = np.random.default_rng(self.base_seed + self.episode)
            if self.settings is None:
                request = sample_scenario(rng)
            else:
                from unity_bridge.training_settings import sample
                request = sample(self.settings,rng)
            if self.scenario is not None:
                request.update(self.scenario)
                request['decisionAgent'] = None
                request['decisionEnabled'] = [False, False]
            # Round seeds must vary across resets, including target motion.
            dual.round = self.base_seed + self.episode
            dual.reset(request)
            env = self

            class Gate:
                def decide(self, now, context):
                    obs = env.encode(now, context)
                    env.events.put(('candidate', obs, now, {}))
                    action = env.actions.get()
                    if action is None:
                        raise _Closed()
                    return dict(approved=bool(action), reason='training_action',
                                snapshotTimeMs=context['time'])

            dual.shooters[self.learner_index].decider = Gate()
            if self.opponent_path is not None:
                from unity_bridge.fire_decider import FireDecider
                if self.opponent is None or str(self.opponent.model_path)!=str(self.opponent_path):
                    self.opponent=FireDecider(self.opponent_path)
                dual.shooters[1-self.learner_index].decider=self.opponent
            while dual.time < self.max_ms:
                packet = dual.step()
                if packet['done']:
                    reward = dual.combat.scores[self.learner_index] - dual.combat.scores[1-self.learner_index]
                    self.events.put(('terminal', reward, dual.time, dict(scores=dual.combat.scores)))
                    return
            # A time limit is an adjudicated draw, not an unobserved future
            # outcome whose value should be bootstrapped by PPO.
            self.events.put(('terminal', 0., dual.time, dict(outcome='time_limit_draw')))
        except _Closed:
            pass
        except BaseException as error:
            self.events.put(('error', error, 0., {}))

    def _receive(self):
        kind, data, time, info = self.events.get(timeout=180)
        if kind == 'error':
            raise RuntimeError('Simulator worker failed') from data
        return kind, data, time, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        if seed is not None:
            self.base_seed, self.episode = seed, 0
        for _ in range(32):
            self.episode += 1
            self.events, self.actions = queue.Queue(), queue.Queue()
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()
            kind, obs, self.last_time, info = self._receive()
            if kind == 'candidate':
                return obs, info
            # No A action existed before B ended the round. It contributes no
            # policy transition; evaluation still counts these whole rounds.
            self.close()
        raise RuntimeError('32 rounds ended before any A decision candidate')

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Expected reject=0 or approve=1')
        self.actions.put(int(action))
        kind, data, time, info = self._receive()
        # Only the final killing blow earns reward. No hit/miss/time shaping.
        reward = 0.
        self.last_time = time
        if kind == 'candidate':
            return data, reward, False, False, info
        return (np.zeros(self.observation_space.shape, np.float32), reward+float(data),
                kind == 'terminal', kind == 'timeout', info)

    def close(self):
        if self.worker is not None:
            self.actions.put(None)
            self.worker.join(timeout=30)
            if self.worker.is_alive():
                raise RuntimeError('Simulator worker did not stop')
            self.worker = None
