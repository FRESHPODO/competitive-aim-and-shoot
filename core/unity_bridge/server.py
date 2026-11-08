"""Loopback-only bridge: one trained-policy decision per /step request."""
import argparse
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from src.agent.simulator import AimandShootSimulator
from src.agent.task import AimandShootSpiderShotTask
from src.configs.constants import AGENT_STATE_ALIAS as AS
from src.configs.constants import TINTERVAL
from src.utils.mymath import Convert, point_about_ray
from unity_bridge.fire_decider import FireDecider, MODEL_PATH
from src.agent.module_aim import Aim
from unity_bridge.aim_belief import estimate_candidate
from unity_bridge.competitive_rewards import alpha_from_values, values_from_alpha
from unity_bridge.gpu_budget import gpu_work

PARAM_KEYS = ('param_motor_noise', 'param_position_noise', 'param_speed_noise',
              'param_clock_noise', 'param_succ_reward', 'param_fail_penalty',
              'param_reward_decay', 'param_penalty_decay')
DEFAULTS = [.20, .08, .04, .04, 40, 12, 40, 45]


def sample_bullet_direction(az,el,spread_deg,rng):
    """Uniform solid-angle sample inside the aim-centered cone (half-angle)."""
    if spread_deg==0:
        return float(az),float(el),0.
    forward=Convert.sphr2cart_scalar(az,el)
    reference=np.array([0.,1.,0.]) if abs(forward[1])<.99 else np.array([1.,0.,0.])
    right=np.cross(forward,reference);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    cos_theta=rng.uniform(np.cos(np.radians(spread_deg)),1.)
    phi=rng.uniform(0.,2*np.pi)
    direction=cos_theta*forward+np.sqrt(max(0.,1-cos_theta*cos_theta))*(np.cos(phi)*right+np.sin(phi)*up)
    angles=Convert.cart2sphr_scalar(*direction)
    return float(angles[0]),float(angles[1]),float(np.degrees(np.arccos(np.clip(cos_theta,-1,1))))


class Session:
    def __init__(self, *, recording=True):
        self.recording = recording
        self.sim = AimandShootSimulator('ijhcs_default', 'latest')
        self.agent = self.sim.agent
        self.episode = 0
        self.done = True
        self.rng_state = np.random.RandomState(1).get_state()

    def reset(self, request):
        params = request.get('parameters', DEFAULTS)
        if len(params) != 8 or not np.isfinite(params).all():
            raise ValueError('Eight finite parameters required')
        for i, val in enumerate(params):
            lo, hi = (0, .5) if i < 4 else ((1, 64) if i < 6 else (5, 95))
            if not lo <= val <= hi:
                raise ValueError('Parameter out of range')
        self.agent.player_state_preset.update(dict(zip(PARAM_KEYS, params)))
        speed = float(request.get('targetSpeed', 12))
        radius = float(request.get('targetRadius', 5))
        if not 0 <= speed <= 40 or not .3 <= radius <= 20:
            raise ValueError('Target condition out of range')
        self.agent.game_env_preset.update(target_speed_deg_s=speed, target_radius_deg=radius)
        outer_rng = np.random.get_state()
        try:
            self.obs, _ = self.agent.reset(seed=int(request.get('seed', self.episode + 1)) % (2**32))
            self.rng_state = np.random.get_state()
        finally:
            np.random.set_state(outer_rng)
        self.initial_camera = self.agent.init_game_env['camera_azel_deg'].copy()
        self.episode += 1
        self.done = False
        self.last_time = 0.
        return self.packet(None)

    def step(self):
        if self.done:
            raise ValueError('Start a new episode first')
        outer_rng = np.random.get_state()
        try:
            np.random.set_state(self.rng_state)
            with gpu_work():action, _ = self.sim.model.predict(self.obs, deterministic=True)
            self.obs, _, terminated, truncated, info = self.agent.step(action)
            self.rng_state = np.random.get_state()
        finally:
            np.random.set_state(outer_rng)
        self.done = bool(terminated or truncated)
        return self.packet(info)

    def packet(self, info):
        agent = self.agent
        end = max(0., float(info['time'])) if info else max(0., float(agent.task_timer))
        if self.recording == 'replay':
            # Record the actual camera trajectory without building live UI packets.
            # Decisions, bullets and cooldown transitions remain in LiveShooter logs.
            hp, _, ht = agent.get_hand_trajectory()
            cp = self.initial_camera + hp * agent.game_env.hand_sensi_deg_per_mm
            order = np.argsort(ht, kind='stable')
            ts = np.asarray(ht)[order]; cp = np.asarray(cp)[order]
            times = np.unique(np.r_[np.arange(self.last_time, end, 10.), end])
            az = np.interp(times, ts, cp[:, 0]); el = np.interp(times, ts, cp[:, 1])
            frames = [dict(timeMs=float(t), cameraAz=float(a), cameraEl=float(e))
                      for t,a,e in zip(times,az,el)]
            self.last_time = end
            return dict(frames=frames, timeMs=end)
        if not self.recording:
            # Preserve the exact camera endpoints used by combat observations.
            # Skip target/gaze interpolation and all display packet fields.
            hp, _, ht = agent.get_hand_trajectory()
            cp = self.initial_camera + hp * agent.game_env.hand_sensi_deg_per_mm
            order = np.argsort(ht, kind='stable')
            ts = np.asarray(ht)[order]
            cp = np.asarray(cp)[order]
            frames = [dict(timeMs=float(t),
                           cameraAz=float(np.interp(t, ts, cp[:, 0])),
                           cameraEl=float(np.interp(t, ts, cp[:, 1])))
                      for t in (self.last_time, end)]
            self.last_time = end
            return dict(frames=frames, timeMs=end)
        times = np.unique(np.r_[np.arange(self.last_time, end, 10.), end])
        hp, _, ht = agent.get_hand_trajectory()
        cp = self.initial_camera + hp * agent.game_env.hand_sensi_deg_per_mm
        gp, gt = agent.get_gaze_trajectory()
        # Source hand trajectories are episode-relative, initial hand position is zero.
        def interp(values, ts, t):
            idx = np.argsort(ts, kind='stable')
            ts, values = np.asarray(ts)[idx], np.asarray(values)[idx]
            return [float(np.interp(t, ts, values[:, j])) for j in range(2)]
        frames = []
        initial = agent.init_game_env
        orbit_axis = Convert.sphr2cart_scalar(*initial['target_orbit_axis_deg'])
        for t in times:
            cam = interp(cp, ht, t)
            # Source shoot_index is shoot_timing-1: preserve that exact endpoint.
            sample_t = max(0., t-1) if self.done and info and not info['is_truncated'] and t == end else t
            world = point_about_ray(initial['target_pos_world'],
                                   float(initial['target_speed_deg_s'])*sample_t/1000,
                                   orbit_axis)
            target = Convert.cart2sphr_scalar(*world)[:2]
            gaze = interp(gp, gt, t)
            frames.append(dict(timeMs=float(t), cameraAz=cam[0], cameraEl=cam[1],
                               targetAz=float(target[0]), targetEl=float(target[1]),
                               gazeX=gaze[0], gazeY=gaze[1]))
        state = info['step_state'] if info else {}
        result = dict(episode=self.episode, frames=frames, done=self.done,
                      truncated=bool(info and info['is_truncated']),
                      hit=bool(info and info['is_success']),
                      timeMs=end, radius=float(agent.game_env.target_radius_deg),
                      errorDeg=float(state.get(AS.SHOOT_ERROR_DEG, 0) or 0),
                      normalizedError=float(state.get(AS.SHOOT_ERROR_MM, 0) or 0)/max(float(agent.game_env.target_radius_mm),1e-6),
                      fireCommitted=bool(agent.waiting_motor_actual_movement_pos is not None and not self.done),
                      action=json_safe(state.get(AS.ACTION, {})), error='')
        self.last_time = end
        return result


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class TargetCombat:
    """Authoritative timestamp-ordered damage, including unbiased simultaneous kills."""
    def __init__(self, hp=1000., minimum=100., maximum=500., critical=.01,weapons=None):
        if not all(np.isfinite([hp,minimum,maximum,critical])) or not (1<=hp<=1000 and 0<=minimum<=maximum and 0<=critical<=.5):
            raise ValueError('Invalid target damage settings')
        self.maximum_hp=hp
        self.hp=hp
        self.minimum=minimum
        self.maximum=maximum
        self.critical=critical
        self.weapons=weapons or [dict(minimum=minimum,maximum=maximum,instantKill=True) for _ in range(2)]
        if len(self.weapons)!=2:raise ValueError('Two weapon configurations required')
        for w in self.weapons:
            if not np.isfinite([w['minimum'],w['maximum']]).all() or not 0<=w['minimum']<=w['maximum'] or type(w['instantKill']) is not bool:
                raise ValueError('Invalid per-agent weapon damage')
        self.events=[]
        self.scores=[0.,0.]
        self.destroyed_at=None

    def resolve(self, shots):
        shots=sorted(shots,key=lambda p:p['timeMs'])
        while shots:
            time=shots[0]['timeMs']
            group=[s for s in shots if abs(s['timeMs']-time)<1e-6]
            shots=shots[len(group):]
            if self.hp<=0: break
            events=[]
            for shot in group:
                ratio=max(0.,shot['normalizedError'])
                critical=shot['hit'] and ratio<=self.critical
                weapon=self.weapons[shot['agent']]
                instant=bool(critical and weapon['instantKill'])
                damage=(self.hp if instant else weapon['minimum']+(weapon['maximum']-weapon['minimum'])*(1-min(1.,ratio))**2) if shot['hit'] else 0.
                events.append(dict(agent=shot['agent'],timeMs=time,damage=float(damage),critical=bool(critical),instantKill=instant,
                                   hit=bool(shot['hit']),normalizedError=float(ratio),scoreDelta=0.,killed=False,
                                   shotAz=shot.get('shotAz',0.),shotEl=shot.get('shotEl',0.),
                                   spreadAngle=shot.get('spreadAngle',0.)))
            self.hp=max(0.,self.hp-sum(e['damage'] for e in events))
            if self.hp<=0:
                self.destroyed_at=time
                contributors=[e for e in events if e['damage']>0]
                critical_contributors=[e for e in contributors if e['instantKill']]
                if critical_contributors:
                    contributors=critical_contributors
                for event in contributors:
                    event['killed']=True
                    event['scoreDelta']=1./len(contributors)
                    self.scores[event['agent']]+=event['scoreDelta']
            for event in events:
                event['hpAfter']=self.hp
            self.events.extend(events)


class SharedMotion:
    """Use the original task's reset/orbit_target, once for both agents."""
    def __init__(self,speed=0.,direction=0.,config=None,radius=1.5,seed=1,
                 random_spawn=False,random_direction=False):
        if not np.isfinite([speed,direction]).all() or not 0<=speed<=40 or not 0<=direction<=360:
            raise ValueError('Invalid shared target motion')
        self.task=AimandShootSpiderShotTask(config)
        initial=self.task.reset(camera_azel_deg=[0.,0.],
            target_pos_world=None if random_spawn else np.array([20.,2.,0.])/np.sqrt(404.),
            target_motion_dir_deg=None if random_direction else float(direction),
            target_speed_deg_s=float(speed),target_radius_deg=float(radius),
            rng=np.random.default_rng(seed))
        self.speed=float(initial['target_speed_deg_s'])
        self.initial=initial['target_pos_world'].copy()
        self.axis_angles=initial['target_orbit_axis_deg'].copy()
        self.axis=Convert.sphr2cart_scalar(*self.axis_angles)
        self.direction=float(initial['target_motion_dir_deg'])

    def position(self,time_ms):
        return self.task.orbit_target(float(time_ms),unit='ms',inplace=False)[0]


class LiveShooter:
    """Plan continuously; cooldown constrains bullet emission, never plan admission."""
    def __init__(self,index,*,recording=True):
        self.session=Session(recording=recording)
        self.index=index
        self.decider=None
        self.decision_mode='candidate'
        self.decision_context=None
        agent=self.session.agent
        original_unpack=agent._unpack_action
        original_plan=agent._plan_hand_movement
        original_check=agent._check_shot_result

        def unpack(action_normalized):
            action=original_unpack(action_normalized)
            pending=agent.waiting_motor_actual_movement_pos is not None or self.commit_started is not None
            self.decision_log.append(dict(timeMs=float(agent.task_timer),requested=bool(action['kc']),
                allowed=True,weaponReady=agent.task_timer>=self.ready_at,actual=False,
                pendingPlan=pending,ppoApproved=None,ppoBlocked=False,cooldownBlocked=False))
            return action

        def plan(state,action):
            row=self.decision_log[-1]
            pending=agent.waiting_motor_actual_movement_pos is not None
            original_limit=agent.config['task']['time_limit_ms']
            agent.config['task']['time_limit_ms']=max(original_limit,agent.task_timer+5000)
            try:
                if pending or not action['kc']:
                    original_plan(state,action)
                    return
                now=float(agent.task_timer)
                delay=max(round((TINTERVAL.BUMP+action['tc']*action['th'])*state[AS.CLOCK_NOISE]),1)
                row.update(plannedShotTimeMs=now+delay,effectiveShotTimeMs=now+delay)
                if now+delay<self.ready_at:
                    # Reject this proposal before commitment, and keep tracking.
                    # A later BUMP makes a fresh proposal; no delayed shot queue.
                    row.update(allowed=False,cooldownBlocked=True)
                    action['kc']=0
                    original_plan(state,action)
                    return
                # Preview the exact source plan, but expose only its ideal path.
                # A rejected preview restores RNG before ordinary tracking.
                rng=np.random.get_state()
                determined=agent.shooting_motor_plan_determined
                original_plan(state,action)
                if agent.waiting_motor_actual_movement_pos is None:
                    return  # Source reaction/motor-planning wait, not a PPO choice.
                approved=True
                if self.decider is not None and self.decision_context is not None:
                    if not self.decision_context['shared_attack_conditions']:
                        raise ValueError('외곽/중앙 피해, 즉사 여부, 쿨다운 길이는 A/B 공통 게임 조건이어야 합니다.')
                    belief=estimate_candidate(agent,state,delay,self.decision_context['game'],
                        self.decision_context['hp'],self.decision_context['critical'])
                    context=dict(self.decision_context,
                        own=dict(self.decision_context['own'],ready_at=self.ready_at),
                        belief=belief,proposal=dict(th=float(action['th']),tc=float(action['tc']),delay_ms=float(delay)))
                    decision=self.decider.decide(now,context)
                    approved=bool(decision['approved'])
                    row.update(ppoApproved=approved,ppoBlocked=not approved)
                    self.strategy_log.append(dict(timeMs=now,decisionStage='request',
                        observedHp=context['hp'],opponentReadyInMs=max(0.,context['opponent']['ready_at']-now),
                        originalRequested=True,pendingPlan=False,ppoBlocked=not approved,
                        cooldownBlocked=False,estimatedDamage=belief['damage'],
                        estimatedNormalizedError=belief['normalized_error'],
                        plannedShotTimeMs=now+delay,**decision))
                if approved:
                    self.commit_started=now
                    row['actual']=True
                else:
                    np.random.set_state(rng)
                    agent.waiting_motor_planned_movement_pos=None
                    agent.waiting_motor_planned_movement_vel=None
                    agent.waiting_motor_actual_movement_pos=None
                    agent.waiting_motor_actual_movement_vel=None
                    agent.shooting_motor_plan_determined=determined
                    action['kc']=0
                    original_plan(state,action)
            finally:
                agent.config['task']['time_limit_ms']=original_limit

        def check(state,action):
            if self.commit_started is None: agent.shoot_timing=1e12
            else:
                planned_at=float(agent.task_timer+agent.shoot_timing)
                if planned_at<self.ready_at-1e-6:
                    raise RuntimeError('Committed shot precedes cooldown; proposal validation failed')
                self.decision_log[-1].update(plannedShotTimeMs=planned_at,effectiveShotTimeMs=planned_at)
            original_check(state,action)
            if state[AS.DONE]:
                event_time=float(state[AS.SHOOT_MOMENT_MS])
                segment=state[AS.HTRAJ_SEG_SHOOT_POS]
                camera=agent.game_env.camera_azel_deg+agent.game_env.hand_sensi_deg_per_mm*(segment[-1]-segment[0])
                self.record_shot(event_time,float(camera[0]),float(camera[1]))
                if self.commit_started is not None:
                    self.commit_spans.append((self.commit_started,event_time))
                self.commit_started=None
                self.last_shot=event_time
                self.ready_at=event_time+self.interval
                # Complete only this shot command. Keep perception, gaze, hand
                # position/velocity and the remainder of the current motor segment.
                agent.waiting_motor_planned_movement_pos=None
                agent.waiting_motor_planned_movement_vel=None
                agent.waiting_motor_actual_movement_pos=None
                agent.waiting_motor_actual_movement_vel=None
                agent.shooting_motor_plan_determined=False
                agent.shoot_timing=1e12
                state[AS.DONE]=False
                state[AS.TRUNC]=False

        def continue_tracking(state,action):
            # Shared-target destruction (DualSession), not a source single-shot
            # termination or its 3-second limit, controls the round lifecycle.
            return agent.get_np_obs(),0.,False,False,dict(time=agent.task_timer,
                result=False,is_success=False,is_truncated=False,step_state=state)

        agent._unpack_action=unpack
        agent._plan_hand_movement=plan
        agent._check_shot_result=check
        agent._terminate_or_truncate=continue_tracking

    def reset(self,profile,radius,seed,interval,motion=None,spread=0.):
        self.motion=motion or SharedMotion()
        self.spread=float(spread)
        self.weapon_rng=np.random.default_rng(seed+29083)
        self.profile=profile
        self.radius=radius
        self.seed=seed
        self.interval=interval
        self.offset=0.
        self.frames=[]
        self.shots=[]
        self.ready_at=-1e12
        self.last_shot=-1e12
        self.packet=None
        self.commit_started=None
        self.commit_spans=[]
        self.decision_log=[]
        self.strategy_log=[]
        self.decision_context=None
        agent=self.session.agent
        agent.game_env_preset.update(camera_azel_deg=[0.,0.],target_pos_world=self.motion.position(0),
                                    target_orbit_axis_deg=self.motion.axis_angles)
        agent.player_state_preset.update(hand_reaction_time=200.,gaze_reaction_time=150.,
                                         gaze_position=[0.,0.],head_position=[0.,77.9,575.])
        self.append(self.session.reset(dict(parameters=profile,targetSpeed=self.motion.speed,
                                            targetRadius=radius,seed=seed)))

    def record_shot(self,time,aim_az,aim_el):
        game=self.session.agent.game_env
        shot_az,shot_el,deviation=sample_bullet_direction(aim_az,aim_el,self.spread,self.weapon_rng)
        projected=Convert.game2monitor(np.zeros(3),np.array([shot_az,shot_el]),self.motion.position(time),
                                       game.camera_fov_deg,game.monitor_half_size_mm)
        ratio=float(np.linalg.norm(projected))/max(float(game.target_radius_mm),1e-6)
        facing=np.dot(Convert.sphr2cart_scalar(shot_az,shot_el),self.motion.position(time))>0
        self.shots.append(dict(agent=self.index,timeMs=time,hit=bool(facing and ratio<=1.),
            normalizedError=ratio,shotAz=shot_az,shotEl=shot_el,spreadAngle=deviation))

    def append(self,packet):
        self.packet=packet
        self.frames.extend(dict(f) for f in packet['frames'])
        self.latest=packet['timeMs']
        if not self.session.recording:
            self.frames=self.frames[-2:]
            self.strategy_log=self.strategy_log[-1:]
            self.decision_log=self.decision_log[-1:]
            self.commit_spans=self.commit_spans[-1:]

    def advance(self,horizon):
        guard=0
        while self.latest<horizon:
            guard+=1
            if guard>100: raise RuntimeError('Agent failed to advance')
            self.append(self.session.step())

    def view(self,start,end):
        def sample(t):
            ts=[f['timeMs'] for f in self.frames]
            result=dict(timeMs=float(t))
            for key in ('cameraAz','cameraEl','targetAz','targetEl','gazeX','gazeY'):
                result[key]=float(np.interp(t,ts,[f[key] for f in self.frames]))
            angles=Convert.cart2sphr_scalar(*self.motion.position(t))
            result['targetAz'],result['targetEl']=float(angles[0]),float(angles[1])
            result['fireCommitted']=bool(any(a<=t<b for a,b in self.commit_spans) or
                (self.commit_started is not None and self.commit_started<=t))
            result['cooldown']=bool(any(e['timeMs']<=t<e['timeMs']+self.interval for e in self.shots))
            decisions=[d for d in self.strategy_log if d['timeMs']<=t]
            rejected=[d for d in decisions if not d['approved']]
            last=decisions[-1] if decisions else None
            current_decision=last if last and t<last['timeMs']+TINTERVAL.BUMP else None
            result['strategyHasDecision']=current_decision is not None
            result['strategyBlocked']=bool(current_decision and not current_decision['approved'])
            result['strategyLastRejectedMs']=float(rejected[-1]['timeMs']) if rejected else -1e12
            result['strategyRejected']=len(rejected)
            result['strategyApproveProbability']=float(current_decision.get('approveProbability',0.)) if current_decision else 0.
            return result
        times=np.unique(np.r_[np.arange(start,end,10.),end])
        packet=dict(self.packet)
        packet['frames']=[sample(t) for t in times]
        packet['timeMs']=end
        packet['done']=False
        packet['cooldown']=self.last_shot<=end<self.ready_at
        packet['strategyEnabled']=self.decider is not None
        packet['strategyModel']=str(getattr(self.decider,'model_path','')) if self.decider is not None else ''
        packet['strategyDecisions']=[d for d in self.strategy_log if start<=d['timeMs']<end]
        packet['strategyRejected']=sum(not d['approved'] for d in self.strategy_log if d['timeMs']<end)
        return packet


class DualSession:
    """Shared-clock multi-shot combat with one authoritative stationary target."""
    def __init__(self,*,recording=True):
        self.recording=recording
        self.shooters=[LiveShooter(0,recording=recording),LiveShooter(1,recording=recording)]
        self.round=0
        self.scores=[0.,0.]
        self.combat=None
        self.bullets=[0,0]
        self.hits=[0,0]
        self.policy_decider=None
        self.policy_signature=None
        self.policy_cache={}
        self.draw=False
        self.last_activity_ms=0.
        self.inactivity_limit_ms=15000.

    def reset(self,request):
        profiles=request.get('profiles',[DEFAULTS,DEFAULTS])
        supplied_rewards=request.get('rewardValues',{})
        self.reward_values={side:values_from_alpha(alpha_from_values(supplied_rewards.get(side,{}))) for side in ('A','B')}
        if len(profiles)!=2: raise ValueError('Two profiles required')
        radius=float(request.get('targetRadius',5))
        motion_seed=self.round+1
        speed=float(request.get('targetSpeed',0))
        if request.get('randomTargetSpeed',False):
            speed_min=float(request.get('targetSpeedMin',0))
            speed_max=float(request.get('targetSpeedMax',40))
            if not np.isfinite([speed_min,speed_max]).all() or not 0<=speed_min<=speed_max<=40:
                raise ValueError('Invalid target speed range')
            speed=float(np.random.default_rng(motion_seed+7349).uniform(speed_min,speed_max))
        self.motion=SharedMotion(speed,0,
            config=self.shooters[0].session.agent.config['task'],radius=radius,seed=self.round+1,
            random_spawn=bool(request.get('randomTargetSpawn',False)),
            random_direction=True)
        intervals=([float(request['cooldown'])]*2 if 'cooldown' in request
                   else request.get('shotIntervalsMs',[150,150]))
        spreads=request.get('shotSpreadDeg',[0,0])
        if len(spreads)!=2 or not np.isfinite(spreads).all() or not all(0<=v<=180 for v in spreads):
            raise ValueError('Two valid bullet spread angles required')
        if len(intervals)!=2 or not all(np.isfinite(intervals)) or not all(50<=v<=2000 for v in intervals):
            raise ValueError('Two valid per-agent shot intervals required')
        weapons=request.get('weapons')
        if 'instantKill' in request:
            weapons=[dict(minimum=float(request.get('minDamage',100)),maximum=float(request.get('maxDamage',300)),
                          instantKill=bool(request['instantKill'])) for _ in range(2)]
        self.combat=TargetCombat(float(request.get('targetHp',1000)),float(request.get('minDamage',100)),
                                 float(request.get('maxDamage',300)),float(request.get('criticalRadius',.01)),weapons)
        self.initial_scores=self.scores.copy()
        self.initial_bullets=self.bullets.copy()
        self.initial_hits=self.hits.copy()
        self.round+=1
        self.time=0.
        self.draw=False
        self.last_activity_ms=0.
        self.event_cursor=[0,0]
        selected=request.get('decisionAgent',0)
        if selected not in (None,0,1): raise ValueError('decisionAgent must be null, 0 or 1')
        enabled=request.get('decisionEnabled',[i==selected for i in range(2)])
        model_ids=request.get('decisionModels',[request.get('decisionModel','')]*2)
        if len(enabled)!=2 or any(type(value) is not bool for value in enabled):
            raise ValueError('Two boolean decisionEnabled values required')
        if len(model_ids)!=2 or any(not isinstance(value,str) for value in model_ids):
            raise ValueError('Two decisionModels strings required')
        from unity_bridge.decision_models import resolve_model
        policies=[]
        for active,identifier in zip(enabled,model_ids):
            selected_path=resolve_model(identifier)
            policy=None
            if active and selected_path.exists():
                signature=(str(selected_path),selected_path.stat().st_mtime_ns,selected_path.stat().st_size)
                cached=self.policy_cache.get(str(selected_path))
                if cached is None or cached[0]!=signature:
                    cached=(signature,FireDecider(selected_path))
                    self.policy_cache[str(selected_path)]=cached
                policy=cached[1]
            policies.append(policy)
        self.policy_decider=policies[0]
        for i,shooter in enumerate(self.shooters):
            shooter.reset(profiles[i],radius,self.round*1000+i*100,float(intervals[i]),self.motion,float(spreads[i]))
            shooter.decider=policies[i]
        return self.response(0.,0.)

    def step(self, deadline_ms=None):
        if self.combat is None or self.combat.hp<=0 or self.draw: raise ValueError('Start a new round first')
        start=self.time
        deadline=self.last_activity_ms+self.inactivity_limit_ms
        horizon=min(self.time+100.,deadline)
        if deadline_ms is not None:
            if not np.isfinite(deadline_ms) or deadline_ms<=self.time: raise ValueError('Deadline must be after current time')
            horizon=min(horizon,float(deadline_ms))
        shots=[]
        # Freeze both public views before advancing either agent. Do not expose
        # the first agent's already-computed future to the second agent.
        views=[]
        shared=(self.combat.weapons[0]==self.combat.weapons[1]
                and self.shooters[0].interval==self.shooters[1].interval)
        game=dict(self.combat.weapons[0],radius=self.shooters[0].radius,speed=self.motion.speed,
                  cooldown=self.shooters[0].interval)
        for shooter in self.shooters:
            last=max((s['timeMs'] for s in shooter.shots if s['timeMs']<=start),default=-1e12)
            views.append(dict(ready_at=last+shooter.interval))
        for i,shooter in enumerate(self.shooters):
            own=dict(views[i],profile=list(shooter.profile[:4]))
            shooter.decision_context=dict(time=start,hp=self.combat.hp,maximum_hp=self.combat.maximum_hp,
                critical=self.combat.critical,game=game,shared_attack_conditions=shared,own=own,opponent=dict(views[1-i]),
                own_reward=dict(self.reward_values['A' if i==0 else 'B']))
        for i,shooter in enumerate(self.shooters):
            shooter.advance(horizon)
            while self.event_cursor[i]<len(shooter.shots):
                shot=shooter.shots[self.event_cursor[i]]
                if shot['timeMs']>horizon: break
                if shot['timeMs']>=deadline: break
                shots.append(shot)
                self.event_cursor[i]+=1
        self.combat.resolve(shots)
        self.time=self.combat.destroyed_at if self.combat.destroyed_at is not None else horizon
        if self.combat.events:
            self.last_activity_ms=max(self.last_activity_ms,self.combat.events[-1]['timeMs'])
        if self.combat.hp>0 and self.time-self.last_activity_ms>=self.inactivity_limit_ms:
            self.draw=True
        self.scores=[a+b for a,b in zip(self.initial_scores,self.combat.scores)]
        self.bullets=[self.initial_bullets[i]+sum(e['agent']==i for e in self.combat.events) for i in range(2)]
        self.hits=[self.initial_hits[i]+sum(e['agent']==i and e['hit'] for e in self.combat.events) for i in range(2)]
        return self.response(start,self.time)

    def reset_scores(self,request):
        cutoff=float(request.get('timeMs',self.time if self.combat else 0))
        same_round=request.get('round')==self.round
        self.initial_scores=[0.,0.]
        if self.combat is not None:
            for event in self.combat.events:
                if not same_round or event['timeMs']<=cutoff:
                    event['scoreDelta']=0.
            self.combat.scores=[sum(e['scoreDelta'] for e in self.combat.events if e['agent']==i) for i in range(2)]
            self.scores=self.combat.scores.copy()
        else:
            self.scores=[0.,0.]
        return dict(scores=self.scores,reset=True)

    def response(self,start,end):
        if self.recording is not True:
            return dict(timeMs=end,done=self.combat.hp<=0 or self.draw)
        return dict(round=self.round,agents=[s.view(start,end) for s in self.shooters],
            timeMs=end,done=self.combat.hp<=0 or self.draw,draw=self.draw,
            drawReason='no_shots_15s' if self.draw else '',
            idleMs=max(0.,end-self.last_activity_ms),inactivityLimitMs=self.inactivity_limit_ms,
            targetHp=self.combat.hp,maxHp=self.combat.maximum_hp,
            events=self.combat.events,initialScores=self.initial_scores,scores=self.scores,
            initialBullets=self.initial_bullets,initialHits=self.initial_hits,bullets=self.bullets,
            motionSpeed=self.motion.speed,
            destroyed=self.combat.hp<=0,error='')


def serve(port):
    session = Session()
    dual = DualSession()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/decision/models':
                from unity_bridge.decision_models import model_catalog
                self.reply(200,model_catalog())
                return
            from unity_bridge.decision_models import active_info
            self.reply(200, dict(ready=True, model='ijhcs_default', protocol=4,
                decisionPolicy='parameter-conditioned PPO',decisionModelReady=MODEL_PATH.exists(),
                activeDecisionModel=active_info()))

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length > 16384:
                    raise ValueError('Request too large')
                request = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/reset':
                    result = session.reset(request)
                elif self.path == '/step':
                    result = session.step()
                elif self.path == '/round/reset':
                    result = dual.reset(request)
                elif self.path == '/round/step':
                    result = dual.step()
                elif self.path == '/scores/reset':
                    result = dual.reset_scores(request)
                else:
                    self.reply(404, dict(error='Unknown endpoint'))
                    return
                self.reply(200, result)
            except Exception as exc:
                traceback.print_exc()
                self.reply(400, dict(error=str(exc)))

        def reply(self, status, body):
            data = json.dumps(body, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f'AI bridge ready: http://127.0.0.1:{port}', flush=True)
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    serve(parser.parse_args().port)
