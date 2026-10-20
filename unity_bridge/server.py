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
from src.utils.mymath import Convert, point_about_ray

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
    def __init__(self):
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
            self.obs, _ = self.agent.reset(seed=int(request.get('seed', self.episode + 1)))
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
            action, _ = self.sim.model.predict(self.obs, deterministic=True)
            self.obs, _, terminated, truncated, info = self.agent.step(action)
            self.rng_state = np.random.get_state()
        finally:
            np.random.set_state(outer_rng)
        self.done = bool(terminated or truncated)
        return self.packet(info)

    def packet(self, info):
        agent = self.agent
        end = max(0., float(info['time'])) if info else max(0., float(agent.task_timer))
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
    def __init__(self, hp=1000., minimum=100., maximum=500., critical=.01):
        if not all(np.isfinite([hp,minimum,maximum,critical])) or not (1<=hp<=1000 and 0<=minimum<=maximum and 0<=critical<=.5):
            raise ValueError('Invalid target damage settings')
        self.maximum_hp=hp
        self.hp=hp
        self.minimum=minimum
        self.maximum=maximum
        self.critical=critical
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
                damage=(self.hp if critical else self.minimum+(self.maximum-self.minimum)*(1-min(1.,ratio))**2) if shot['hit'] else 0.
                events.append(dict(agent=shot['agent'],timeMs=time,damage=float(damage),critical=bool(critical),
                                   hit=bool(shot['hit']),normalizedError=float(ratio),scoreDelta=0.,killed=False,
                                   shotAz=shot.get('shotAz',0.),shotEl=shot.get('shotEl',0.),
                                   spreadAngle=shot.get('spreadAngle',0.)))
            self.hp=max(0.,self.hp-sum(e['damage'] for e in events))
            if self.hp<=0:
                self.destroyed_at=time
                contributors=[e for e in events if e['damage']>0]
                critical_contributors=[e for e in contributors if e['critical']]
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
    """Continuous tracking; gate only the shoot decision during weapon cooldown."""
    def __init__(self,index):
        self.session=Session()
        self.index=index
        agent=self.session.agent
        original_unpack=agent._unpack_action
        original_plan=agent._plan_hand_movement
        original_check=agent._check_shot_result

        def unpack(action_normalized):
            action=original_unpack(action_normalized)
            requested=bool(action['kc'])
            allowed=agent.task_timer>=self.ready_at
            if not allowed: action['kc']=0
            self.decision_log.append(dict(timeMs=float(agent.task_timer),requested=requested,
                                          allowed=allowed,actual=bool(action['kc'])))
            return action

        def plan(state,action):
            already_committed=agent.waiting_motor_actual_movement_pos is not None
            # The source pads a committed plan to its single-shot episode limit.
            # Supply a bounded rolling horizon instead of resetting the episode.
            original_limit=agent.config['task']['time_limit_ms']
            agent.config['task']['time_limit_ms']=max(original_limit,agent.task_timer+5000)
            try: original_plan(state,action)
            finally: agent.config['task']['time_limit_ms']=original_limit
            if not already_committed and agent.waiting_motor_actual_movement_pos is not None:
                self.commit_started=float(agent.task_timer)

        def check(state,action):
            if self.commit_started is None: agent.shoot_timing=1e12
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
            return result
        times=np.unique(np.r_[np.arange(start,end,10.),end])
        packet=dict(self.packet)
        packet['frames']=[sample(t) for t in times]
        packet['timeMs']=end
        packet['done']=False
        packet['cooldown']=self.last_shot<=end<self.ready_at
        return packet


class DualSession:
    """Shared-clock multi-shot combat with one authoritative stationary target."""
    def __init__(self):
        self.shooters=[LiveShooter(0),LiveShooter(1)]
        self.round=0
        self.scores=[0.,0.]
        self.combat=None
        self.bullets=[0,0]
        self.hits=[0,0]

    def reset(self,request):
        profiles=request.get('profiles',[DEFAULTS,DEFAULTS])
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
        intervals=request.get('shotIntervalsMs',[150,150])
        spreads=request.get('shotSpreadDeg',[0,0])
        if len(spreads)!=2 or not np.isfinite(spreads).all() or not all(0<=v<=180 for v in spreads):
            raise ValueError('Two valid bullet spread angles required')
        if len(intervals)!=2 or not all(np.isfinite(intervals)) or not all(50<=v<=2000 for v in intervals):
            raise ValueError('Two valid per-agent shot intervals required')
        self.combat=TargetCombat(float(request.get('targetHp',1000)),float(request.get('minDamage',100)),
                                 float(request.get('maxDamage',300)),float(request.get('criticalRadius',.01)))
        self.initial_scores=self.scores.copy()
        self.initial_bullets=self.bullets.copy()
        self.initial_hits=self.hits.copy()
        self.round+=1
        self.time=0.
        self.event_cursor=[0,0]
        for i,shooter in enumerate(self.shooters):
            shooter.reset(profiles[i],radius,self.round*1000+i*100,float(intervals[i]),self.motion,float(spreads[i]))
        return self.response(0.,0.)

    def step(self):
        if self.combat is None or self.combat.hp<=0: raise ValueError('Start a new round first')
        start=self.time
        horizon=self.time+100.
        shots=[]
        for i,shooter in enumerate(self.shooters):
            shooter.advance(horizon)
            while self.event_cursor[i]<len(shooter.shots):
                shot=shooter.shots[self.event_cursor[i]]
                if shot['timeMs']>horizon: break
                shots.append(shot)
                self.event_cursor[i]+=1
        self.combat.resolve(shots)
        self.time=self.combat.destroyed_at if self.combat.destroyed_at is not None else horizon
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
        return dict(round=self.round,agents=[s.view(start,end) for s in self.shooters],
            timeMs=end,done=self.combat.hp<=0,targetHp=self.combat.hp,maxHp=self.combat.maximum_hp,
            events=self.combat.events,initialScores=self.initial_scores,scores=self.scores,
            initialBullets=self.initial_bullets,initialHits=self.initial_hits,bullets=self.bullets,
            motionSpeed=self.motion.speed,
            destroyed=self.combat.hp<=0,error='')


def serve(port):
    session = Session()
    dual = DualSession()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply(200, dict(ready=True, model='ijhcs_default', protocol=4))

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
