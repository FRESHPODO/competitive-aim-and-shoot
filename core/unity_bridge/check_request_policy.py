"""Request-only policy, private observations, belief estimates and re-planning."""
import copy
import numpy as np
import torch
from unity_bridge.server import DualSession,DEFAULTS
from unity_bridge.fire_decider import encode_context,FEATURE_NAMES
from unity_bridge.replay_native import state_at
from unity_bridge.aim_belief import estimate_candidate
import unity_bridge.server as server


def simulate(fire=True,horizon=100,trigger=0.,interval=455,approve=True,noise=False):
    dual=DualSession(recording='replay');dual.round=141
    profile=list(DEFAULTS)
    if not noise:profile[:4]=[0.,0.,0.,0.]
    dual.reset(dict(profiles=[profile,profile],targetSpeed=12,targetRadius=5,
        weapons=[dict(minimum=1,maximum=2,instantKill=False)]*2,
        shotIntervalsMs=[interval,interval],decisionEnabled=[False,False]))
    calls=[[],[]];agent=dual.shooters[0].session.agent
    action=np.zeros(len(agent.action_list),dtype=np.float32)
    for key,value in (('th',horizon),('tc',trigger),('kc',int(fire))):
        index=agent.action_list.index(key);lo,hi=agent.action_range[index]
        action[index]=2*(value-lo)/(hi-lo)-1
    actors={id(s.session.sim.model):s.session.sim.model for s in dual.shooters}
    originals={key:actor.predict for key,actor in actors.items()}
    for actor in actors.values():actor.predict=lambda *args,**kwargs:(action.copy(),None)
    class Gate:
        def __init__(self,i):self.i=i
        def decide(self,now,c):
            assert set(c['opponent'])=={'ready_at'}
            assert set(c['own'])=={'profile','ready_at'}
            assert c['shared_attack_conditions']
            assert len(c['own']['profile'])==4
            assert 'error' not in c['own'] and 'normalized_error' not in c['own']
            obs=encode_context(now,c)
            changed=copy.deepcopy(c);changed['opponent'].update(profile=[99]*8,error=999)
            changed['opponent_reward']=dict(win=999,loss=999)
            changed['own']['spread']=99;changed['opponent']['spread']=99
            assert np.array_equal(obs,encode_context(now,changed))
            calls[self.i].append((now,copy.deepcopy(c)))
            return dict(approved=approve,reason='request_test',snapshotTimeMs=c['time'])
    for i,s in enumerate(dual.shooters):s.decider=Gate(i);s.decision_mode='bump'
    try:
        while dual.time<3000:dual.step(deadline_ms=3000)
    finally:
        for key,actor in actors.items():actor.predict=originals[key]
    return dual,calls


def replay(dual):
    return dict(hp=1000,duration=dual.time,settings=dict(shotIntervalsMs=[s.interval for s in dual.shooters]),
        events=dual.combat.events,shots=[dict(agent=i,t=p['timeMs']) for i,s in enumerate(dual.shooters) for p in s.shots],
        decisions=[s.strategy_log for s in dual.shooters],originalDecisions=[s.decision_log for s in dual.shooters])


def main():
    torch.set_num_threads(1)
    assert not any('spread' in k or 'opponent_parameter' in k or 'opponent_center_error' in k for k in FEATURE_NAMES)
    idle,calls=simulate(fire=False)
    assert not any(calls) and not any(s.shots for s in idle.shooters)
    assert all(not a['sub_visible'] and a['sub']=='' for a in state_at(replay(idle),500)['agents'])
    print('PASS no request: no PPO call, no training decision, no replay sub box')
    early,calls=simulate()
    for i,s in enumerate(early.shooters):
        assert [p['timeMs'] for p in s.shots[:3]]==[200.,700.,1200.]
        blocked=[d for d in s.decision_log if d['cooldownBlocked']]
        assert len(blocked)>1 and all(not d['actual'] and d['ppoApproved'] is None for d in blocked)
        assert all(d['requested'] and not d['pendingPlan'] for d in s.decision_log if d['ppoApproved'] is not None)
        assert any(d['actual'] and not d['weaponReady'] for d in s.decision_log)
        assert len(calls[i])==len(s.strategy_log)
    a=state_at(replay(early),250)['agents'][0]
    assert a['sub']=='쿨다운 중' and not a['plan_active']
    print('PASS early proposal: tracking and next-BUMP re-evaluation, not queued at cooldown expiry')
    late,_=simulate(horizon=400,trigger=1.,interval=400)
    assert [p['timeMs'] for p in late.shooters[0].shots[:3]]==[600.,1100.,1600.]
    a=state_at(replay(late),750)['agents'][0]
    assert a['plan_active'] and not a['sub_visible']
    print('PASS future-valid proposal starts during cooldown; committed plan is not re-gated')
    tracking,_=simulate(fire=False,noise=True)
    held,calls=simulate(approve=False,noise=True)
    assert all(len(c)>2 for c in calls) and not any(s.shots for s in held.shooters)
    for x,y in zip(tracking.shooters,held.shooters):
        assert x.frames==y.frames
    print('PASS rejected preview preserves ordinary tracking trajectory and RNG')
    original=server.estimate_candidate;checked=[]
    def checked_estimator(agent,state,delay,weapon,hp,critical):
        before=np.random.get_state();estimate=original(agent,state,delay,weapon,hp,critical)
        truth=agent.game_env.target_pos_world.copy();actual=agent.waiting_motor_actual_movement_pos.copy()
        try:
            agent.game_env.target_pos_world=-truth
            agent.waiting_motor_actual_movement_pos[:]+=10000
            assert estimate==original(agent,state,delay,weapon,hp,critical)
        finally:
            agent.game_env.target_pos_world=truth
            agent.waiting_motor_actual_movement_pos[:]=actual
        after=np.random.get_state()
        assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
        altered=dict(state);altered[agent._STATE_TARGET_POS_0_HAT]=np.asarray(state[agent._STATE_TARGET_POS_0_HAT])+np.array([100.,31.])
        assert original(agent,altered,delay,weapon,hp,critical)['normalized_error']!=estimate['normalized_error']
        checked.append(True);return estimate
    server.estimate_candidate=checked_estimator
    try:simulate(horizon=400,trigger=1.,interval=400,noise=True)
    finally:server.estimate_candidate=original
    assert checked
    print('PASS estimates use aiming belief and ideal plan, not true target or actual future motor noise')


if __name__=='__main__':main()
