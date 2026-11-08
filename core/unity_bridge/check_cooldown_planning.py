"""Cooldown limits actual bullets, while accepted plans can start beforehand."""
import math
import numpy as np
import torch
from unity_bridge.server import DualSession, DEFAULTS
from unity_bridge.replay_native import state_at


def run_case(mode, horizon, trigger, interval):
    dual=DualSession(recording='replay')
    profile=list(DEFAULTS);profile[:4]=[0.,0.,0.,0.]
    dual.reset(dict(profiles=[profile,profile],targetSpeed=0,targetRadius=5,
                    weapons=[dict(minimum=1,maximum=1,instantKill=False)]*2,
                    shotIntervalsMs=[interval,interval],decisionEnabled=[False,False]))
    agent=dual.shooters[0].session.agent
    action=np.zeros(len(agent.action_list),dtype=np.float32)
    for key,value in (('th',horizon),('tc',trigger),('kc',1.)):
        index=agent.action_list.index(key);lo,hi=agent.action_range[index]
        action[index]=2*(value-lo)/(hi-lo)-1
    actors={id(s.session.sim.model):s.session.sim.model for s in dual.shooters}
    originals={key:actor.predict for key,actor in actors.items()}
    for actor in actors.values():actor.predict=lambda *args,**kwargs:(action.copy(),None)
    class Gate:
        def decide(self,now,context):
            return dict(approved=True,reason='test_approve',snapshotTimeMs=context['time'])
    try:
        for shooter in dual.shooters:shooter.decider=Gate();shooter.decision_mode=mode
        while dual.time<3000:dual.step(deadline_ms=3000)
    finally:
        for key,actor in actors.items():actor.predict=originals[key]
    for shooter in dual.shooters:
        times=[s['timeMs'] for s in shooter.shots if s['timeMs']<=dual.time]
        assert len(times)>=3,times
        first=times[0]
        expected=math.ceil(first/100)*100+100+trigger*horizon
        if expected<first+interval:
            expected+=math.ceil((first+interval-expected)/100)*100
        assert abs(times[1]-expected)<1e-6,(times,expected)
        assert all(b-a>=interval-1e-6 for a,b in zip(times,times[1:])),times
        starts=[d for d in shooter.decision_log if d['actual']]
        assert any(not d['weaponReady'] and not d['cooldownBlocked'] for d in starts)
        # A committed plan remains immune to subsequent PPO calls.
        assert all(d['ppoApproved'] is None for d in shooter.decision_log if d['pendingPlan'])
        assert not any(d.get('cooldownDelayed') for d in shooter.decision_log)
        if horizon==100:assert any(d.get('cooldownBlocked') for d in shooter.decision_log)
    replay=dict(hp=1000,duration=dual.time,settings=dict(shotIntervalsMs=[interval,interval]),
                events=dual.combat.events,shots=[dict(agent=i,t=s['timeMs']) for i,shooter in enumerate(dual.shooters) for s in shooter.shots],
                decisions=[s.strategy_log for s in dual.shooters],originalDecisions=[s.decision_log for s in dual.shooters])
    mid=state_at(replay,dual.shooters[0].shots[0]['timeMs']+50)['agents'][0]
    assert mid['cooldown']>0
    if horizon==100:assert not mid['plan_active'] and mid['sub']=='쿨다운 중'
    else:assert mid['plan_active']
    print(f'PASS {mode}: Th={horizon}, tc={trigger}, cooldown={interval}; first bullets {times[:3]}')


def main():
    torch.set_num_threads(1)
    for mode in ('bump','candidate'):
        # Early, late, exact-boundary, and off-grid cooldown completion.
        for case in ((100,0.,400),(400,1.,400),(300,1.,400),(100,0.,455)):
            run_case(mode,*case)


if __name__=='__main__':main()
