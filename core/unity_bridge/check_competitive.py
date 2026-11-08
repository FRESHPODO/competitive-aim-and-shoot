"""Verify BUMP observations, repeated shooting, shared rounds and discrete sampling."""
import numpy as np
from unity_bridge.training_settings import defaults,validate,sample
from unity_bridge.server import DualSession
from unity_bridge.fire_decider import encode_context,FEATURE_NAMES


def main():
    settings=defaults();settings['A.motor'].update(mode='uniform',min=.1,max=.3,step=.05)
    validate(settings);rng=np.random.default_rng(73)
    values={round(sample(settings,rng)['profiles'][0][0],8) for _ in range(200)}
    assert values=={.1,.15,.2,.25,.3}
    settings['A.motor']['step']=0
    try:validate(settings)
    except ValueError:pass
    else:raise AssertionError('Zero extraction step accepted')
    dual=DualSession();request=sample(defaults(),rng)
    request.update(targetSpeed=0,targetRadius=20,minDamage=10,maxDamage=10,criticalRadius=0,
                   weapons=[dict(minimum=10,maximum=10,instantKill=False)]*2)
    dual.reset(request);round_id=dual.round
    seen=[[],[]]
    class Gate:
        def __init__(self,i):self.i=i
        def decide(self,now,c):
            obs=encode_context(now,c)
            assert np.isclose(obs[FEATURE_NAMES.index('hp_fraction')],c['hp']/c['maximum_hp'])
            assert np.isclose(obs[FEATURE_NAMES.index('opponent_ready_in')],max(0,c['opponent']['ready_at']-now)/2000)
            seen[self.i].append((now,c['hp'],max(0,c['opponent']['ready_at']-now)))
            return dict(approved=True,reason='test',snapshotTimeMs=c['time'])
    for i,s in enumerate(dual.shooters):s.decider=Gate(i);s.decision_mode='bump'
    for _ in range(80):
        if dual.combat.hp<=0:break
        dual.step()
    assert dual.round==round_id
    assert all(len(s.shots)>1 for s in dual.shooters)
    assert all(len(v)>10 and len({r[1] for r in v})>1 and any(r[2]>0 for r in v) for v in seen)
    for shooter in dual.shooters:
        assert all(not d['actual'] or d['requested'] for d in shooter.decision_log)
        times=[s['timeMs'] for s in shooter.shots]
        assert all(b-a>=shooter.interval for a,b in zip(times,times[1:]))
    dual.reset(request)
    for s in dual.shooters:s.decider=None;s.decision_mode='candidate'
    while dual.time<435 and dual.combat.hp>0:dual.step(deadline_ms=435)
    assert dual.time==435
    dual.reset(request)
    class RejectThenApprove:
        def __init__(self):self.calls=0
        def decide(self,now,c):
            self.calls+=1
            return dict(approved=self.calls>5,reason='test_reconsider',snapshotTimeMs=c['time'])
    gates=[RejectThenApprove(),RejectThenApprove()]
    for i,s in enumerate(dual.shooters):s.decider=gates[i];s.decision_mode='bump'
    for _ in range(5):dual.step()
    assert not any(s.shots for s in dual.shooters)
    for _ in range(80):
        if dual.combat.hp<=0:break
        dual.step()
    assert all(g.calls>5 for g in gates) and all(s.shots for s in dual.shooters)
    assert all(any(d['requested'] and d['actual'] for d in s.decision_log[5:]) for s in dual.shooters)
    print('PASS: rejection produces no shot, original policy is asked again on later BUMPs, approval never forces an unrequested shot.')
    print('PASS: discrete grid, per-BUMP HP/opponent cooldown, continued multi-shot tracking, cooldown spacing, exact timeout.')


if __name__=='__main__':main()
