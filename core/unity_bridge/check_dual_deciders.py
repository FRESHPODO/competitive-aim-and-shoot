"""Verify independently selected PPO gates and the exact no-shot draw deadline."""
from unity_bridge.server import DualSession


class Reject:
    def decide(self,now,context):
        return dict(approved=False,reason='test_reject',snapshotTimeMs=context['time'])


def main():
    dual=DualSession()
    packet=dual.reset(dict(decisionEnabled=[True,True],decisionModels=['','']))
    assert all(agent['strategyEnabled'] for agent in packet['agents'])
    dual.reset(dict(decisionEnabled=[True,True],decisionModels=['','monitor-check/candidate.zip']))
    assert dual.shooters[1].decider.model.num_timesteps==256
    assert dual.shooters[0].decider.model.num_timesteps!=256
    dual.reset(dict(decisionEnabled=[False,True]))
    assert dual.shooters[0].decider is None and dual.shooters[1].decider is not None
    dual.reset(dict(decisionAgent=None))
    assert all(s.decider is None for s in dual.shooters)
    for shooter in dual.shooters: shooter.decider=Reject()
    for _ in range(149):
        packet=dual.step()
        assert not packet['done']
    packet=dual.step()
    assert packet['timeMs']==15000 and packet['draw'] and packet['done']
    assert not packet['destroyed'] and packet['targetHp']==1000
    assert dual.combat.scores==[0,0] and dual.combat.events==[]
    assert all(shooter.strategy_log for shooter in dual.shooters)
    # A miss is still activity, even though it causes no damage. Schedule a
    # single authoritative miss and ensure it postpones the deadline exactly.
    dual.reset(dict(decisionAgent=None))
    for shooter in dual.shooters: shooter.decider=Reject()
    for _ in range(149): dual.step()
    dual.shooters[0].shots.append(dict(agent=0,timeMs=14950.,hit=False,normalizedError=2.))
    packet=dual.step()
    assert not packet['done'] and dual.last_activity_ms==14950
    while not packet['done']: packet=dual.step()
    assert packet['draw'] and packet['timeMs']==29950
    packet=dual.reset(dict(decisionEnabled=[True,True]))
    assert not packet['draw'] and packet['idleMs']==0 and packet['timeMs']==0
    print('PASS: both PPO / independent checkpoints / B-only / legacy API; no-shot draw at 15s; miss resets deadline; zero draw points; clean next round.')


if __name__=='__main__': main()
