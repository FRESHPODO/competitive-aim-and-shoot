"""Playback telemetry must not reveal decisions before their timestamp."""
from unity_bridge.server import LiveShooter, SharedMotion, DEFAULTS


def main():
    shooter=LiveShooter(0)
    motion=SharedMotion(config=shooter.session.agent.config['task'])
    shooter.reset(DEFAULTS,5,123,150,motion)
    shooter.strategy_log=[dict(timeMs=100.,approved=False,approveProbability=.2),
                          dict(timeMs=200.,approved=True,approveProbability=.8)]
    before=shooter.view(90,90)['frames'][0]
    blocked=shooter.view(150,150)['frames'][0]
    approved=shooter.view(250,250)['frames'][0]
    idle=shooter.view(350,350)['frames'][0]
    assert not before['strategyHasDecision'] and before['strategyRejected']==0
    assert blocked['strategyBlocked'] and blocked['strategyRejected']==1
    assert blocked['strategyLastRejectedMs']==100 and blocked['strategyApproveProbability']==.2
    assert not approved['strategyBlocked'] and approved['strategyRejected']==1
    assert approved['strategyApproveProbability']==.8
    assert not idle['strategyHasDecision'] and not idle['strategyBlocked']
    shooter.strategy_log=[dict(timeMs=100.,approved=False,approveProbability=.2)]
    after_rejection=shooter.view(200,200)['frames'][0]
    assert not after_rejection['strategyHasDecision'] and not after_rejection['strategyBlocked']
    print('PASS: each candidate has its own result; old rejection clears on next BUMP without needing an approval; no future decisions shown.')


if __name__=='__main__': main()
