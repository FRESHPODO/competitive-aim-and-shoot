import numpy as np
from unity_bridge.server import LiveShooter, SharedMotion, DEFAULTS


def main():
    shooter=LiveShooter(0)
    motion=SharedMotion(10,0,config=shooter.session.agent.config['task'])
    shooter.reset(DEFAULTS,5,123,600,motion)
    agent_id=id(shooter.session.agent)
    for t in range(100,6100,100):
        shooter.advance(t)
    assert id(shooter.session.agent)==agent_id and shooter.session.episode==1
    assert len(shooter.shots)>=3, len(shooter.shots)
    assert shooter.latest>=6000
    for a,b in zip(shooter.shots,shooter.shots[1:]):
        assert b['timeMs']-a['timeMs']>=600
    for shot in shooter.shots:
        decisions=[d for d in shooter.decision_log if shot['timeMs']<=d['timeMs']<shot['timeMs']+600]
        assert all(not d['actual'] for d in decisions)
    first=shooter.shots[0]['timeMs']
    frames=shooter.view(first,first+550)['frames']
    assert all(f['cooldown'] and not f['fireCommitted'] for f in frames)
    movement=np.ptp([f['cameraAz'] for f in frames])+np.ptp([f['cameraEl'] for f in frames])
    assert movement>.05, movement
    assert shooter.session.agent.task_timer>=6000
    print('PASS: one environment reset per target, tracking moves during cooldown, no shoot decisions during cooldown, shot spacing, continued tracking beyond source episode limit.')


if __name__=='__main__':main()
