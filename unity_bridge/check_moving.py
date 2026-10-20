import json
import numpy as np
from pathlib import Path
from unity_bridge.server import DualSession, SharedMotion
from src.agent.task import AimandShootSpiderShotTask


def main():
    d=DualSession()
    config=d.shooters[0].session.agent.config['task']
    for direction in (0,45,90,180,270):
        shared=SharedMotion(20,direction,config=config,seed=42)
        original=AimandShootSpiderShotTask(config)
        original.reset(camera_azel_deg=[0,0],target_pos_world=shared.initial,
                       target_motion_dir_deg=direction,target_speed_deg_s=20,
                       target_radius_deg=1.5,rng=np.random.default_rng(42))
        for t in (0,100,500,1500):
            assert np.allclose(shared.position(t),original.orbit_target(t,unit='ms',inplace=False)[0],atol=1e-10)
            assert np.isclose(np.linalg.norm(shared.position(t)),1)
    random_a=SharedMotion(10,0,config=config,seed=1,random_spawn=True,random_direction=True)
    random_b=SharedMotion(10,0,config=config,seed=2,random_spawn=True,random_direction=True)
    assert not np.allclose(random_a.initial,random_b.initial)
    speeds=[]
    directions=[]
    for _ in range(4):
        d.reset(dict(randomTargetSpeed=True,targetSpeedMin=5,targetSpeedMax=15))
        speeds.append(d.motion.speed)
        directions.append(d.motion.direction)
        assert 5<=d.motion.speed<=15
        for shooter in d.shooters:
            assert shooter.motion is d.motion
            assert shooter.session.agent.init_game_env['target_speed_deg_s']==d.motion.speed
    assert len(set(speeds))>1 and len(set(directions))>1
    d.reset(dict(randomTargetSpeed=True,targetSpeedMin=7.5,targetSpeedMax=7.5))
    assert d.motion.speed==7.5
    d.reset(dict(randomTargetSpeed=False,targetSpeed=12))
    assert d.motion.speed==12
    p=d.reset(dict(targetSpeed=10,targetDirection=45,targetHp=1000,minDamage=1,maxDamage=1,criticalRadius=0))
    motion=d.motion
    before=motion.position(0)
    for _ in range(100):
        p=d.step()
        a,b=p['agents']
        assert [f['timeMs'] for f in a['frames']]==[f['timeMs'] for f in b['frames']]
        for fa,fb in zip(a['frames'],b['frames']):
            assert fa['targetAz']==fb['targetAz'] and fa['targetEl']==fb['targetEl']
        for shooter in d.shooters:
            initial=shooter.session.agent.init_game_env
            assert np.allclose(initial['target_pos_world'],motion.position(shooter.offset),atol=1e-6)
        if len(p['events'])>=6:break
    assert not np.allclose(before,motion.position(p['timeMs']))
    assert sum(p['bullets'])==len(p['events'])>=6
    # A kill gives one point and counts just the shots before destruction.
    p=d.reset(dict(targetSpeed=0,targetHp=10,minDamage=20,maxDamage=80,criticalRadius=.08))
    for _ in range(100):
        p=d.step()
        if p['done']:break
    assert p['done'] and sum(p['scores'])==1
    bullets=d.bullets.copy()
    d.reset_scores(dict(round=d.round,timeMs=d.time))
    assert d.scores==[0,0] and d.bullets==bullets
    p=d.reset(dict(targetSpeed=0))
    assert p['initialScores']==[0,0] and p['initialBullets']==bullets
    # Reset at an earlier displayed clock must not erase a later buffered kill.
    for _ in range(100):
        p=d.step()
        if p['done']:break
    kill_time=d.combat.destroyed_at
    assert kill_time is not None
    d.reset_scores(dict(round=d.round,timeMs=kill_time-1))
    assert sum(d.scores)==1
    d.reset_scores(dict(round=d.round,timeMs=kill_time))
    assert sum(d.scores)==0
    Path('local-results/moving-target-check.json').write_text(json.dumps(dict(
        testedSpeed=10,testedDirection=45,bullets=bullets,scoreReset=True,sharedTrajectory=True),indent=2),encoding='utf-8')
    print('PASS: co-located shared motion, restart continuity, equal frame timestamps, cumulative bullets, score reset without erasing future events.')


if __name__=='__main__':main()
