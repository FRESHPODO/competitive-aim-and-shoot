import json
from pathlib import Path
from unity_bridge.server import DualSession, TargetCombat, DEFAULTS


def shot(agent,time,ratio,hit=True):
    return dict(agent=agent,timeMs=time,normalizedError=ratio,hit=hit)


def main():
    large=TargetCombat(100,5000,1000000,0)
    large.resolve([shot(0,100,.5)])
    assert large.hp==0 and large.events[0]['damage']>1000
    zero=TargetCombat(100,0,0,0)
    zero.resolve([shot(0,100,.5)])
    assert zero.hp==100 and zero.events[0]['damage']==0
    c=TargetCombat(100,15,80,.08)
    c.resolve([shot(0,100,1)])
    assert c.hp==85 and c.scores==[0,0]
    c.resolve([shot(1,200,.01)])
    assert c.hp==0 and c.scores==[0,1]
    c.resolve([shot(0,300,0)])
    assert c.scores==[0,1] and len(c.events)==2
    c=TargetCombat(50,15,80,.08)
    c.resolve([shot(0,120,0),shot(1,100,0)])
    assert c.scores==[0,1] and c.destroyed_at==100
    c=TargetCombat(100,15,80,.08)
    c.resolve([shot(0,100,0),shot(1,100,0)])
    assert c.scores==[.5,.5]
    # Simultaneous normal kills split; exactly one critical takes the full point.
    for critical_agent in (0,1):
        c=TargetCombat(100,100,100,.08)
        c.resolve([shot(i,100,0 if i==critical_agent else .5) for i in (0,1)])
        assert c.scores[critical_agent]==1 and c.scores[1-critical_agent]==0
        assert sum(e['scoreDelta'] for e in c.events)==1
    c=TargetCombat(100,60,60,.08)
    c.resolve([shot(0,100,.5),shot(1,100,.5)])
    assert c.scores==[.5,.5]
    c=TargetCombat(100,15,80,.08)
    c.resolve([shot(0,100,0,False)])
    assert c.hp==100
    dual=DualSession()
    dual.reset(dict(shotIntervalsMs=[100,700]))
    assert dual.shooters[0].interval==100 and dual.shooters[1].interval==700
    p=dual.reset(dict(targetHp=1000,minDamage=1,maxDamage=1,criticalRadius=0))
    previous=0
    for _ in range(100):
        p=dual.step()
        assert p['timeMs']>=previous
        assert all(a['frames'][0]['timeMs']==previous for a in p['agents'])
        previous=p['timeMs']
        json.dumps(p,allow_nan=False)
        if len(p['events'])>=8:break
    assert not p['done'] and all(sum(e['agent']==i for e in p['events'])>=2 for i in (0,1)), p['events']
    assert all(a['frames'][-1]['timeMs']==p['timeMs'] for a in p['agents'])
    p=dual.reset(dict(targetHp=20,minDamage=30,maxDamage=80,criticalRadius=.08))
    for _ in range(100):
        p=dual.step()
        if p['done']:break
    assert p['done'] and p['destroyed'] and p['targetHp']==0
    assert abs(sum(p['scores'])-sum(p['initialScores'])-1)<1e-6
    assert p['timeMs']==p['events'][-1]['timeMs']
    Path('local-results/combat-check.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    print('PASS: edge/center/miss damage, instant kill, shot ordering, simultaneous split, repeated shots, shared clock, kill score.')


if __name__=='__main__':main()
