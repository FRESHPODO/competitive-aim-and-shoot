import numpy as np
from unity_bridge.server import sample_bullet_direction, DualSession
from src.utils.mymath import Convert


def main():
    a=np.random.default_rng(42);b=np.random.default_rng(42)
    assert sample_bullet_direction(12,8,0,a)==(12.,8.,0.)
    assert a.random()==b.random(), 'Zero spread must not consume weapon noise'
    for az,el in ((0,0),(179,85),(-179,-85)):
        rng=np.random.default_rng(77)
        center=Convert.sphr2cart_scalar(az,el)
        for _ in range(300):
            sa,se,offset=sample_bullet_direction(az,el,12,rng)
            direction=Convert.sphr2cart_scalar(sa,se)
            angle=np.degrees(np.arccos(np.clip(np.dot(center,direction),-1,1)))
            assert angle<=12.000001 and abs(angle-offset)<1e-6
    rng=np.random.default_rng(10)
    samples=[np.cos(np.radians(sample_bullet_direction(0,0,60,rng)[2])) for _ in range(3000)]
    assert abs(np.mean(samples)-.75)<.015
    d=DualSession();p=d.reset(dict(targetSpeed=0,shotSpreadDeg=[0,12],targetHp=1000,minDamage=1,maxDamage=1,criticalRadius=0))
    for _ in range(100):
        p=d.step()
        if all(sum(e['agent']==i for e in p['events'])>=3 for i in (0,1)):break
    assert any(e['agent']==1 and e['spreadAngle']>0 for e in p['events'])
    assert all(e['spreadAngle']==0 for e in p['events'] if e['agent']==0)
    assert all(0<=e['spreadAngle']<=12 for e in p['events'] if e['agent']==1)
    print('PASS: zero-spread equivalence, cone bounds across poles, uniform solid-angle sampling, separate per-agent spread and actual shot events.')


if __name__=='__main__':main()
