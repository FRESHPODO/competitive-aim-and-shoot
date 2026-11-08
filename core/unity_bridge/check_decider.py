"""Current observation contract and explicit public/private input boundaries."""
from copy import deepcopy
import numpy as np
from unity_bridge.fire_decider import encode_context,OBS_SIZE,FEATURE_NAMES


def main():
    weapon=dict(minimum=100.,maximum=400.,instantKill=True)
    c=dict(hp=1000.,maximum_hp=1000.,critical=.01,time=0.,
        game=dict(weapon,radius=5.,speed=12.,cooldown=400.),
        own=dict(profile=[.2,.08,.04,.04],ready_at=0.),
        opponent=dict(ready_at=900.),
        belief=dict(normalized_error=.1,damage=343.,critical=False),
        proposal=dict(th=500.,tc=.7,delay_ms=450.),own_reward=dict(win=1.,loss=1.))
    base=encode_context(100.,c);assert base.shape==(OBS_SIZE,) and OBS_SIZE==22
    # Absolute episode time and snapshot freshness must not enter the policy.
    shifted=deepcopy(c);shifted['time']=-12345.
    shifted['own']['ready_at']+=5000.;shifted['opponent']['ready_at']+=5000.
    assert np.array_equal(base,encode_context(5100.,shifted))
    changed=deepcopy(c);changed['time']=999999.
    changed.update(elapsed=999.,snapshot_age=999.,remaining_time=999.,timeout=999.)
    assert np.array_equal(base,encode_context(100.,changed))
    assert not {'elapsed','snapshot_age','remaining_time','timeout'}.intersection(FEATURE_NAMES)
    for key in ('minimum','maximum','cooldown','radius','speed'):
        changed=deepcopy(c);changed['game'][key]*=1.5
        assert not np.array_equal(base,encode_context(100.,changed))
    for key in ('th','tc','delay_ms'):
        changed=deepcopy(c);changed['proposal'][key]*=1.5
        assert not np.array_equal(base,encode_context(100.,changed))
    for key in ('win','loss'):
        changed=deepcopy(c);changed['own_reward'][key]=5.
        assert not np.array_equal(base,encode_context(100.,changed))
    for i in range(4):
        changed=deepcopy(c);changed['own']['profile'][i]+=.1
        assert not np.array_equal(base,encode_context(100.,changed))
    changed=deepcopy(c);changed['opponent'].update(profile=[.5]*8,error=99.,normalized_error=99.,spread=99.)
    changed['opponent_reward']=dict(win=99.,loss=99.);changed['own']['spread']=99.
    assert np.array_equal(base,encode_context(100.,changed))
    assert not any('opponent_parameter' in k or 'spread' in k for k in FEATURE_NAMES)
    print('PASS 22-feature schema: no episode time, remaining time or snapshot age; cooldown durations preserved')


if __name__=='__main__':main()
