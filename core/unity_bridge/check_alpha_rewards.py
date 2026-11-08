"""Alpha configuration, sampling, actual payoff and displayed payoff agree."""
from copy import deepcopy
import numpy as np
from unity_bridge.competitive_rewards import defaults,normalize,validate,sample,values_from_alpha,from_kill_events,preview_text,reachable_bounds
from unity_bridge.fire_decider import FEATURE_NAMES,OBS_SIZE


def main():
    for alpha,win,loss in ((0,0,2),(.25,.5,1.5),(.5,1,1),(.75,1.5,.5),(1,2,0)):
        v=values_from_alpha(alpha);assert v['win']==win and v['loss']==loss and v['draw']==0
    config=defaults();config['A']['alpha'].update(mode='uniform',min=.2,max=.8,step=.3)
    config['B']['alpha']['value']=.75;validate(config);rng=np.random.default_rng(17);seen=set()
    for _ in range(100):
        values=sample(config,rng);seen.add(values['A']['alpha']);assert values['B']['alpha']==.75
        for v in values.values():assert abs(v['win']+v['loss']-2)<1e-12
    assert seen=={.2,.5,.8}
    v={'A':values_from_alpha(.75),'B':values_from_alpha(.25)}
    assert from_kill_events([dict(agent=0,killed=True)],v)==[1.5,-1.5]
    assert from_kill_events([dict(agent=1,killed=True)],v)==[-.5,.5]
    assert from_kill_events([],v)==[0.,0.]
    old=normalize({'A':dict(win=3,loss=1)})
    assert old['A']['alpha']['value']==.75
    oldrange=normalize({'A':dict(win=dict(mode='uniform',min=1,max=3,value=1,step=1),loss=1)})
    assert oldrange['A']['alpha']['min']==.5 and oldrange['A']['alpha']['max']==.75
    assert preview_text(dict(mode='fixed',value=.75))=='실제 적용값\n성공 보상 +1.5\n실패 비용 −0.5'
    spec=dict(mode='uniform',min=0,max=.95,step=.3)
    assert reachable_bounds(spec)==(0.,.9)
    assert '+1.8' in preview_text(spec) and '−0.2' in preview_text(spec)
    for alpha in (-.1,1.1,float('nan'),float('inf')):
        bad=defaults();bad['A']['alpha']['value']=alpha
        try:validate(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid alpha accepted')
    bad=deepcopy(config);bad['A']['alpha']['step']=0
    try:validate(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid extraction step accepted')
    assert OBS_SIZE==22 and 'own_reward_alpha' in FEATURE_NAMES
    assert 'own_kill_reward' not in FEATURE_NAMES and 'own_loss_cost' not in FEATURE_NAMES
    print('PASS alpha endpoints, sampling grid, normalized rewards, kill outcomes, legacy conversion, displayed reachable bounds, validation and 22-feature policy')


if __name__=='__main__':main()
