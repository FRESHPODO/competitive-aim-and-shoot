"""One reward-preference alpha per agent; outcome rewards have fixed sum 2."""
import math
from decimal import Decimal
FIELDS=(('alpha','보상 비율 α (0~1)',.5),)


def defaults():
    return {side:dict(alpha=dict(mode='fixed',value=.5,min=0.,max=1.,step=.05)) for side in 'AB'}


def values_from_alpha(alpha):
    alpha=float(alpha)
    if not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('α는 0~1 범위의 유한한 값이어야 합니다.')
    return dict(alpha=alpha,win=2*alpha,loss=2*(1-alpha),draw=0.)


def alpha_from_values(values):
    if 'alpha' in values:return values_from_alpha(values['alpha'])['alpha']
    win=float(values.get('win',1.));loss=float(values.get('loss',1.))
    if not math.isfinite(win+loss) or min(win,loss)<0:raise ValueError('보상과 비용은 유한한 0 이상 값이어야 합니다.')
    return win/(win+loss) if win+loss else .5


def _legacy_bounds(value):
    if not isinstance(value,dict):return float(value),float(value),float(value),False
    fixed=float(value.get('value',1.))
    if value.get('mode','fixed')=='fixed':return fixed,fixed,fixed,False
    return float(value['min']),float(value['max']),fixed,True


def normalize(config):
    config=config or {};result=defaults()
    for side in 'AB':
        item=config.get(side,{})
        if not isinstance(item,dict):item=dict(alpha=item)
        if 'alpha' in item:
            spec=item['alpha']
            if isinstance(spec,dict):
                result[side]['alpha'].update(spec)
                for key in ('value','min','max','step'):result[side]['alpha'][key]=float(result[side]['alpha'][key])
            else:result[side]['alpha']['value']=float(spec)
        elif 'win' in item or 'loss' in item:
            wl,wh,w,wr=_legacy_bounds(item.get('win',1.));ll,lh,l,lr=_legacy_bounds(item.get('loss',1.))
            if any(not math.isfinite(v) or v<0 for v in (wl,wh,w,ll,lh,l)):raise ValueError('이전 보상 설정을 α로 변환할 수 없습니다.')
            s=result[side]['alpha'];s['value']=alpha_from_values(dict(win=w,loss=l))
            if wr or lr:
                s.update(mode='uniform',min=alpha_from_values(dict(win=wl,loss=lh)),
                         max=alpha_from_values(dict(win=wh,loss=ll)),step=.01)
    return result


def validate_spec(spec):
    if spec.get('mode') not in ('fixed','uniform'):raise ValueError('α: 고정/범위를 선택하세요.')
    keys=('value',) if spec['mode']=='fixed' else ('min','max')
    for key in keys:values_from_alpha(spec[key])
    if spec['mode']=='uniform':
        step=spec.get('step',0)
        if not math.isfinite(step) or step<=0 or spec['max']<spec['min']:raise ValueError('α: 범위와 양수 추출 단위를 확인하세요.')
        if (Decimal(str(spec['max']))-Decimal(str(spec['min'])))/Decimal(str(step))>10**12:raise ValueError('α 추출 단위가 너무 작습니다.')


def validate(config):
    for item in normalize(config).values():validate_spec(item['alpha'])


def reachable_bounds(spec):
    validate_spec(spec)
    if spec['mode']=='fixed':return float(spec['value']),float(spec['value'])
    lo=Decimal(str(spec['min']));step=Decimal(str(spec['step']))
    n=int((Decimal(str(spec['max']))-lo)//step)
    return float(lo),float(lo+n*step)


def preview_text(spec):
    lo,hi=reachable_bounds(spec)
    low=values_from_alpha(lo);high=values_from_alpha(hi)
    if lo==hi:return f"실제 적용값\n성공 보상 +{low['win']:.6g}\n실패 비용 −{low['loss']:.6g}"
    return (f"추출 가능한 적용 범위\n성공 보상 +{low['win']:.6g} ~ +{high['win']:.6g}"
            f"\n실패 비용 −{low['loss']:.6g} ~ −{high['loss']:.6g}")


def sample(config,rng):
    config=normalize(config);result={}
    for side in 'AB':
        s=config[side]['alpha'];validate_spec(s)
        if s['mode']=='fixed':alpha=s['value']
        else:
            lo=Decimal(str(s['min']));step=Decimal(str(s['step']))
            n=int((Decimal(str(s['max']))-lo)//step)+1
            alpha=float(lo+int(rng.integers(n))*step)
        result[side]=values_from_alpha(alpha)
    return result


def terminal(scores,values):
    return [values[side]['win'] if scores[i]>scores[1-i] else -values[side]['loss'] if scores[i]<scores[1-i] else 0. for i,side in enumerate(('A','B'))]


def from_kill_events(events,values):
    killers={event['agent'] for event in events if event.get('killed')}
    if len(killers)!=1:return [0.,0.]
    winner=next(iter(killers))
    return [values[side]['win'] if i==winner else -values[side]['loss'] for i,side in enumerate(('A','B'))]
