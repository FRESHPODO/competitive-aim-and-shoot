"""Explicit fixed/uniform episode distributions, shared by GUI and training."""
import math
from decimal import Decimal

GAME = [
    ('targetHp','표적 체력',1000,1,1000),
    ('criticalRadius','중앙 치명타 반지름 비율',.01,0,.5),
    ('targetRadius','표적 반지름 (도)',5,.3,20),
    ('targetSpeed','표적 속도 (도/초)',12,0,40),
]
AGENT = [
    ('motor','운동 노이즈',.20,0,.5),
    ('position','위치 노이즈',.08,0,.5),
    ('speed','속도 노이즈',.04,0,.5),
    ('clock','시계 노이즈',.04,0,.5),
    ('success','원본 조준 성공 조건 (PPO 보상 아님)',40,1,64),
    ('failure','원본 조준 실패 조건 (PPO 비용 아님)',12,1,64),
    ('reward_decay','원본 조준 보상 감쇠',40,5,95),
    ('penalty_decay','원본 조준 페널티 감쇠',45,5,95),
]


WEAPON = [
    ('minDamage','외곽 데미지',100,0,None),
    ('maxDamage','중앙 데미지',400,0,None),
    ('cooldown','재장전 딜레이 (ms)',400,50,2000),
    ('spread','탄 퍼짐 (도)',0,0,180),
]
TOGGLES=[('instantKill','치명타 즉사',1,0,1),('spreadEnabled','탄 퍼짐 사용',0,0,1)]
COMMON_ATTACK_KEYS=('minDamage','maxDamage','instantKill','cooldown')


def validate_common_attack(settings):
    for key in COMMON_ATTACK_KEYS:
        def effective(spec):
            fields=('mode','value') if spec['mode']=='fixed' else ('mode','min','max','step')
            return tuple(spec.get(k) for k in fields)
        if effective(settings['A.'+key])!=effective(settings['B.'+key]):
            raise ValueError(f'{key}: A/B 개인 조건이 아닌 공통 게임 조건입니다. 양쪽 설정을 동일하게 지정하세요.')


def migrate(settings):
    result=defaults()
    result.update({k:v for k,v in settings.items() if k in result})
    for side in ('A','B'):
        for key in COMMON_ATTACK_KEYS:
            if key in settings:result[f'{side}.{key}']=dict(settings[key])
        if f'{side}.spreadEnabled' not in settings and f'{side}.spread' in settings:
            old=settings[f'{side}.spread'];enabled=old.get('value',0) if old.get('mode')=='fixed' else old.get('max',0)
            result[f'{side}.spreadEnabled']['value']=int(enabled>0)
    return result


def defaults():
    result={}
    for key,_,value,*_ in GAME:
        result[key]={'mode':'fixed','value':value,'min':value,'max':value}
    result['targetSpeed'].update(mode='uniform',min=0,max=40)
    for side in ('A','B'):
        for key,_,value,*_ in AGENT+WEAPON+TOGGLES:
            if side=='A' and key in ('motor','position','speed','clock'): value*=1.5
            result[f'{side}.{key}']={'mode':'fixed','value':value,'min':value,'max':value}
    for key,s in result.items():
        s['step']=.01 if key.split('.')[-1] in ('motor','position','speed','clock','criticalRadius') else 1.
    return result


def validate(settings):
    specs=[(key,lo,hi) for key,_,_,lo,hi in GAME]
    specs += [(f'{side}.{key}',lo,hi) for side in ('A','B') for key,_,_,lo,hi in AGENT+WEAPON+TOGGLES]
    if set(settings)!=set(k for k,_,_ in specs): raise ValueError('설정 항목이 누락되었거나 알 수 없는 항목이 있습니다.')
    for key,lo,hi in specs:
        s=settings[key]
        if s['mode'] not in ('fixed','uniform'): raise ValueError(f'{key}: 고정 또는 균등 범위를 선택하세요.')
        values=[s['value']] if s['mode']=='fixed' else [s['min'],s['max']]
        if any(not math.isfinite(v) or v<lo or (hi is not None and v>hi) for v in values):
            raise ValueError(f'{key}: 허용 범위 {lo} ~ {hi if hi is not None else "상한 없음"}')
        if len(values)==2 and values[0]>values[1]: raise ValueError(f'{key}: 최솟값이 최댓값보다 큽니다.')
        if s['mode']=='uniform':
            step=s.get('step',0)
            if not math.isfinite(step) or step<=0: raise ValueError(f'{key}: 추출 단위는 양수여야 합니다.')
            if (Decimal(str(s['max']))-Decimal(str(s['min'])))/Decimal(str(step))>10**12:
                raise ValueError(f'{key}: 추출 단위가 너무 작습니다.')
    validate_common_attack(settings)
    for side in ('A','B'):
        edge=settings[f'{side}.minDamage'];center=settings[f'{side}.maxDamage']
        edge_hi=edge['value'] if edge['mode']=='fixed' else edge['max']
        center_lo=center['value'] if center['mode']=='fixed' else center['min']
        if edge_hi>center_lo:raise ValueError(f'{side}: 외곽 데미지 상한은 중앙 데미지 하한 이하여야 합니다.')
        for key,*_ in TOGGLES:
            toggle=settings[f'{side}.{key}']
            if toggle['mode']!='fixed' or toggle['value'] not in (0,1):raise ValueError(f'{side}.{key}: 토글은 켜기/끄기만 가능합니다.')



def sample(settings,rng):
    validate_common_attack(settings)
    def draw(key):
        s=settings[key]
        if s['mode']=='fixed':return float(s['value'])
        lo=Decimal(str(s['min']));step=Decimal(str(s['step']))
        count=int((Decimal(str(s['max']))-lo)//step)+1
        return float(lo+int(rng.integers(count))*step)
    request={key:draw(key) for key,*_ in GAME}
    common={key:draw('A.'+key) for key in COMMON_ATTACK_KEYS}
    request.update(common)
    request.update(profiles=[[draw(f'{side}.{key}') for key,*_ in AGENT[:8]] for side in ('A','B')],
        shotIntervalsMs=[common['cooldown']]*2,
        shotSpreadDeg=[draw(f'{side}.spread') if draw(f'{side}.spreadEnabled') else 0. for side in ('A','B')],
        weapons=[dict(minimum=common['minDamage'],maximum=common['maxDamage'],
            instantKill=bool(common['instantKill'])) for side in ('A','B')],
        randomTargetSpawn=True,decisionAgent=None,decisionEnabled=[False,False])
    return request
