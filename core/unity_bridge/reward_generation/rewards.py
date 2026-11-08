"""Outcome + actual damage fraction + independent critical-kill bonus."""
from copy import deepcopy
from itertools import groupby
from unity_bridge import competitive_rewards as base

FACTORS = ('damageFactor', 'criticalFactor')


def defaults():
    result = base.defaults()
    for side in 'AB':
        for key in FACTORS:
            result[side][key] = dict(mode='uniform', value=0., min=0., max=1., step=.1)
    return result


def normalize(config):
    result = base.normalize(config)
    for side in 'AB':
        side_config = (config or {}).get(side, {})
        if not isinstance(side_config, dict):
            side_config = {}
        for key in FACTORS:
            spec = side_config.get(key, dict(mode='fixed', value=0.))
            spec = deepcopy(spec) if isinstance(spec, dict) else dict(mode='fixed', value=float(spec))
            result[side][key] = dict(mode='fixed', value=0., min=0., max=1., step=.1)
            result[side][key].update(spec)
            for field in ('value', 'min', 'max', 'step'):
                result[side][key][field] = float(result[side][key][field])
    return result


def validate(config):
    config = normalize(config)
    base.validate(config)
    for side in 'AB':
        for key in FACTORS:
            try:
                base.validate_spec(config[side][key])
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{side}.{key}: 0~1 고정값 또는 유효한 추출 범위가 필요합니다.') from error


def sample(config, rng):
    config = normalize(config)
    validate(config)
    result = base.sample(config, rng)
    for side in 'AB':
        for key in FACTORS:
            result[side][key] = base.sample({s: dict(alpha=config[side][key]) for s in 'AB'}, rng)['A']['alpha']
    return result


def breakdown(events, maximum_hp, values):
    if maximum_hp <= 0:
        raise ValueError('Maximum HP must be positive')
    damage = [0., 0.]
    hp = float(maximum_hp)
    # Attribute simultaneous overkill proportionally; never award damage beyond HP.
    for _, batch in groupby(sorted(events, key=lambda e: e['timeMs']), key=lambda e: e['timeMs']):
        batch = list(batch)
        raw = sum(max(0., e['damage']) for e in batch)
        actual = min(hp, raw)
        for event in batch:
            damage[event['agent']] += actual * max(0., event['damage']) / raw if raw else 0.
        hp -= actual
    outcome = base.from_kill_events(events, values)
    result = []
    for i, side in enumerate('AB'):
        critical_kill = any(e['agent'] == i and e.get('critical') and e.get('killed') for e in events)
        fraction = min(1., damage[i] / maximum_hp)
        damage_reward = values[side]['damageFactor'] * fraction
        critical_reward = values[side]['criticalFactor'] * int(critical_kill)
        result.append(dict(outcome=outcome[i], actualDamage=damage[i], damageFraction=fraction,
                           damageReward=damage_reward, criticalKill=critical_kill,
                           criticalReward=critical_reward, total=outcome[i]+damage_reward+critical_reward))
    return result
