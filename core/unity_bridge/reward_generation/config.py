"""Validated, persistent configuration for one learner and recent frozen opponents."""
from copy import deepcopy
import secrets
import math
from unity_bridge.training_settings import defaults as game_defaults, migrate, validate as validate_game
from unity_bridge.reward_generation.rewards import defaults as reward_defaults, normalize, validate as validate_rewards
from unity_bridge.player_presets import PRESETS, apply, provenance

CUSTOM = '__custom__'
KIND = 'generational-damage-critical-v1'


def defaults():
    settings = game_defaults()
    # The new screen exposes attack conditions once, as shared game settings.
    for side in 'AB':
        settings[f'{side}.spreadEnabled']['value'] = 0
    return dict(kind=KIND, episodes=150000, timeout=12., batchEpisodes=16, workers=0,
                generationGames=10000, retainedGenerations=3, checkpointSeconds=60., seed=None,
                settings=settings, rewardConfig=reward_defaults(),
                presetChoices={side: ['professional_mean', 'amateur_mean'] for side in 'AB'})


def validate(config, *, resolve_seed=False):
    c = deepcopy(config)
    if c.get('kind') != KIND:
        raise ValueError('세대 순환 학습기 설정 파일이 아닙니다.')
    for key in ('episodes', 'batchEpisodes', 'generationGames'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError(f'{key}: 양의 정수가 필요합니다.')
    if type(c.get('workers')) is not int or c['workers'] < 0 or c['workers'] > 64:
        raise ValueError('병렬 경기 수: 0~64 정수가 필요합니다.')
    if type(c.get('retainedGenerations')) is not int or c['retainedGenerations'] not in (2, 3):
        raise ValueError('유지할 상대 세대 수는 2 또는 3입니다.')
    for key in ('timeout', 'checkpointSeconds'):
        if not isinstance(c.get(key), (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key}: 양수가 필요합니다.')
    if c.get('seed') in ('', None):
        c['seed'] = secrets.randbits(32) if resolve_seed else None
    elif type(c['seed']) is not int or not 0 <= c['seed'] < 2**63:
        raise ValueError('Seed는 빈칸 또는 0 이상 2^63 미만 정수입니다.')
    c['settings'] = migrate(c['settings'])
    validate_game(c['settings'])
    c['rewardConfig'] = normalize(c['rewardConfig'])
    validate_rewards(c['rewardConfig'])
    choices = c.get('presetChoices', {})
    if set(choices) != {'A', 'B'}:
        raise ValueError('학습자·상대의 조준 프리셋 목록이 필요합니다.')
    for side in 'AB':
        ids = choices[side]
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError(f'{side}: 사용할 조준 프리셋을 하나 이상 선택하세요.')
        if any(pid != CUSTOM and pid not in PRESETS for pid in ids):
            raise ValueError(f'{side}: 알 수 없는 조준 프리셋입니다.')
    return c


def sample_scenario(config, rng):
    """Uniform, independent preset choices per side, fixed for the whole game."""
    settings = config['settings']
    selections = {}
    for side in 'AB':
        ids = config['presetChoices'][side]
        pid = ids[int(rng.integers(len(ids)))]
        selections[side] = provenance(pid) if pid != CUSTOM else dict(id=CUSTOM, label='커스텀 범위')
        if pid != CUSTOM:
            settings = apply(settings, side, pid)
    return settings, selections
