"""Ordered frozen opponents, with immutable per-run checkpoint snapshots."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

from unity_bridge.player_presets import PRESETS as PLAYERS, apply, provenance
from unity_bridge.weapon_presets import PRESETS as WEAPONS
from unity_bridge.storage import resolve_path


def normalize(config, *, check_files=True):
    config = deepcopy(config or {})
    config.setdefault('enabled', False)
    config.setdefault('learnerSide', 'A')
    config.setdefault('opponents', [])
    if type(config['enabled']) is not bool or not isinstance(config['opponents'],list):
        raise ValueError('상대 목록 설정의 enabled는 참/거짓, opponents는 목록이어야 합니다.')
    if not config['enabled']:
        return config
    if config['learnerSide'] != 'A':
        raise ValueError('현재 상대 목록 학습은 A 학습 / B 고정 상대 방식입니다.')
    if not config['opponents']:
        raise ValueError('고정 상대 목록에 모델을 하나 이상 추가하세요.')
    for index, entry in enumerate(config['opponents']):
        if not isinstance(entry,dict):raise ValueError('각 상대 항목은 모델과 프리셋 설정이어야 합니다.')
        entry['name'] = str(entry.get('name') or f'상대 {index+1}')
        if not entry.get('model'):
            raise ValueError(f'{entry["name"]}: PPO 모델 파일을 선택하세요.')
        if check_files:
            path = resolve_path(entry['model'])
            if not path.is_file():
                raise ValueError(f'{entry["name"]}: 모델 파일이 없습니다: {path}')
            entry['model'] = str(path)
        if entry.get('presetId') and entry['presetId'] not in PLAYERS:
            raise ValueError(f'{entry["name"]}: 알 수 없는 플레이어 프리셋')
        if entry.get('weaponPreset') and WEAPONS.get(entry['weaponPreset']) is None:
            raise ValueError(f'{entry["name"]}: 알 수 없는 무기 프리셋')
    return config


def prepare(config, output):
    config = normalize(config)
    if not config['enabled']:
        return config
    from unity_bridge.fire_decider import FireDecider
    folder = Path(output) / 'opponent-inputs'
    folder.mkdir(exist_ok=True)
    for index, entry in enumerate(config['opponents']):
        source = resolve_path(entry['model'])
        destination = folder / f'opponent-{index:03d}.zip'
        shutil.copyfile(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if entry.get('sha256') and entry['sha256'] != digest:
            raise ValueError(f'{entry["name"]}: 고정 상대 체크포인트 내용이 변경되었습니다.')
        # Fail early on incompatible checkpoint schemas, before collecting games.
        FireDecider(destination)
        entry.update(source=str(source), model=str(destination.resolve()), sha256=digest)
    return config


def select(config, episode_index):
    if not config or not config.get('enabled'):
        return None
    index = episode_index % len(config['opponents'])
    return dict(config['opponents'][index], index=index, side='B')


def scenario(settings, presets, entry):
    if entry is None:
        return settings, presets
    settings = deepcopy(settings)
    presets = deepcopy(presets)
    side = entry['side']
    if entry.get('presetId'):
        settings = apply(settings, side, entry['presetId'])
        presets[side] = provenance(entry['presetId'])
    if entry.get('weaponPreset'):
        for key, value in WEAPONS[entry['weaponPreset']].items():
            settings[f'{side}.{key}'] = deepcopy(value)
    return settings, presets
