"""Compact per-game results shared by all competitive trainers."""
import json
import mmap


def write_episode_result(stream, row, *, detailed=False):
    if detailed:
        result = row
    else:
        keys = ('episode', 'seed', 'outcome', 'winner', 'timeMs', 'rewards',
                'decisions', 'approvals', 'shots', 'learnerGeneration', 'learningEnabled',
                'rewardComponents')
        result = {key: row[key] for key in keys if key in row}
        result['rewardValues'] = {
            side: {key: values[key] for key in ('alpha', 'damageFactor', 'criticalFactor') if key in values}
            for side, values in row['rewardValues'].items()
        }
        request = row['settings']
        conditions = ('targetHp', 'targetSpeed', 'targetRadius', 'criticalRadius',
                      'minDamage', 'maxDamage', 'instantKill', 'cooldown', 'shotSpreadDeg', 'profiles')
        result['settings'] = {key: request[key] for key in conditions if key in request}
        presets = row.get('playerPresets', request.get('playerPresets', {}))
        result['playerPresets'] = {
            side: {'id': preset['id']} for side, preset in presets.items()
            if isinstance(preset, dict) and 'id' in preset
        }
        if 'opponent' in row:
            result['opponent'] = {key: row['opponent'][key] for key in ('kind', 'generation')}
        if row.get('opponentPool'):
            opponent = row['opponentPool']
            result['opponentPool'] = {key: opponent[key] for key in ('index', 'presetId', 'weaponPreset')}
    stream.write(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')


def trim_episode_results(path, completed_episodes):
    """On resume, remove only the tail beyond the restored checkpoint."""
    if not path.exists():
        return
    with path.open('r+b') as stream:
        if stream.seek(0, 2) == 0:
            return
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as view:
            end = view.rfind(b'\n') + 1
            while end:
                start = view.rfind(b'\n', 0, end - 1) + 1
                try:
                    row = json.loads(view[start:end])
                except (ValueError, UnicodeDecodeError):
                    row = None
                episode = row.get('episode') if isinstance(row, dict) else None
                if type(episode) is int and episode <= completed_episodes:
                    break
                end = start
        stream.truncate(end)
