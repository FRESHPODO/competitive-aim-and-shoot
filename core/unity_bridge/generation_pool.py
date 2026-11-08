"""Bounded recent-policy window. Generation cadence is measured in finished games."""
import hashlib
import shutil
from pathlib import Path
from unity_bridge.trainer_state import save_state

BOOTSTRAP = dict(kind='always_approve', generation=0, name='무보류 상대')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(opponents, rng):
    return dict(opponents[int(rng.integers(len(opponents)))]) if opponents else dict(BOOTSTRAP)


def publish(model, model_root, number, episodes, previous, keep):
    if keep not in (2, 3):
        raise ValueError('Expected 2 or 3 recent generations')
    root = Path(model_root).resolve()
    folder = root / 'generations' / f'{number:06d}'
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'learner.zip'
    model.save(path)
    entry = dict(kind='frozen_policy', generation=number, episodes=episodes,
                 name=f'{number}세대 · {episodes:,}경기', relativePath=path.relative_to(root).as_posix(), sha256=digest(path))
    save_state(folder / 'complete.json', entry)
    return (previous + [entry])[-keep:]


def materialize(opponents, model_root):
    root = Path(model_root).resolve()
    result = []
    for entry in opponents:
        path = (root / entry['relativePath']).resolve()
        if not path.is_relative_to(root / 'generations') or not path.is_file() or digest(path) != entry['sha256']:
            raise ValueError('상대 세대 모델이 없거나 저장 당시 내용과 다릅니다.')
        result.append(dict(entry, model=str(path)))
    return result


def prune(model_root, opponents):
    """Call only after a complete recovery checkpoint references the new window."""
    root = Path(model_root).resolve() / 'generations'
    retained = {entry['generation'] for entry in opponents}
    if not root.is_dir():
        return
    for folder in root.iterdir():
        # Delete only this trainer's numbered generation folders, inside its root.
        resolved = folder.resolve()
        if not folder.is_dir() or not folder.name.isdigit() or not resolved.is_relative_to(root):
            continue
        if int(folder.name) not in retained:
            shutil.rmtree(resolved)
