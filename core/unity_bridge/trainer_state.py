"""Durable UI drafts and consistent paired-checkpoint discovery (no ML imports)."""
import json
from pathlib import Path
import uuid
import shutil
from unity_bridge.storage import training_directory,model_directory,model_lock
from unity_bridge.atomic_io import replace_with_retry


def save_state(path, state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.'+uuid.uuid4().hex+'.tmp.json')
    try:
        tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
        replace_with_retry(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def load_state(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return None


def checkpoint(run):
    records=training_directory(run);run=model_directory(run)
    live=[]
    for marker in (run/'live').glob('*/complete.json'):
        data=load_state(marker)
        if data and all((marker.parent/f'agent_{s}.zip').is_file() for s in ('a','b')):live.append((data['episodes'],marker.parent,data))
    if live:
        _,source,data=max(live,key=lambda item:item[0]);return source,data
    metadata=load_state(records/'training.json') or {}
    completed=metadata.get('completedEpisodes')
    if isinstance(completed,int):
        latest=run/'snapshots'/f'{completed:08d}';data=load_state(latest/'complete.json')
        if data and all((latest/f'agent_{s}.zip').is_file() for s in ('a','b')):return latest,data
    candidates=sorted((run/'snapshots').glob('*/complete.json'),reverse=True)
    for marker in candidates:
        data=load_state(marker)
        if data and all((marker.parent/f'agent_{s}.zip').is_file() for s in ('a','b')):
            return marker.parent,data
    metadata=load_state(records/'training.json')
    if metadata and 'completedEpisodes' in metadata and all((run/f'agent_{s}.zip').is_file() for s in ('a','b')):
        return run,dict(episodes=metadata['completedEpisodes'])
    raise ValueError('이어 학습 가능한 양측 체크포인트가 없습니다.')

def copy_checkpoint(run,destination):
    """Copy an immutable paired view while excluding a writer rotating live slots."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with model_lock(run):
        source,data=checkpoint(run)
        for name in ('agent_a.zip','agent_b.zip','torch_rng.pt','cuda_rng.pt','complete.json'):
            if (source/name).is_file():shutil.copy2(source/name,destination/name)
    return destination,data
