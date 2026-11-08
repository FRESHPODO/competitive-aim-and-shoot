"""List and atomically activate independently trained decision checkpoints."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unity_bridge.fire_decider import MODEL_DIR, MODEL_PATH, FireDecider, FEATURE_NAMES
from unity_bridge.atomic_io import replace_with_retry


def checkpoints():
    return sorted((p for p in MODEL_DIR.rglob('*.zip') if '.tmp' not in p.name),
                  key=lambda p:p.stat().st_mtime,reverse=True)


def active_info():
    path=MODEL_DIR/'active.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def model_catalog():
    models=[]
    for path in checkpoints():
        if path==MODEL_PATH: continue
        try:
            with zipfile.ZipFile(path) as archive:
                data=json.loads(archive.read('data'))
            if tuple(data.get('decision_feature_names',())) != FEATURE_NAMES:
                continue
            identifier=path.relative_to(MODEL_DIR).as_posix()
            label=path.stem+' ['+path.parent.name+']'
            models.append(dict(id=identifier,label=label,decisions=int(data['num_timesteps'])))
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):
            continue
    return dict(models=models,activeSource=active_info().get('source',''))


def resolve_model(identifier):
    if not identifier: return MODEL_PATH
    # Select only registered local checkpoints; never accept an arbitrary path.
    allowed={path.relative_to(MODEL_DIR).as_posix():path for path in checkpoints()}
    if identifier not in allowed: raise ValueError('Unknown decision model; refresh model list')
    return allowed[identifier]


def activate(source):
    # Only read the source. Avoid Windows final-path handle queries, which may
    # be denied for a readable checkpoint in restricted desktop environments.
    source=Path(os.path.abspath(source))
    if not source.is_file(): raise FileNotFoundError(source)
    if os.path.normcase(str(source))==os.path.normcase(str(MODEL_PATH.absolute())):
        return active_info()
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    handle, temporary=tempfile.mkstemp(prefix='activate-',suffix='.tmp.zip',dir=MODEL_DIR)
    os.close(handle)
    temporary=Path(temporary)
    try:
        shutil.copyfile(source,temporary)
        verified=FireDecider(temporary)
        metadata=dict(source=str(source),sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
                      decisions=int(verified.model.num_timesteps),
                      activatedAt=datetime.now(timezone.utc).isoformat())
        replace_with_retry(temporary,MODEL_PATH)
        manifest=MODEL_DIR/'active.tmp.json'
        manifest.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        replace_with_retry(manifest,MODEL_DIR/'active.json')
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['list','activate'])
    parser.add_argument('checkpoint',nargs='?')
    args=parser.parse_args()
    if args.command=='list':
        print(json.dumps(dict(active=active_info(),checkpoints=[str(p) for p in checkpoints()]),indent=2))
    else:
        if not args.checkpoint: parser.error('activate requires a checkpoint')
        print(json.dumps(activate(args.checkpoint),indent=2))


if __name__=='__main__': main()
