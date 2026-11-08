"""Separate model, training-record, and match-result roots with legacy lookup."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=ROOT.parent
TRAINING_ROOT=ROOT/'training-results'
TRAINED_MODELS=PROJECT_ROOT/'학습모델'
MODEL_ROOT=TRAINED_MODELS/'일반'
CURRICULUM_MODEL_ROOT=TRAINED_MODELS/'상대순환'
GENERATIONAL_MODEL_ROOT=TRAINED_MODELS/'세대순환'
CRITICAL_MODEL_ROOT=TRAINED_MODELS/'피해-Critical'
EVALUATION_MODEL_ROOT=TRAINED_MODELS/'평가보관'
DECISION_MODEL_ROOT=TRAINED_MODELS/'단일판단'
MODEL_COPIES_ROOT=TRAINED_MODELS/'보관사본'
MATCH_ROOT=ROOT/'match-results'
REPORT_ROOT=ROOT/'storage-reports'
MIGRATIONS=REPORT_ROOT/'path-migrations.json'
LEGACY_MODEL_MOVES={
    ROOT/TRAINED_MODELS.name:TRAINED_MODELS,
    ROOT/'trained-models'/'competitive-ppo':MODEL_ROOT,
    ROOT/'trained-models'/'curriculum-ppo':CURRICULUM_MODEL_ROOT,
    ROOT/'trained-models'/'generational-ppo'/'damage-critical':CRITICAL_MODEL_ROOT,
    ROOT/'trained-models'/'generational-ppo':GENERATIONAL_MODEL_ROOT,
    ROOT/'trained-models'/'generational-extracts':EVALUATION_MODEL_ROOT,
    ROOT/'local-results'/'decision-ppo':DECISION_MODEL_ROOT,
    ROOT/'.important_models':MODEL_COPIES_ROOT,
}

def read_json(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {}

def resolve_path(path):
    path=Path(path).expanduser()
    path=(path if path.is_absolute() else ROOT/path).resolve()
    mappings={**read_json(MIGRATIONS).get('moves',{}), **{str(old):str(new) for old,new in LEGACY_MODEL_MOVES.items()}}
    visited=set()
    while path not in visited:
        # An older project-root redirect must not pull these folders back into core.
        if path.is_relative_to(TRAINED_MODELS) or path.is_relative_to(PROJECT_ROOT/'.important_models'):
            return path
        visited.add(path)
        for old,new in sorted(mappings.items(),key=lambda pair:len(pair[0]),reverse=True):
            old_path,new_path=Path(old),Path(new)
            # A move into a child folder must not repeatedly prepend that child.
            if old_path!=new_path and new_path.is_relative_to(old_path) and path.is_relative_to(new_path):continue
            try:updated=new_path/path.relative_to(old_path)
            except ValueError:continue
            if updated==path:return path
            path=updated
            break
        else:return path
    return path


def training_directory(path):
    path=resolve_path(path)
    manifest=read_json(path/'model-manifest.json')
    return resolve_path(manifest['trainingDirectory']) if manifest else path

def model_directory(path):
    path=resolve_path(path)
    if (path/'model-manifest.json').exists():return path
    metadata=read_json(path/'training.json')
    return resolve_path(metadata['modelDirectory']) if metadata.get('modelDirectory') else path

def model_lock(path):
    from filelock import FileLock
    directory=model_directory(path);directory.mkdir(parents=True,exist_ok=True)
    return FileLock(str(directory/'.checkpoint.lock'),timeout=60)


def checkpoint_directory(model):
    """Find the run that owns a selected model or one of its snapshots."""
    path=resolve_path(model)
    return next((parent for parent in path.parents if (parent/'model-manifest.json').is_file()),None)
