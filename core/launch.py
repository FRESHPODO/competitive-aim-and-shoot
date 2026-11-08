"""Run the relocated app and repair saved internal paths without scanning snapshots."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from unity_bridge.storage import MODEL_COPIES_ROOT, TRAINED_MODELS
for name, folder in [('TCL_LIBRARY', 'tcl8.6'), ('TK_LIBRARY', 'tk8.6')]:
    location = Path(sys.base_prefix) / 'tcl' / folder
    if location.is_dir():
        os.environ[name] = str(location)
    else:
        os.environ.pop(name, None)

marker = ROOT / 'local-results/portable-root.json'
previous = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else str(ROOT)
external_models = Path(previous) / '.important_models'
previous_trained_models = Path(previous).parent / TRAINED_MODELS.name


def rebase(value):
    if isinstance(value, str):
        for prefix in (str(previous_trained_models), previous_trained_models.as_posix()):
            if value == prefix or value.startswith(prefix + '/') or value.startswith(prefix + '\\'):
                return str(TRAINED_MODELS) + value[len(prefix):]
        # The user deliberately keeps these models outside the application.
        for prefix in (str(external_models), external_models.as_posix()):
            if value == prefix or value.startswith(prefix + '/') or value.startswith(prefix + '\\'):
                base = external_models if external_models.is_dir() else MODEL_COPIES_ROOT
                return str(base) + value[len(prefix):]
        for prefix in (previous, previous.replace('\\', '/')):
            if value == prefix or value.startswith(prefix + '/') or value.startswith(prefix + '\\'):
                return str(ROOT) + value[len(prefix):]
    if isinstance(value, list):
        return [rebase(item) for item in value]
    if isinstance(value, dict):
        return {key: rebase(item) for key, item in value.items()}
    return value


if previous != str(ROOT):
    migration = ROOT / 'storage-reports/path-migrations.json'
    document = json.loads(migration.read_text(encoding='utf-8-sig')) if migration.exists() else {}
    moves = {old: rebase(new) for old, new in document.get('moves', {}).items()}
    moves[previous] = str(ROOT)
    moves[str(external_models)] = str(external_models if external_models.is_dir() else MODEL_COPIES_ROOT)
    moves[str(previous_trained_models)] = str(TRAINED_MODELS)
    document['moves'] = moves
    migration.parent.mkdir(exist_ok=True)
    migration.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = list((ROOT / 'local-results').glob('*.json'))
    paths.extend((ROOT / 'app-settings').rglob('*.json'))
    for folder in ('training-results', 'match-results', 'curriculum-results', 'generational-results'):
        paths.extend((ROOT / folder).glob('*/*.json'))
    paths.extend((ROOT / 'generational-results' / 'damage-critical').glob('*/*.json'))
    paths.extend(TRAINED_MODELS.glob('*/*/model-manifest.json'))
    for path in paths:
        if path == marker:
            continue
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        updated = rebase(data)
        if updated != data:
            path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding='utf-8')
marker.parent.mkdir(exist_ok=True)
marker.write_text(json.dumps(str(ROOT)), encoding='utf-8')

if __name__ == '__main__':
    if '--reward-generational' in sys.argv:
        sys.argv.remove('--reward-generational')
        from unity_bridge.reward_generation.ui import main
    elif '--generational' in sys.argv:
        sys.argv.remove('--generational')
        from unity_bridge.generation_trainer import main
    elif '--curriculum' in sys.argv:
        sys.argv.remove('--curriculum')
        from unity_bridge.curriculum_trainer import main
    elif '--matches' in sys.argv:
        sys.argv.remove('--matches')
        from unity_bridge.match_simulator import main
    else:
        from unity_bridge.decision_trainer import main
    main()
