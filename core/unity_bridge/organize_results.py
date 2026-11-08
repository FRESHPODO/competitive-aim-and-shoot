"""Archive generated verification clutter without deleting results or models."""
import argparse,json,shutil
from datetime import datetime
from pathlib import Path
from unity_bridge.trainer_state import save_state

ROOT=Path(__file__).resolve().parents[1]/'local-results'

def plan():
    keep={'competitive','custom-duels','decision-ppo','experiments','resource-control','archive','logs'}
    operations=[]
    for source in sorted(ROOT.iterdir()):
        if source.is_dir():
            if source.name in keep:continue
            target=ROOT/'experiments'/source.name if source.name in ('paper-review','parameter-pilot-20261029') else ROOT/'archive'/'verification'/source.name
        else:
            if source.name in ('trainer-state.json','match-simulator-state.json','README.md','organization-manifest.json'):continue
            if source.name.lower().startswith(('unity','compile','upload-manifest')):target=ROOT/'archive'/'legacy-unity'/source.name
            elif source.suffix=='.log':target=ROOT/'logs'/'legacy'/source.name
            else:target=ROOT/'archive'/'verification'/'files'/source.name
        source=source.resolve();target=target.resolve()
        if not source.is_relative_to(ROOT.resolve()) or not target.is_relative_to(ROOT.resolve()):raise ValueError('Move outside results root')
        if target.exists():raise FileExistsError(target)
        operations.append((source,target))
    return operations

def organize():
    moves=plan();mapping={str(a):str(b) for a,b in moves}
    manifest=ROOT/'organization-manifest.json'
    previous=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {'moves':{}}
    previous['moves'].update(mapping);previous['updated']=datetime.now().isoformat()
    # Record the map first so interrupted moves remain reviewable.
    save_state(manifest,previous)
    for source,target in moves:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(target))
    replacements=[]
    for source,target in mapping.items():
        replacements.extend([(source,target),(source.replace('\\','/'),target.replace('\\','/'))])
    def rewrite(value):
        if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
        if isinstance(value,list):return [rewrite(v) for v in value]
        if isinstance(value,str):
            for old,new in replacements:value=value.replace(old,new)
        return value
    changed=0
    for path in ROOT.rglob('*.json'):
        if path==manifest or path.stat().st_size>5_000_000:continue
        try:data=json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError,UnicodeError):continue
        new=rewrite(data)
        if new!=data:save_state(path,new);changed+=1
    for _,target in moves:
        if not target.is_dir():continue
        for path in target.rglob('*.jsonl'):
            lines=[];dirty=False
            for line in path.read_text(encoding='utf-8').splitlines():
                data=json.loads(line);new=rewrite(data);dirty|=new!=data;lines.append(json.dumps(new,ensure_ascii=False))
            if dirty:path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'README.md').write_text('''# 실행 결과 폴더

- `competitive/`: 실제 경쟁 학습, 모델, 체크포인트, 경기 기록
- `custom-duels/`: 대전 시뮬레이터 결과와 리플레이
- `decision-ppo/`: 기존 의사결정 모델
- `experiments/`: 실험 및 논문 검토 자료
- `logs/`: 프로그램 실행 로그
- `resource-control/`: 선택적 GPU 작업의 공용 사용률 제어
- `archive/verification/`: 개발 중 검증·스모크 테스트 및 그 모델
- `archive/legacy-unity/`: 과거 Unity 빌드 기록
- `trainer-state.json`, `match-simulator-state.json`: 현재 화면 설정
- `organization-manifest.json`: 이동 전후 경로 대응표

정리 과정에서 결과나 모델을 삭제하지 않았습니다. 기존 학습 및 대전 폴더는 이동하지 않았습니다.
''',encoding='utf-8')
    return {'moved':len(moves),'updatedJsonReferences':changed}

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.apply:print(json.dumps(organize()))
    else:
        for source,target in plan():print(f'{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}')
if __name__=='__main__':main()
