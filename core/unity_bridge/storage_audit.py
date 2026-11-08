"""Read-only size inventory and explicit cleanup candidates; never deletes data."""
import csv,json,os,re,time
from pathlib import Path
from unity_bridge.storage import ROOT,MODEL_ROOT,TRAINING_ROOT,MATCH_ROOT,REPORT_ROOT,read_json

def size_of(path):
    count=0;size=0
    if path.is_file():return path.stat().st_size,1
    for directory,_,files in os.walk(path):
        for name in files:
            try:size+=(Path(directory)/name).stat().st_size;count+=1
            except OSError:pass
    return size,count

def main():
    REPORT_ROOT.mkdir(parents=True,exist_ok=True)
    references=set()
    for base in [TRAINING_ROOT,MATCH_ROOT,ROOT/'local-results',ROOT/'configs',ROOT/'.important_models']:
        for path in base.rglob('*.json'):
            if path.stat().st_size>5_000_000 or 'snapshots' in path.parts:continue
            try:text=path.read_text(encoding='utf-8-sig')
            except (OSError,UnicodeError):continue
            references.update(re.findall(r'(?:competitive|competitive-ppo)[\\/]+([^\\/"\s]+)[\\/]+snapshots[\\/]+(\d+)',text))
    candidates=[];runs=[]
    for models in sorted(MODEL_ROOT.glob('*')):
        if not models.is_dir():continue
        snapshots=sorted(p for p in (models/'snapshots').glob('*') if p.is_dir() and p.name.isdigit())
        protected={p.name for p in snapshots[-2:]}|({snapshots[0].name} if snapshots else set())
        protected.update(p.name for p in snapshots if int(p.name)%10000==0 or (models.name,p.name) in references)
        total=0;candidate_size=0;candidate_count=0
        for path in snapshots:
            size,count=size_of(path);total+=size
            if path.name not in protected:
                candidates.append(dict(path=str(path),bytes=size,files=count,reason='과도하게 촘촘한 과거 배치 체크포인트 (최근 2개·초기·1만 경기 간격·명시 참조 보존)'))
                candidate_size+=size;candidate_count+=1
        runs.append(dict(run=models.name,snapshots=len(snapshots),snapshotBytes=total,protectedSnapshots=len(protected),candidateSnapshots=candidate_count,candidateBytes=candidate_size))
    inventory=[]
    for path in [MODEL_ROOT.parent,TRAINING_ROOT,MATCH_ROOT,ROOT/'local-results',ROOT/'.important_models',ROOT/'.venv']:
        size,count=size_of(path);inventory.append(dict(path=str(path.relative_to(ROOT)),bytes=size,files=count))
    for path,reason in [(ROOT/'local-results/archive/verification','개발 검증 기록·검증용 모델: 실제 학습/대전 보존 여부와 별개로 검토 가능'),
                        (ROOT/'local-results/archive/legacy-unity','사용하지 않는 이전 Unity 빌드/업로드 기록'),
                        (ROOT/'.venv/Lib/site-packages/~orch','PyTorch 교체 시 남은 pip 임시 패키지: 실행 환경 확인 후 정리 가능')]:
        if path.exists():
            size,count=size_of(path);candidates.append(dict(path=str(path),bytes=size,files=count,reason=reason))
    report=dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'),inventory=inventory,checkpointRuns=runs,candidateBytes=sum(c['bytes'] for c in candidates),candidates=candidates,deletedFiles=0)
    (REPORT_ROOT/'storage-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (REPORT_ROOT/'cleanup-candidates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','files','reason']);writer.writeheader();writer.writerows(candidates)
    lines=['# 저장 용량 점검','', '단위: GB = 1,000,000,000바이트. 이 보고서는 파일을 삭제하지 않습니다.','', '| 폴더 | 용량 (GB) | 파일 수 |','|---|---:|---:|']
    lines += [f"| {r['path']} | {r['bytes']/1e9:.3f} | {r['files']:,} |" for r in inventory]
    lines += ['', '## 구분', '', '- 보존: 현재 모델 A/B, 최신 복구본, 초기 모델, 10,000경기 간격 체크포인트, 설정에서 명시 참조하는 체크포인트, `.important_models`의 중요 모델.',
              '- 정리 후보: 위 보존 조건을 제외한 과거 배치 체크포인트. 모델이 모두 동일한 복사본이라는 뜻은 아니며, 중간 학습 이력 보관 간격을 줄일 때 삭제할 수 있는 후보입니다.',
              '- 별도 검토 후보: 개발 검증 결과, 과거 Unity 기록, pip 교체 잔여물. 학습 결과나 필수 실행 패키지와 구분합니다.',
              '- 학습 경기 JSONL·설정·진행 기록과 대전 결과·리플레이는 보존합니다.',
              '',f"정리 후보 합계: **{report['candidateBytes']/1e9:.3f} GB**. 실제 삭제: **0개**.", '',
              '## 모델별 과거 체크포인트', '', '| 학습 작업 | 과거 체크포인트 수 | 보존 수 | 후보 수 | 후보 용량 (GB) |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['run']} | {r['snapshots']:,} | {r['protectedSnapshots']:,} | {r['candidateSnapshots']:,} | {r['candidateBytes']/1e9:.3f} |" for r in runs]
    lines += ['', '개별 경로와 용량은 `cleanup-candidates.csv`, 기계 판독용 전체 결과는 `storage-audit.json`을 참조하세요.']
    (REPORT_ROOT/'storage-audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(inventory=inventory,candidateBytes=report['candidateBytes'],checkpointCandidateBytes=sum(r['candidateBytes'] for r in runs),deletedFiles=0),ensure_ascii=False))

if __name__=='__main__':main()
