# Competitive Aim-and-Shoot

학습된 Aim-and-Shoot 정책 두 개를 같은 표적·같은 시각에 실행하고, 조건별 조준·발사 행동을 관찰하는 Unity 시뮬레이터입니다.

원본: [jsyoon-k5/Aim-and-Shoot](https://github.com/jsyoon-k5/Aim-and-Shoot). 원본 코드·모델과 실험 데이터를 기반으로, 추가된 경쟁 환경과 Unity 화면을 함께 제공합니다. [원본 README](docs/UPSTREAM-README.md).

## 주요 기능

- 관찰자 화면과 에이전트 A/B 개인 화면. 동일 위치에서 시작하며 개인 시선·조준은 독립적입니다.
- 공통 시각에 따른 이동 표적, 체력, 중심 오차에 따른 피해, 즉사 영역, 처치 점수.
- 에이전트별 8개 인지·보상 파라미터, Shot cooldown, 원뿔 분포 탄퍼짐.
- 발사 후 초기화 없이 추적 유지. Cooldown 중 발사 결정만 차단하고 종료 후 다음 판단부터 재허용.
- 개인 화면 중앙 쪽에 점수·현재 타겟 발사 수·누적 피해 표시. 즉사 처치 시 CRITICAL 표시.
- 타겟 파괴 후 기본 1.5초 결과 정지와 타겟별 CSV 기록.

**기존 정책을 경쟁 환경에서 재사용하는 관찰 도구입니다. 체력·상대 점수·탄퍼짐·쿨다운에 최적화하도록 재학습한 경쟁 정책은 아닙니다.**

## 설치 및 실행 (Windows)

Python 3.12와 Unity Editor **6000.3.22f1**을 사용해 확인했습니다. Unity에는 Windows 빌드 모듈과 유효한 Editor 라이선스가 필요합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-installed.txt
.\Build-Unity-Simulator.ps1 -UnityEditorPath "C:\path\to\Editor\Unity.exe"
.\Start-Unity-Simulator.ps1
```

`requirements-installed.txt`는 검증 당시 환경의 고정 버전 목록입니다. CPU용 PyTorch는 필요한 경우 공식 CPU 배포에서 별도 설치할 수 있습니다. 학습된 기본 모델은 `data/ans_agent_models/ijhcs_default/checkpoints/rl_model_20000000_steps.zip`에 포함되어 있습니다.

빌드 이후에는 `Start-Unity-Simulator.cmd`를 실행하고 **START BOTH AGENTS**를 누릅니다. 실행 파일·가상환경은 Git에 포함하지 않으므로 새 PC에서는 설치·빌드가 필요합니다.

## 기본값

| 항목 | 기본값 |
|---|---:|
| Playback speed | 0.5배 (0.1배 간격) |
| Radius | 5° |
| Health | 1000 (최대 1000) |
| Edge / Center damage | 100 / 300 |
| Instant-kill 반지름 | 타겟 반지름의 1% |
| Shot cooldown | 에이전트별 150ms |
| Bullet spread | 에이전트별 0° |
| 이동 속도 | 표적마다 0–40°/s 균등 추출 |
| 이동 방향 | 표적마다 원본 분포에서 무작위 |

피해 수치는 슬라이더 없이 직접 입력하며 상한이 없습니다. 변경된 표적·에이전트 조건은 다음 타겟부터 적용합니다.

## 구성과 검증

```text
UnitySimulator/       Unity 화면, 씬, 빌더 및 프로젝트 설정
unity_bridge/         Python 모델 연결, 공유 표적/피해/점수, 회귀 검증
src/ configs/ data/   원본 모델 및 시뮬레이션 코드·설정·체크포인트
empirical/           원본 공개 실험 데이터 (대용량 60Hz 궤적은 선택 다운로드)
Build-Unity-Simulator.ps1   Windows 실행 파일 빌드
Start-Unity-Simulator.ps1   로컬 모델 서버와 실행 파일 시작
```

```powershell
.\.venv\Scripts\python.exe -m unity_bridge.check_combat
.\.venv\Scripts\python.exe -m unity_bridge.check_moving
.\.venv\Scripts\python.exe -m unity_bridge.check_spread
.\.venv\Scripts\python.exe -m unity_bridge.check_cooldown
```

[조작·결과 기록 안내](Unity-실행안내.md) · [구현 범위와 한계](SIMULATOR-SCOPE.md) · [GitHub Pages 실행 가능성 검토](docs/GITHUB-PAGES.md)

## 선택 실험 데이터

Unity 실시간 실행에는 필요하지 않은 원본 `empirical/trajectories_60hz.h5`는 대용량 파일로 별도 다운로드합니다. 원본 커밋을 고정하고 SHA-256으로 검증합니다. 나머지 실행 코드와 학습 모델은 저장소에 포함됩니다.

```powershell
.\.venv\Scripts\python.exe -m unity_bridge.download_upstream_data
```

## 출처와 이용 조건

원본 프로젝트와 모델의 저작권·이용 조건은 원저자에게 있습니다. 이 저장소는 원본 자료에 새로운 재배포 라이선스를 부여하지 않습니다. 원본 공개 저장소의 현재 조건을 확인하고 이용하세요.
