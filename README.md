# Competitive Aim-and-Shoot

공유 표적을 두고 대전하는 조준·발사 에이전트의 PPO 학습기와 경기 시뮬레이터입니다. 학습기와 리플레이는 Windows에서 Python으로 실행합니다.

## 설치

Tcl/Tk가 포함된 Python을 설치하고, 저장소 루트에서 아래 명령을 실행합니다.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 실행

| 실행 파일 | 용도 |
|---|---|
| [Start-Decision-Trainer.cmd](Start-Decision-Trainer.cmd) | 일반 PPO 학습 |
| [Start-Curriculum-Trainer.cmd](Start-Curriculum-Trainer.cmd) | 고정 상대 목록을 교대하는 학습 |
| [Start-Generational-Trainer.cmd](Start-Generational-Trainer.cmd) | 세대 순환 학습 |
| [Start-Reward-Generational-Trainer.cmd](Start-Reward-Generational-Trainer.cmd) | 누적 피해와 크리티컬 처치 보상을 사용하는 세대 순환 학습 |
| [Start-Match-Simulator.cmd](Start-Match-Simulator.cmd) | 모델 대전 및 리플레이 |

실행 코드는 [core](core)에, 완료 모델과 관측 버전은 [학습모델](학습모델/README.md)에 있습니다. 대전 시뮬레이터에서 관측 v7 ZIP을 A/B 모델로 선택할 수 있습니다.

기반 조준 코드와 사전 학습 모델은 [jsyoon-k5/Aim-and-Shoot](https://github.com/jsyoon-k5/Aim-and-Shoot)를 사용합니다.
