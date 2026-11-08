# 피해·Critical 보상을 추가한 세대 학습기

최상위 `Start-Reward-Generational-Trainer.cmd`로 실행한다.
기존 세대 학습기를 복사한 별도 프로그램으로, 기존 학습·모델·설정은 변경하지 않는다.

## 보상

`총 보상 = 기존 α 처치 보상 + βD × 실제 누적 피해 / 최대 체력 + βC × Critical 처치 여부`

- 자신이 처치하면 `+2α`, 상대가 처치하면 `−2(1−α)`, 무승부는 기존 처치 보상 0.
- `βD`, `βC`는 각각 0~1. Critical 보상은 α와 독립된 추가 보상이다.
- 피해는 실제 감소한 체력만 인정한다. 마지막 샷의 초과 피해는 제외한다.
- 동시 명중으로 초과 피해가 발생하면 실제 감소량을 해당 시점의 피해량에 비례해 분배한다.
- Critical **명중**이 아니라 자신이 Critical로 **처치**한 경우에만 βC를 더한다.
- 기존 구조처럼 라운드 총 보상을 마지막 PPO 판단에 기록한다. 시간 비용·빗나감 비용은 추가하지 않는다.
- `episodes.jsonl`에 라운드별 승패, 보상 계수, `rewardComponents`, 실제 경기 조건 및 프리셋 ID를 저장한다. CLI의 `--record-episodes`는 전체 시뮬레이터 입력과 상세 프리셋 설명도 저장한다. 재개 시 체크포인트의 경기 수를 기준으로 그 이후의 미확정 기록을 제거한다.

## 입력·설정

기존 22개 입력에 자신의 `βD`, `βC`만 추가한 관측 버전 8, 총 24개 입력이다.
상대의 세 보상 계수는 공개하지 않는다. 조준은 고정 모델이며 PPO는 발사 요청의 승인·거절만 결정한다.
쿨다운·기본 피해·Critical 즉사 설정은 양쪽 공통이다. 이전 경기 결과는 관측에 포함하지 않는다.

화면에서 A/B의 두 계수를 각각 고정 또는 범위·추출 단위로 설정한다.
초기 범위는 각각 0~1, 단위 0.1이다. 기존 화면 초안을 별도 설정 파일로 복사하되 이어 학습 경로는 비웠다.
`configs/reward_generational_training.example.json`은 최신 기존 학습 작업의 조건을 복사한 설정 예시다.

최근 2~3세대 상대, 첫 세대 전 무보류 상대, 세대 교체 주기, 중단·저장·복구 구조는 유지한다.
새 관측 구조이므로 기존 v7 모델을 이어 학습 모델이나 상대 모델로 사용하지 않는다.
대전은 최상위 `Start-Match-Simulator.cmd`에서 실행한다. 양쪽 모델 파일을 선택하고 A/B 각각의 α·누적 피해·Critical 계수를 설정할 수 있다. 대전에서는 v7과 v8 모델을 함께 선택할 수 있다.

완료 작업을 이어서 학습하려면 이 학습기에서 작업 기록 또는 모델 폴더를 선택하고, 총 학습 경기 수를 저장된 경기 수보다 큰 값으로 변경한 뒤 **이어서 학습**을 누른다. 기존 조건과 복구 체크포인트를 사용한다. 선택된 작업의 `learner.zip` 전체 경로는 화면에서 복사할 수 있다.

## 분리된 저장 위치

- 설정: `app-settings/generational-trainer-damage-critical.json`
- 요청: `app-settings/generational-requests/damage-critical/`
- 기록: `generational-results/damage-critical/`
- 모델: 프로젝트 최상위 `학습모델/피해-Critical/`
- 로그: `generational-logs/damage-critical/`

CLI 실행: `python -m unity_bridge.reward_generation.train --config configs/reward_generational_training.example.json --output generational-results/damage-critical/새폴더`

검증: `python -m unity_bridge.reward_generation.check`
