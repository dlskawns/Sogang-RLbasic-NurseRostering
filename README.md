# Sogang-RLbasic-NurseRostering

강화학습의 기초 프로젝트 - 간호사 근무표 생성

## RL 기반 간호사 근무표 실험 사용 방법

- 사전 준비

  - 파이썬 환경에서 `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch` 등이 설치되어 있어야 합니다.
  - `dataset_output/` 디렉토리에 `nurses.csv`, `requirements.csv`, `preferences.csv`가 존재해야 합니다.

- 1단계: CSV → 시나리오 피클 생성

  - 루트 디렉토리에서 아래 명령을 실행합니다.
  - 시나리오 ID는 CSV에 들어있는 값만 유효하며, `scenario_id=1`이 없을 수도 있습니다.

  ```bash
  cd /Users/david/Desktop/assignments/DataEngine/sogangRL/Sogang-RLbasic-NurseRostering
  python utils/data_loader_csv.py
  ```

- 2단계: 단일 알고리즘 개별 학습 실행

  - 같은 시나리오에 대해 알고리즘별로 각각 학습시키고 싶다면 아래와 같이 명령을 별도로 실행합니다.

  ```bash
  # Bandit
  python train.py --algo bandit --episodes 200 --seed 42 --scenario 5

  # DQN
  python train.py --algo dqn --episodes 200 --seed 42 --scenario 5

  # REINFORCE
  python train.py --algo reinforce --episodes 200 --seed 42 --scenario 5

  # PPO
  python train.py --algo ppo --episodes 200 --seed 42 --scenario 5
  ```

  - 학습 로그는 `logs/` 디렉토리에 `algo_seed*_*.csv` 형태로 저장됩니다.
  - DQN / REINFORCE / PPO는 에피소드마다 Hard Violation/Score 기준으로 최적 모델을 `models/<algo>/`에 저장합니다.

- 3단계: 모든 알고리즘 배치 학습 실행

  ```bash
  python train_all.py --episodes 200 --scenario 5
  ```

  - Bandit, DQN, REINFORCE, PPO가 순차적으로 실행되며, 각자의 로그와 모델이 `logs/`, `models/`에 쌓입니다.

- 4단계: 저장된 모델 평가

  - 학습이 완료된 뒤, 다음과 같이 각 알고리즘의 정책을 다양한 시나리오에 대해 평가할 수 있습니다.

  ```bash
  # DQN
  python eval_dqn.py --model models/dqn/dqn_scenario5_seed42.pth --scenarios 3 5 7 8 10 12 13 15

  # PPO
  python eval_ppo.py --model models/ppo/ppo_scenario5_seed42.pth --scenarios 3 5 7 8 10 12 13 15

  # REINFORCE
  python eval_reinforce.py --model models/reinforce/reinforce_scenario5_seed42.pth --scenarios 3 5 7 8 10 12 13 15

  # Bandit (모델 파일 없이 휴리스틱만 실행)
  python eval_bandit.py --scenarios 3 5 7 8 10 12 13 15
  ```

- 5단계: 학습 결과 시각화

  - `logs/`에 저장된 CSV들을 읽어 학습 곡선과 요약 바 차트를 생성합니다.

  ```bash
  python utils/plotter.py
  ```

  - 결과 이미지는 루트 디렉토리에 `results_learning_curves.png`, `results_summary_bar.png` 파일로 저장됩니다.

- 6단계: 최종 근무표 생성(추론)

  - 학습이 끝난 뒤, 선택한 알고리즘과 시나리오 ID에 대해 최종 근무표를 D/E/N/O 문자 매트릭스로 출력할 수 있습니다.
  - Bandit의 경우 파라미터가 없으므로 모델 경로 없이 실행하며, DQN/REINFORCE/PPO는 기본 모델 경로를 자동으로 참조합니다.

  ```bash
  # Bandit 기반 최종 근무표 생성 (scenario 5)
  python generate_roster.py --algo bandit --scenario 5 --max_steps 50

  # DQN 기반 최종 근무표 생성 (학습 시 사용한 scenario_id에 맞춰 경로 조정)
  python generate_roster.py --algo dqn --scenario 5 \
      --model models/dqn/dqn_scenario5_seed42.pth --max_steps 200

  # REINFORCE 기반
  python generate_roster.py --algo reinforce --scenario 5 \
      --model models/reinforce/reinforce_scenario5_seed42.pth --max_steps 200

  # PPO 기반
  python generate_roster.py --algo ppo --scenario 5 \
      --model models/ppo/ppo_scenario5_seed42.pth --max_steps 200
  ```

  - 출력 예시는 다음과 같이 각 간호사×일자 셀에 `O/D/E/N`이 깔끔하게 정렬된 형태입니다.

  ```text
  === Generated Nurse Roster ===
  - Algorithm   : BANDIT
  - Scenario ID : 5
  - Nurses      : 40
  - Days        : 31
  - Final Score : -2738.00
  - Hard Viol.  : 124
  - Shortage    : 263

  Nurse\Day | 01 02 03 04 ...
  Nurse_0   | N N E E D ...
  Nurse_1   | D N E O D ...
  ...
  ```
