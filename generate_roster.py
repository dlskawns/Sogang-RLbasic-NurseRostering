import argparse
import os
from typing import Literal

import numpy as np

from envs.roster_env import NurseRosterEnv
from agents.baseline_bandit import GreedyBanditAgent
from agents.dqn import DQNAgent
from agents.reinforce import ReinforceAgent
from agents.ppo import PPOAgent


SHIFT_SYMBOLS = {
    0: "O",  # Off
    1: "D",  # Day
    2: "E",  # Evening
    3: "N",  # Night
}


AlgoType = Literal["bandit", "dqn", "reinforce", "ppo"]


def build_agent(algo: AlgoType, env: NurseRosterEnv, model_path: str | None) -> object:
    """
    알고리즘 이름과 환경, 모델 경로를 받아 적절한 에이전트를 생성합니다.

    Args:
        algo (str): 사용할 알고리즘 이름 ('bandit', 'dqn', 'reinforce', 'ppo').
        env (NurseRosterEnv): 시나리오가 로드된 환경 인스턴스.
        model_path (str | None): 학습된 모델 파라미터 경로. Bandit의 경우 사용되지 않습니다.

    Returns:
        object: 알고리즘에 대응하는 에이전트 인스턴스.
    """
    if algo == "bandit":
        return GreedyBanditAgent(env, epsilon=0.3)

    if algo == "dqn":
        agent = DQNAgent(env)
    elif algo == "reinforce":
        agent = ReinforceAgent(env)
    elif algo == "ppo":
        agent = PPOAgent(env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    if model_path is not None:
        if not os.path.exists(model_path):
            msg = f"[Error] Model file not found: {model_path}"
            raise FileNotFoundError(msg)
        agent.load(model_path)

    return agent


def print_roster_matrix(env: NurseRosterEnv, algo: AlgoType, scenario_id: int, info: dict) -> None:
    """
    환경의 현재 근무표(`env.current_roster`)를 기반으로
    D/E/N/O 문자 매트릭스를 깔끔하게 콘솔에 출력합니다.

    Args:
        env (NurseRosterEnv): 최종 근무표가 반영된 환경.
        algo (str): 사용한 알고리즘 이름.
        scenario_id (int): 시나리오 ID.
        info (dict): 마지막 스텝에서 반환된 제약 위반 정보 딕셔너리.

    Returns:
        None
    """
    roster = env.current_roster  # shape: (N, D)
    n_nurses, n_days = roster.shape

    print("\n=== Generated Nurse Roster ===")
    print(f"- Algorithm   : {algo.upper()}")
    print(f"- Scenario ID : {scenario_id}")
    print(f"- Nurses      : {n_nurses}")
    print(f"- Days        : {n_days}")
    print(f"- Final Score : {env.current_score:.2f}")
    if info is not None:
        hard = info.get("hard_violations", "NA")
        shortage = info.get("coverage_shortage", "NA")
        print(f"- Hard Viol.  : {hard}")
        print(f"- Shortage    : {shortage}")

    # 헤더(일자)
    day_labels = [f"{d:02d}" for d in range(1, n_days + 1)]
    header = "Nurse\\Day | " + " ".join(day_labels)
    print("\n" + header)
    print("-" * len(header))

    # 간호사 이름이 있으면 사용, 없으면 인덱스 사용
    nurse_names = []
    for nurse in env.nurses:
        name = nurse.get("name")
        if not name:
            name = f"Nurse_{nurse.get('id', len(nurse_names))}"
        nurse_names.append(name)

    # 각 행 출력
    for i in range(n_nurses):
        row_shifts = [SHIFT_SYMBOLS.get(code, "?") for code in roster[i]]
        name = nurse_names[i] if i < len(nurse_names) else f"Nurse_{i}"
        label = f"{name:8s}"
        print(f"{label} | " + " ".join(row_shifts))

    print("\nLegend: O=Off, D=Day, E=Evening, N=Night\n")


def generate_roster(
    algo: AlgoType,
    scenario_id: int,
    model_path: str | None = None,
    data_path: str = "data/scenarios.pkl",
    max_steps: int = 200,
) -> None:
    """
    지정한 알고리즘과 시나리오를 사용하여 근무표를 생성하고,
    최종 근무표를 D/E/N/O 문자로 출력합니다.

    Args:
        algo (str): 사용할 알고리즘 이름 ('bandit', 'dqn', 'reinforce', 'ppo').
        scenario_id (int): 사용할 시나리오 ID (예: 5).
        model_path (str | None): 학습된 모델 파라미터 경로.
            - Bandit: 사용하지 않음.
            - DQN/REINFORCE/PPO: None이면 기본 경로(models/<algo>/<algo>_scenario{scenario_id}_seed42.pth)를 시도.
        data_path (str): 시나리오 피클 파일 경로. 기본은 'data/scenarios.pkl'.
        max_steps (int): 최대 스텝 수(예: 200).

    Returns:
        None
    """
    # 1. 환경 로드
    env = NurseRosterEnv(data_path=data_path, scenario_id=scenario_id)

    # 2. 기본 모델 경로 설정 (필요 시)
    resolved_model_path: str | None = model_path
    if algo in ("dqn", "reinforce", "ppo") and model_path is None:
        algo_dir = algo
        default_path = os.path.join(
            "models",
            algo_dir,
            f"{algo}_scenario{scenario_id}_seed42.pth",
        )
        resolved_model_path = default_path

    # 3. 에이전트 생성 및 파라미터 로드
    agent = build_agent(algo, env, resolved_model_path)

    # 4. 에피소드 실행 (정책에 따라 근무표 수정)
    obs = env.reset()
    done = False
    steps = 0
    last_info: dict | None = None

    while not done and steps < max_steps:
        if algo == "dqn":
            action = agent.select_action(obs, training=False)  # type: ignore[attr-defined]
        else:
            action = agent.select_action(obs)  # type: ignore[attr-defined]

        obs, reward, done, info = env.step(action)
        last_info = info
        steps += 1

    # 5. 최종 근무표 출력
    print_roster_matrix(env, algo, scenario_id, last_info or {})


def main() -> None:
    """
    CLI로부터 인자를 받아 근무표 생성 함수를 실행합니다.
    """
    parser = argparse.ArgumentParser(
        description="학습된 RL/Heuristic 정책으로 최종 간호사 근무표(D/E/N/O)를 생성합니다.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["bandit", "dqn", "reinforce", "ppo"],
        help="사용할 알고리즘 이름.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        required=True,
        help="시나리오 ID (CSV에서 생성된 data/scenarios.pkl 기준).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "학습된 모델 파라미터 경로(DQN/REINFORCE/PPO 전용). "
            "지정하지 않으면 models/<algo>/<algo>_scenario{scenario}_seed42.pth 를 시도합니다."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/scenarios.pkl",
        help="시나리오 피클 파일 경로.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="한 에피소드에서 수행할 최대 액션 수.",
    )

    args = parser.parse_args()

    generate_roster(
        algo=args.algo,  # type: ignore[arg-type]
        scenario_id=args.scenario,
        model_path=args.model,
        data_path=args.data,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()


