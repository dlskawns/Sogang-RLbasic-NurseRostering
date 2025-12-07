import argparse
import os

from envs.roster_env import NurseRosterEnv
from agents.dqn import DQNAgent


def eval_dqn_on_scenarios(
    model_path: str,
    scenario_ids: list[int],
    max_steps: int = 200,
) -> None:
    """
    저장된 DQN 정책을 불러와 여러 시나리오에서 추론만 수행하고
    최종 점수와 제약 위반 개수를 출력합니다.
    """
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        return

    print("=== DQN Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Scenarios: {scenario_ids}")
    print("-" * 40)

    for sid in scenario_ids:
        env = NurseRosterEnv(scenario_id=sid)
        agent = DQNAgent(env)
        agent.load(model_path)

        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(obs, training=False)
            obs, reward, done, info = env.step(action)
            steps += 1

        print(
            f"Scenario {sid:2d} | "
            f"Final Score: {env.current_score:8.2f} | "
            f"Hard Violations: {info['hard_violations']:3d}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/dqn/dqn_scenario1_seed42.pth",
        help="Path to saved DQN model",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=[3, 5, 7, 8, 10, 12, 13, 15],
        help="Scenario IDs to evaluate on (must have same number of days as training scenario)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode during evaluation",
    )
    args = parser.parse_args()

    eval_dqn_on_scenarios(args.model, args.scenarios, args.max_steps)


if __name__ == "__main__":
    main()


