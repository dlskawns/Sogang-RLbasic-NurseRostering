import argparse

from rl_experiments.envs.roster_env import NurseRosterEnv
from rl_experiments.agents.baseline_bandit import GreedyBanditAgent


def eval_bandit_on_scenarios(
    scenario_ids: list[int],
    max_steps: int = 200,
    epsilon: float = 0.3,
) -> None:
    """
    Greedy Bandit 휴리스틱을 여러 시나리오에 대해 실행하여
    최종 점수와 제약 위반 개수를 출력합니다.

    학습 파라미터는 없고, epsilon(탐험률)과 휴리스틱 규칙이 사실상의 "하이퍼파라미터"입니다.
    """
    print("=== Bandit Evaluation ===")
    print(f"Epsilon: {epsilon}")
    print(f"Scenarios: {scenario_ids}")
    print("-" * 40)

    for sid in scenario_ids:
        env = NurseRosterEnv(scenario_id=sid)
        agent = GreedyBanditAgent(env, epsilon=epsilon)

        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(obs)
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
        "--scenarios",
        type=int,
        nargs="+",
        default=[1],
        help="Scenario IDs to evaluate on",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode during evaluation",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Epsilon value for epsilon-greedy Bandit",
    )
    args = parser.parse_args()

    eval_bandit_on_scenarios(args.scenarios, args.max_steps, args.epsilon)


if __name__ == "__main__":
    main()


