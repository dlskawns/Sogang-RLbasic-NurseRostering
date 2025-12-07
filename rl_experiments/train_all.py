import os
import argparse
import subprocess
import time

def run_experiment(algo, episodes, seed, scenario):
    """단일 실험 실행 (subprocess)"""
    cmd = [
        "python", "-m", "rl_experiments.train",
        "--algo", algo,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--scenario", str(scenario)
    ]
    
    print(f"  >> Running {algo.upper()} (Seed {seed})...")
    start = time.time()
    
    # 실행 및 로그 실시간 출력 끄기 (너무 시끄러움)
    # 대신 완료 후 요약만 출력
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"     [Success] Completed in {elapsed:.1f}s")
    else:
        print(f"     [Failed] Error:\n{result.stderr}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=1, help='Scenario ID to train on')
    parser.add_argument('--episodes', type=int, default=1000, help='Episodes per experiment')
    args = parser.parse_args()
    
    # 실험 설정
    algos = ['bandit', 'dqn', 'reinforce', 'ppo']
    seeds = [42] # 시간 관계상 1개 시드만 먼저 (나중에 [42, 100, 2024]로 확장)
    
    print(f"=== Starting Batch Experiments (Scenario {args.scenario}) ===")
    print(f"Algorithms: {algos}")
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {seeds}")
    print("="*50)
    
    total_jobs = len(algos) * len(seeds)
    current_job = 0
    
    for algo in algos:
        for seed in seeds:
            current_job += 1
            print(f"\n[{current_job}/{total_jobs}] Experiment: {algo.upper()} (Seed {seed})")
            run_experiment(algo, args.episodes, seed, args.scenario)
            
    print("\n=== All Experiments Completed ===")
    print("Check rl_experiments/logs/ for results.")
    print("Run 'python rl_experiments/utils/plotter.py' to visualize.")

if __name__ == "__main__":
    main()

