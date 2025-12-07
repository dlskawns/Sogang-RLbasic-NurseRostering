import argparse
import os
import csv
import time
import numpy as np
import random
import torch
from datetime import datetime

from envs.roster_env import NurseRosterEnv
from agents.baseline_bandit import GreedyBanditAgent
from agents.dqn import DQNAgent
from agents.reinforce import ReinforceAgent
from agents.ppo import PPOAgent


def train(algo, episodes=1000, seed=42, scenario_id=1):
    """
    선택한 알고리즘으로 간호사 근무표 강화학습 실험을 수행합니다.
    
    Args:
        algo (str): 사용할 알고리즘 이름('bandit', 'dqn', 'reinforce', 'ppo').
        episodes (int): 학습 에피소드 수(예: 200 에피소드).
        seed (int): 난수 시드 값(예: 42).
        scenario_id (int): 사용할 시나리오 ID(예: 1).
    
    Returns:
        None: 결과는 로그 CSV와 최적 모델 파라미터 파일로 저장됩니다.
    """
    # 1. 환경 및 시드 설정
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 시나리오 ID 지정
    try:
        env = NurseRosterEnv(scenario_id=scenario_id) 
    except Exception as e:
        print(f"Error loading environment: {e}")
        return
    
    # 2. 에이전트 선택
    if algo == 'bandit':
        agent = GreedyBanditAgent(env, epsilon=0.3)
    elif algo == 'dqn':
        agent = DQNAgent(env, lr=1e-4, gamma=0.95)
    elif algo == 'reinforce':
        agent = ReinforceAgent(env, lr=5e-4, gamma=0.99)
    elif algo == 'ppo':
        agent = PPOAgent(env, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
        
    # 3. 로거 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{algo}_seed{seed}_{timestamp}.csv"
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'steps', 'total_reward', 'final_score', 'hard_violations', 'coverage_shortage', 'time', 'epsilon', 'loss'])

    print(f"=== Training Start: {algo.upper()} (Seed={seed}, Scenario={scenario_id}) ===")

    # 4. 모델 저장 설정 (최적 모델만 저장)
    model_dir = os.path.join("models", algo)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir, f"{algo}_scenario{scenario_id}_seed{seed}.pth"
    )
    best_hard = float("inf")
    best_score = -float("inf")
    
    # 5. 에피소드 루프
    start_time = time.time()
    
    for ep in range(1, episodes + 1):
        obs = env.reset()
        episode_reward = 0
        loss_sum = 0
        update_cnt = 0
        
        done = False
        while not done:
            # Action Selection
            if algo in ['dqn', 'reinforce', 'ppo']:
                action = agent.select_action(obs)
            else:
                action = agent.select_action(obs)
                
            next_obs, reward, done, info = env.step(action)
            
            # Agent Update / Memory
            if algo == 'dqn':
                agent.remember(obs, action, reward, next_obs, done)
                loss = agent.update()
                if loss > 0:
                    loss_sum += loss
                    update_cnt += 1
            elif algo == 'reinforce':
                agent.remember_reward(reward)
            elif algo == 'ppo':
                agent.remember_reward(reward, done)
            elif algo == 'bandit':
                agent.update(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            obs = next_obs
            
        # Episode End Updates
        if algo == 'dqn' and ep % 10 == 0:
            agent.update_target_network()
        elif algo in ['reinforce', 'ppo']:
            loss = agent.update()
            loss_sum = loss
            update_cnt = 1
            
        # Best 모델 갱신 및 저장 (Hard Violation 우선, 그 다음 Score)
        if algo in ['dqn', 'reinforce', 'ppo']:
            cur_hard = info['hard_violations']
            cur_score = env.current_score
            is_better = (cur_hard < best_hard) or (
                cur_hard == best_hard and cur_score > best_score
            )
            if is_better:
                best_hard = cur_hard
                best_score = cur_score
                # 에이전트에 save 메서드가 정의되어 있다고 가정
                try:
                    agent.save(model_path)
                    # 너무 자주 출력되면 시끄러우니 50 에피소드 단위로만 표시
                    if ep % 50 == 0:
                        print(f"  >> New best model saved: Hard={best_hard}, Score={best_score:.2f}")
                except AttributeError:
                    pass

        # 통계
        avg_loss = loss_sum / update_cnt if update_cnt > 0 else 0
        eps = agent.epsilon if hasattr(agent, 'epsilon') else 0
            
        # 로그 기록
        elapsed = time.time() - start_time
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, env.steps, 
                f"{episode_reward:.2f}", 
                f"{env.current_score:.2f}",
                info['hard_violations'], 
                info['coverage_shortage'],
                f"{elapsed:.1f}",
                f"{eps:.3f}",
                f"{avg_loss:.4f}"
            ])
            
        if ep % 10 == 0:
            print(f"Ep {ep:4d} | Score: {env.current_score:8.2f} | Hard: {info['hard_violations']:3d} | Reward: {episode_reward:8.2f} | Eps: {eps:.3f} | Loss: {avg_loss:.4f}")


    print(f"=== Training Finished. Log saved to {log_file} ===")
    if algo in ['dqn', 'reinforce', 'ppo']:
        print(f"Best model (Hard={best_hard}, Score={best_score:.2f}) saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='bandit', help='Algorithm to use (bandit, dqn, ppo)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--scenario', type=int, default=1, help='Scenario ID')
    args = parser.parse_args()
    
    train(args.algo, args.episodes, args.seed, args.scenario)

