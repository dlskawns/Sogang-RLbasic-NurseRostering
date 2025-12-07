import numpy as np
import os

from utils.data_gen import create_and_save_dataset
from envs.roster_env import NurseRosterEnv


def main():
    # 1. 데이터셋 생성 (없으면 생성)
    data_path = "data/dataset_medium.pkl"
    if not os.path.exists(data_path):
        create_and_save_dataset(filepath=data_path)
    
    # 2. 환경 초기화
    print("\n=== 환경 테스트 시작 ===")
    env = NurseRosterEnv(data_path)
    obs = env.reset()
    
    print(f"초기 점수: {env.current_score:.2f}")
    print(f"상태 크기 (Nurse): {obs['nurse_state'].shape} (N, D, Feat)")
    print(f"상태 크기 (Global): {obs['global_state'].shape} (D, Feat)")
    
    # 3. 랜덤 액션 수행 (5번)
    print("\n--- 랜덤 행동 수행 ---")
    for i in range(5):
        # 랜덤 액션: (간호사ID, 날짜ID, 근무코드)
        action = (
            np.random.randint(0, env.N),
            np.random.randint(0, env.D),
            np.random.randint(0, 4) # 0~3
        )
        
        next_obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}: Action {action} -> Reward {reward:+.2f} | Score {env.current_score:.2f}")
        print(f"       Violations: Hard={info['hard_violations']}, Shortage={info['coverage_shortage']}")

    print("\n테스트 완료! 환경이 정상적으로 작동합니다.")

if __name__ == "__main__":
    main()

