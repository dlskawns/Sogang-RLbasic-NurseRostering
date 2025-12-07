import numpy as np
import pickle
import os
import random

def create_and_save_dataset(
    filepath="rl_experiments/data/dataset_medium.pkl",
    n_nurses=30,
    n_days=30
):
    """
    강화학습 실험을 위한 고정 데이터셋을 생성하고 저장합니다.
    
    Args:
        filepath (str): 저장할 파일 경로
        n_nurses (int): 간호사 수 (기본 30명)
        n_days (int): 근무 일수 (기본 30일)
    """
    
    print(f"=== 데이터셋 생성 시작 (N={n_nurses}, D={n_days}) ===")
    
    # 1. 간호사 정보 생성
    # 예시: {'id': 0, 'experience': 0.7, 'is_night_only': 0, 'team_id': 1}
    nurses = []
    for i in range(n_nurses):
        nurse = {
            'id': i,
            'name': f"Nurse_{i}",
            'experience': round(random.uniform(0.1, 1.0), 2),  # 경력 (0~1 정규화)
            'is_night_only': 1 if random.random() < 0.1 else 0, # 10% 확률로 나이트 전담
            'team_id': random.randint(1, 3), # 1~3팀 중 하나
            'min_off': 8 # 월 최소 휴무일
        }
        nurses.append(nurse)

    # 2. 근무 제약 설정 (Constraints)
    config = {
        'max_consecutive_work': 5,   # 최대 5일 연속 근무
        'max_consecutive_night': 3,  # 최대 3일 연속 야간
        'min_staff': {               # 일자별 최소 필요 인원
            'D': 6, # Day 최소 6명
            'E': 5, # Evening 최소 5명
            'N': 3  # Night 최소 3명
        },
        'forbidden_patterns': ['ND', 'ED', 'NE'] # 금지 패턴 (Night->Day 등)
    }

    # 3. 선호도(Request) 생성
    # 예시: 5번 간호사가 10일차에 휴무(Off)를 원함
    requests = {}
    for i in range(n_nurses):
        # 각 간호사마다 2~3개의 휴무 신청을 랜덤하게 생성
        req_days = random.sample(range(n_days), k=random.randint(2, 3))
        requests[i] = {d: 'O' for d in req_days} # {날짜: 'O'} 형태

    # 4. 초기 근무표 (Initial Roster)
    # 모두 0(Off)으로 초기화하거나, 랜덤으로 채웁니다.
    # 0: Off, 1: Day, 2: Evening, 3: Night
    initial_roster = np.zeros((n_nurses, n_days), dtype=int)

    # 데이터 패키징
    dataset = {
        'nurses': nurses,
        'config': config,
        'requests': requests,
        'initial_roster': initial_roster,
        'meta': {
            'n_nurses': n_nurses,
            'n_days': n_days,
            'shift_map': {0: 'O', 1: 'D', 2: 'E', 3: 'N'}
        }
    }

    # 저장
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"=== 데이터셋 저장 완료: {filepath} ===")
    
    # 예시 데이터 출력 (사용자가 이해하기 쉽도록)
    print("\n[데이터 예시]")
    print(f" - 간호사 0번: {nurses[0]}")
    print(f" - 제약 조건: {config['min_staff']}")
    print(f" - 간호사 0번 요청: {requests[0]}")

if __name__ == "__main__":
    create_and_save_dataset()

