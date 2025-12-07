import numpy as np
import pickle
from .score import RosterScore

class NurseRosterEnv:
    """
    간호사 근무표 생성 문제를 풀기 위한 강화학습 환경입니다.
    Gym Interface (reset, step)를 따릅니다.
    
    State:
        - Nurse Features (N, D, F_n)
        - Global Features (D, F_g)
    
    Action:
        - (Nurse_Index, Day_Index, Shift_Type)
    """
    
    def __init__(self, data_path="rl_experiments/data/dataset_medium.pkl"):
        # 1. 데이터 로드
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.nurses = self.data['nurses']   # 간호사 데이터
        self.config = self.data['config']   # 제약 조건 데이터
        self.requests = self.data['requests'] # 요청 사항 데이터
        self.initial_roster = self.data['initial_roster'] # 초기 근무표 데이터
        
        self.N = len(self.nurses)   # 간호사 수
        self.D = self.initial_roster.shape[1] # 날짜 수 
        
        # 2. 점수 계산기 (심판)
        self.scorer = RosterScore(self.config)
        
        # 3. 내부 상태
        self.current_roster = None # 현재 근무표
        self.current_score = -float('inf') # 현재 점수
        self.steps = 0 # 현재 스텝
        self.max_steps = 200 # 에피소드 당 최대 수정 횟수 (최대 200번)
        
    def reset(self):
        """
        환경을 초기화합니다.
        
        Returns:
            observation (dict): 초기 상태 (tensors)
        """
        self.steps = 0 # 스텝 초기화
        
        # 근무표 초기화 (모두 0 또는 랜덤)
        # 실험을 위해 매번 똑같은 0(Off)으로 시작하거나, 랜덤하게 섞을 수 있음
        # 여기서는 완전 랜덤하게 시작해서 "고쳐나가는" 과정을 학습시킴
        self.current_roster = np.random.randint(0, 4, size=(self.N, self.D))
        
        # 초기 점수 계산
        self.current_score, _ = self.scorer.calculate_score(
            self.current_roster, self.requests
        )
        
        return self._get_observation()

    def step(self, action):
        """
        에이전트의 행동을 실행하고 결과를 반환합니다.
        
        Args:
            action (tuple): (nurse_idx, day_idx, shift_code)
                            예: (5, 10, 3) -> 5번 간호사 10일차를 Night(3)로 변경
        
        Returns:
            next_state (dict): 변경된 상태
            reward (float): 점수 변화량 (Action이 좋았으면 양수, 나빴으면 음수)
            done (bool): 에피소드 종료 여부
            info (dict): 추가 정보
        """
        n_idx, d_idx, s_code = action
        
        # 1. 변경 전 상태 저장 (Undo를 위함이 아니라, 변화량 계산용)
        prev_score = self.current_score
        
        # 2. 근무표 수정 (Action 적용)
        self.current_roster[n_idx, d_idx] = s_code
        
        # 3. 변경 후 점수 계산
        new_score, info = self.scorer.calculate_score(
            self.current_roster, self.requests
        )
        
        # 4. 보상 계산 (Reward = 점수 향상분)
        # 예: -100점 -> -90점이 되면 +10점 보상
        # 예: -90점 -> -100점이 되면 -10점 벌점
        reward = new_score - prev_score
        
        self.current_score = new_score
        self.steps += 1
        
        # 5. 종료 조건 (Max Steps 도달 혹은 완벽한 해?)
        done = (self.steps >= self.max_steps)
        if info['hard_violations'] == 0 and info['coverage_shortage'] == 0:
            # 완벽한 해를 찾았다면 큰 보너스 주고 종료할 수도 있음
            # reward += 100
            # done = True
            pass
            
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """
        현재 상황을 신경망이 이해할 수 있는 Tensor(Numpy Array)로 변환합니다.
        
        Returns:
            dict: {
                'nurse_state': (N, D, F_n),
                'global_state': (D, F_g)
            }
        """
        # 1. Nurse State (개인별 상태)
        # Feature 0~3: One-hot Shift (D, E, N, O)
        # Feature 4: 휴무 신청 여부 (0 or 1)
        nurse_state = np.zeros((self.N, self.D, 5))
        
        # One-hot Encoding
        # current_roster는 (N, D) 정수형
        for s in range(4):
            # 해당 근무인 곳만 1.0으로 표시
            nurse_state[:, :, s] = (self.current_roster == s).astype(float)
            
        # 휴무 신청 표시
        for n_idx, reqs in self.requests.items():
            for day, shift in reqs.items():
                if shift == 'O':
                    nurse_state[n_idx, day, 4] = 1.0
                    
        # 2. Global State (병동 전체 상태)
        # Feature 0: Day 부족분
        # Feature 1: Eve 부족분
        # Feature 2: Nig 부족분
        global_state = np.zeros((self.D, 3))
        
        # 날짜별 근무자 수 카운트
        counts = {
            1: np.sum(self.current_roster == 1, axis=0), # Day Counts (D,)
            2: np.sum(self.current_roster == 2, axis=0), # Eve Counts
            3: np.sum(self.current_roster == 3, axis=0), # Nig Counts
        }
        
        # 부족분 계산 (양수면 부족함) -> AI에게 "이만큼 부족해!"라고 알려줌
        global_state[:, 0] = self.config['min_staff']['D'] - counts[1]
        global_state[:, 1] = self.config['min_staff']['E'] - counts[2]
        global_state[:, 2] = self.config['min_staff']['N'] - counts[3]
        
        return {
            'nurse_state': nurse_state,
            'global_state': global_state
        }

