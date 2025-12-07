import numpy as np
import pickle
import os
from .score import RosterScore

class NurseRosterEnv:
    """
    간호사 근무표 생성 문제를 풀기 위한 강화학습 환경입니다.
    Gym Interface (reset, step)를 따릅니다.
    
    특정 시나리오 ID를 지정하여 로드할 수 있습니다.
    """
    
    def __init__(self, data_path="rl_experiments/data/scenarios.pkl", scenario_id=1):
        # 1. 데이터 로드 (없으면 생성 시도하지 않고 에러)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Scenario file not found at {data_path}. Run data_loader_csv.py first.")
            
        with open(data_path, 'rb') as f:
            all_scenarios = pickle.load(f)
            
        if scenario_id not in all_scenarios:
            raise ValueError(f"Scenario ID {scenario_id} not found in data.")
            
        self.data = all_scenarios[scenario_id]
        
        self.nurses = self.data['nurses']
        self.config = self.data['config']
        self.requests = self.data['requests']
        self.initial_roster = self.data['initial_roster']
        
        self.N = self.data['meta']['N']
        self.D = self.data['meta']['D']
        
        # 2. 점수 계산기 (심판)
        self.scorer = RosterScore(self.config)
        
        # 3. 내부 상태
        self.current_roster = None
        self.current_score = -float('inf')
        self.steps = 0
        self.max_steps = 200
        
    def reset(self):
        """환경 초기화"""
        self.steps = 0
        
        # 근무표 초기화 (완전 랜덤)
        self.current_roster = np.random.randint(0, 4, size=(self.N, self.D))
        
        # 초기 점수 계산
        self.current_score, info = self.scorer.calculate_score(
            self.current_roster, self.requests
        )
        
        return self._get_observation()

    def step(self, action):
        n_idx, d_idx, s_code = action
        
        prev_score = self.current_score
        
        # 근무표 수정
        self.current_roster[n_idx, d_idx] = s_code
        
        # 점수 계산
        new_score, info = self.scorer.calculate_score(
            self.current_roster, self.requests
        )
        
        reward = new_score - prev_score
        
        self.current_score = new_score
        self.steps += 1
        
        done = (self.steps >= self.max_steps)
            
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # 1. Nurse State (N, D, 5)
        nurse_state = np.zeros((self.N, self.D, 5))
        for s in range(4):
            nurse_state[:, :, s] = (self.current_roster == s).astype(float)
            
        for n_idx, reqs in self.requests.items():
            for day, shift in reqs.items():
                if shift == 'O':
                    nurse_state[n_idx, day, 4] = 1.0
                    
        # 2. Global State (D, 3)
        global_state = np.zeros((self.D, 3))
        
        counts = {
            1: np.sum(self.current_roster == 1, axis=0),
            2: np.sum(self.current_roster == 2, axis=0),
            3: np.sum(self.current_roster == 3, axis=0),
        }
        
        # 동적 min_staff 반영
        min_staff_dyn = self.config.get('min_staff_dynamic', {})
        
        for d in range(self.D):
            reqs = min_staff_dyn.get(d, {'D':0, 'E':0, 'N':0})
            global_state[d, 0] = reqs.get('D', 0) - counts[1][d]
            global_state[d, 1] = reqs.get('E', 0) - counts[2][d]
            global_state[d, 2] = reqs.get('N', 0) - counts[3][d]
        
        return {
            'nurse_state': nurse_state,
            'global_state': global_state
        }
