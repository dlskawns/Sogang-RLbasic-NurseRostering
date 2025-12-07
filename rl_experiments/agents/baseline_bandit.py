import numpy as np
import random

class GreedyBanditAgent:
    """
    규칙 기반(Rule-based)의 간단한 Bandit 에이전트입니다.
    가장 문제가 심각한 간호사나 날짜를 찾아 우선적으로 수정하는 'Heuristic'을 사용합니다.
    
    알고리즘: Epsilon-Greedy
    1. Epsilon 확률로 랜덤한 위치를 수정 시도 (Explore)
    2. (1-Epsilon) 확률로 'Hard Violation'이나 'Shortage'가 있는 곳을 타겟팅 (Exploit)
    3. 수정 후 점수가 오르면 유지, 내리면 원상복구 (Hill Climbing)
    """
    
    def __init__(self, env, epsilon=0.1):
        """
        Args:
            env: NurseRosterEnv 인스턴스
            epsilon: 랜덤 탐색 확률 (0.0 ~ 1.0)
        """
        self.env = env
        self.epsilon = epsilon
        self.N = env.N
        self.D = env.D
        
        # 이전 점수 기억용
        self.prev_score = -float('inf')

    def select_action(self, observation):
        """
        현재 상태를 보고 수정할 행동(Action)을 결정합니다.
        
        Bandit은 상태(State Tensor)를 깊게 분석하지 않고,
        환경에서 제공하는 '위반 정보'나 '단순 통계'를 활용합니다.
        
        Returns:
            action (tuple): (nurse_idx, day_idx, shift_code)
        """
        # 1. 탐험 (Explore): 무작위 행동
        if random.random() < self.epsilon:
            return self._get_random_action()
            
        # 2. 활용 (Exploit): 가장 급한 불 끄기
        # 환경에서 현재 어떤 문제가 있는지 힌트를 얻으면 좋겠지만, 
        # State Tensor만으로는 알기 어렵습니다.
        # 따라서 여기서는 Global State(인력 부족분)를 보고 판단합니다.
        
        global_state = observation['global_state'] # (D, 3) -> Day, Eve, Nig 부족분
        
        # 2-1. 인력 부족이 가장 심한 날짜 찾기
        # 부족분(양수)이 가장 큰 날짜와 근무 타입을 찾습니다.
        # argmax를 쓰기 위해 1차원으로 폅니다.
        flat_idx = np.argmax(global_state) # 0 ~ (D*3 - 1)
        
        max_shortage = global_state.flatten()[flat_idx]
        
        if max_shortage > 0:
            # 부족한 곳이 있다면, 그 날짜에 '노는 사람(Off)'을 투입합니다.
            day_idx = flat_idx // 3
            shift_type = (flat_idx % 3) + 1 # 0->1(Day), 1->2(Eve), 2->3(Nig)
            
            # 그 날짜에 쉬고 있는(0) 간호사 중 랜덤 선택
            # (더 똑똑하게 하려면 '연속 근무'가 적은 사람을 골라야 하지만, 여기선 랜덤)
            current_roster = self.env.current_roster
            candidates = np.where(current_roster[:, day_idx] == 0)[0]
            
            if len(candidates) > 0:
                nurse_idx = np.random.choice(candidates)
                return (nurse_idx, day_idx, shift_type)
        
        # 2-2. 인력 부족이 없다면, 랜덤하게 수정 (Soft Constraint 최적화 단계)
        return self._get_random_action()

    def _get_random_action(self):
        """완전 무작위 행동 반환"""
        return (
            np.random.randint(0, self.N),
            np.random.randint(0, self.D),
            np.random.randint(0, 4)
        )

    def update(self, state, action, reward, next_state, done):
        """
        Bandit은 학습(가중치 업데이트)을 하지 않습니다.
        다만, Hill Climbing(점수 오르면 유지, 내리면 복구)을 위해
        환경에 'Undo'를 요청하거나, 다음 행동에 반영해야 합니다.
        
        하지만 현재 env.step()은 무조건 반영하므로,
        점수가 떨어졌을 때 에이전트가 기억했다가 '되돌리는 액션'을 취해야 합니다.
        """
        # Hill Climbing 로직:
        # 보상이 음수(점수 하락)라면, 방금 한 행동을 취소하는 행동을 
        # 다음 스텝에 하도록 예약할 수 있습니다. 
        # (단, 여기서는 단순함을 위해 생략하고 '매번 새로운 시도'를 하는 구조로 둡니다.)
        pass

