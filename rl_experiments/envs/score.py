import numpy as np

class RosterScore:
    """
    근무표의 품질을 평가하고 제약 위반 여부를 계산하는 클래스입니다.
    CP-SAT Solver 없이 순수 Python/Numpy 연산으로 동작합니다.
    """
    
    def __init__(self, config):
        self.cfg = config
        # 근무 타입 매핑 (0:Off, 1:Day, 2:Eve, 3:Night)
        self.OFF, self.DAY, self.EVE, self.NIG = 0, 1, 2, 3
        
        # 벌점 가중치 (Penalty Weights)
        self.weights = {
            'hard': 10.0,    # 법적 제약 위반 (매우 큼)
            'soft': 1.0,     # 선호도 위반
            'coverage': 5.0  # 인원 부족
        }

    def calculate_score(self, roster, requests):
        """
        전체 근무표의 점수를 계산합니다.
        
        Args:
            roster (np.array): (N_nurses, N_days) 크기의 정수 배열
            requests (dict): {nurse_id: {day: shift}} 형태의 요청 사항
            
        Returns:
            total_reward (float): 총 점수 (음수일수록 나쁨)
            info (dict): 위반 사항 상세 카운트
        """
        N, D = roster.shape
        score = 0.0
        info = {
            'hard_violations': 0,
            'coverage_shortage': 0,
            'soft_violations': 0
        }

        # 1. Coverage Check (인원수 부족 확인) - Global Constraint
        # 각 날짜별로 D, E, N 근무자 수를 셉니다.
        # axis=0은 '간호사들'을 따라가며 세는 것 = 즉 날짜별 합계
        day_counts = np.sum(roster == self.DAY, axis=0) # (D,)
        eve_counts = np.sum(roster == self.EVE, axis=0)
        nig_counts = np.sum(roster == self.NIG, axis=0)
        
        # 부족분 계산 (음수면 부족) -> 절댓값 씌워서 벌점
        # 예: 필요 6명 - 실제 4명 = 2명 부족 -> -2 * 5.0 = -10점
        short_d = np.maximum(0, self.cfg['min_staff']['D'] - day_counts)
        short_e = np.maximum(0, self.cfg['min_staff']['E'] - eve_counts)
        short_n = np.maximum(0, self.cfg['min_staff']['N'] - nig_counts)
        
        total_shortage = np.sum(short_d + short_e + short_n)
        score -= (total_shortage * self.weights['coverage'])
        info['coverage_shortage'] = total_shortage

        # 2. Hard Constraints (법적 제약) - Nurse Constraint
        # (N, D) 행렬 전체를 한 번에 검사하지 않고, 간호사별로 순회하거나 벡터 연산
        
        # 2-1. 연속 근무 제한 (Max Consecutive Work)
        # 근무일(1,2,3)을 1로, 휴무(0)를 0으로 바꾼 마스크 생성
        is_work = (roster > 0).astype(int)
        
        # 간단한 로직: 6일 연속 1이 나오면 안됨.
        # Numpy로 "Sliding Window" 혹은 "Convolution"을 쓰면 빠름
        # 여기서는 이해를 위해 간단한 로직으로 구현 (추후 최적화 가능)
        
        # 각 간호사 별로 위반 체크
        for n in range(N):
            # 연속 근무 체크
            work_days = is_work[n]
            # 1이 6번 연속 나오는지 체크 (max_consecutive_work=5)
            # 스트링으로 변환해서 찾는 방식이 가장 직관적임 (속도는 느릴 수 있음)
            # 예: "111111" 패턴이 있으면 위반
            work_str = "".join(map(str, work_days))
            bad_pattern = "1" * (self.cfg['max_consecutive_work'] + 1)
            if bad_pattern in work_str:
                # 위반 횟수만큼 감점 (단순 포함 여부보다 몇 번 위반했는지)
                count = work_str.count(bad_pattern)
                score -= (count * self.weights['hard'])
                info['hard_violations'] += count

            # 2-2. 금지 패턴 (Night -> Day 등)
            # 패턴: Night(3) -> Day(1)
            shifts = roster[n]
            # (오늘Night AND 내일Day) 인 날짜 개수
            # shifts[:-1] == 3 (오늘 Night)
            # shifts[1:] == 1 (내일 Day)
            nd_violation = np.sum((shifts[:-1] == self.NIG) & (shifts[1:] == self.DAY))
            score -= (nd_violation * self.weights['hard'])
            info['hard_violations'] += nd_violation

        # 3. Soft Constraints (개인 선호도)
        for n_idx, reqs in requests.items():
            for day, wanted_shift in reqs.items():
                wanted_code = self._char_to_code(wanted_shift)
                actual_code = roster[n_idx, day]
                
                if actual_code != wanted_code:
                    score -= self.weights['soft']
                    info['soft_violations'] += 1

        return score, info

    def _char_to_code(self, char):
        mapping = {'O': 0, 'D': 1, 'E': 2, 'N': 3}
        return mapping.get(char, 0)

