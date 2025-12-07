import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    상태를 입력받아 각 행동(Nurse, Day, Shift)의 확률 분포(Logits)를 출력하는 정책 신경망
    """
    def __init__(self, N, D, n_features=5, g_features=3):
        super(PolicyNetwork, self).__init__()
        self.N = N
        self.D = D
        
        # 1. Feature Encoders (DQN과 동일 구조)
        self.nurse_conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) 
        )
        
        self.global_conv = nn.Sequential(
            nn.Conv1d(g_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        combined_dim = 64 + 32
        
        # 2. Heads (Policy Decoders) - Softmax는 밖에서 Categorical로 처리
        self.head_nurse = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, N)
        )
        
        self.head_day = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, D)
        )
        
        self.head_shift = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, n_state, g_state):
        batch_size = n_state.size(0)
        
        # Features
        n_in = n_state.view(-1, self.D, 5).permute(0, 2, 1)
        n_feat = self.nurse_conv(n_in).view(batch_size, self.N, 64)
        n_summary = torch.max(n_feat, dim=1)[0]
        
        g_in = g_state.permute(0, 2, 1)
        g_feat = self.global_conv(g_in).view(batch_size, 32)
        
        combined = torch.cat([n_summary, g_feat], dim=1)
        
        # Logits 출력 (확률로 변환되기 전 값)
        logits_n = self.head_nurse(combined)
        logits_d = self.head_day(combined)
        logits_s = self.head_shift(combined)
        
        return logits_n, logits_d, logits_s

class ReinforceAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99):
        self.env = env
        self.N = env.N
        self.D = env.D
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(self.N, self.D).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 에피소드 저장소
        # log_probs: 선택한 행동의 로그 확률
        # rewards: 받은 보상
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        """확률 분포에 따라 행동 선택 (Stochastic Policy)"""
        n_state = torch.FloatTensor(obs['nurse_state']).unsqueeze(0).to(self.device)
        g_state = torch.FloatTensor(obs['global_state']).unsqueeze(0).to(self.device)
        
        # 1. 확률 분포 생성
        logits_n, logits_d, logits_s = self.policy(n_state, g_state)
        
        dist_n = Categorical(logits=logits_n)
        dist_d = Categorical(logits=logits_d)
        dist_s = Categorical(logits=logits_s)
        
        # 2. 샘플링 (Sampling)
        action_n = dist_n.sample()
        action_d = dist_d.sample()
        action_s = dist_s.sample()
        
        # 3. 로그 확률 저장 (나중에 학습할 때 미분하기 위해 필요)
        # Joint Probability: P(n, d, s) = P(n) * P(d) * P(s) (독립 가정)
        # Log P = log P(n) + log P(d) + log P(s)
        log_prob = dist_n.log_prob(action_n) + dist_d.log_prob(action_d) + dist_s.log_prob(action_s)
        self.log_probs.append(log_prob)
        
        return (action_n.item(), action_d.item(), action_s.item())

    def remember_reward(self, reward):
        """매 스텝 받은 보상 저장"""
        self.rewards.append(reward)

    def update(self):
        """에피소드가 끝난 후 정책 업데이트 (Monte-Carlo)"""
        if not self.rewards:
            return 0.0
            
        R = 0
        returns = []
        
        # 1. Return(G_t) 계산: 뒤에서부터 할인율(Gamma) 적용하며 누적
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        
        # 2. Baseline 적용 (Returns 정규화) - 학습 안정성을 위해 필수
        # 평균 0, 표준편차 1로 맞춰주면 분산이 줄어듦
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        # 3. Loss 계산: -log_prob * return
        # "보상이 크면 그 행동의 확률을 높여라(log_prob 증가)"
        # "보상이 작으면(음수면) 그 행동의 확률을 낮춰라"
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # Sum -> Backprop
        total_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 초기화 (다음 에피소드를 위해)
        self.log_probs = []
        self.rewards = []
        
        return total_loss.item()

    def save(self, path: str) -> None:
        """
        현재 정책 네트워크의 파라미터를 지정한 경로에 저장합니다.
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """
        저장된 파라미터를 로드하여 정책 네트워크를 복원합니다.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)

