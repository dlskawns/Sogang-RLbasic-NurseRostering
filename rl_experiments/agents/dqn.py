import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """
    상태를 입력받아 (Nurse, Day, Shift) 각각에 대한 Q-value를 출력하는 신경망
    """
    def __init__(self, N, D, n_features=5, g_features=3):
        super(QNetwork, self).__init__()
        self.N = N
        self.D = D
        
        # 1. Feature Encoders
        # Nurse State (N, D, 5) -> (N, Hidden)
        # Global State (D, 3) -> (Hidden)
        
        # Nurse 정보를 1D Conv로 처리 (Time축=Day)
        self.nurse_conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # (N, 64, 1) -> 각 간호사의 대표 특징 추출
        )
        
        # Global 정보를 1D Conv로 처리
        self.global_conv = nn.Sequential(
            nn.Conv1d(g_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # (1, 32, 1) -> 전체 상황 요약
        )
        
        # Combined Feature Size
        combined_dim = 64 + 32
        
        # 2. Heads (Action Decoders)
        # Nurse Head: 누가 문제인가? (Output: N)
        self.head_nurse = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, N)
        )
        
        # Day Head: 언제가 문제인가? (Output: D)
        self.head_day = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, D)
        )
        
        # Shift Head: 뭘로 바꿀까? (Output: 4)
        self.head_shift = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, n_state, g_state):
        """
        Args:
            n_state: (Batch, N, D, F_n)
            g_state: (Batch, D, F_g)
        Returns:
            q_n, q_d, q_s
        """
        batch_size = n_state.size(0)
        
        # Permute for Conv1d: (Batch*N, F_n, D)
        # Nurse Feature Extraction
        # n_state를 (Batch*N, F_n, D) 형태로 변환해야 Conv1d 통과 가능
        n_in = n_state.view(-1, self.D, 5).permute(0, 2, 1) # (B*N, 5, D)
        n_feat = self.nurse_conv(n_in).view(batch_size, self.N, 64) # (B, N, 64)
        
        # Nurse 전체의 평균/Max 특징을 뽑아서 Global Context로 씀
        n_summary = torch.max(n_feat, dim=1)[0] # (B, 64)
        
        # Global Feature Extraction
        g_in = g_state.permute(0, 2, 1) # (B, 3, D)
        g_feat = self.global_conv(g_in).view(batch_size, 32) # (B, 32)
        
        # Combine
        combined = torch.cat([n_summary, g_feat], dim=1) # (B, 96)
        
        # Heads
        q_n = self.head_nurse(combined) # (B, N)
        q_d = self.head_day(combined)   # (B, D)
        q_s = self.head_shift(combined) # (B, 4)
        
        return q_n, q_d, q_s

class DQNAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64):
        self.env = env
        self.N = env.N
        self.D = env.D
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Device Check
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_net = QNetwork(self.N, self.D).to(self.device)
        self.target_net = QNetwork(self.N, self.D).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Replay Buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Epsilon (Exploration)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, obs, training=True):
        """Epsilon-Greedy Action Selection"""
        if training and random.random() < self.epsilon:
            return (
                random.randint(0, self.N-1),
                random.randint(0, self.D-1),
                random.randint(0, 3)
            )
        
        # Tensor 변환
        n_state = torch.FloatTensor(obs['nurse_state']).unsqueeze(0).to(self.device)
        g_state = torch.FloatTensor(obs['global_state']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_n, q_d, q_s = self.q_net(n_state, g_state)
            
            # 각각 Argmax로 선택 (Independent Selection)
            # Joint Action Q(s, n, d, s)를 근사하기 위해 합칠 수도 있지만
            # 여기선 독립적으로 가장 높은 가치를 가진 요소들을 조합
            nurse_idx = torch.argmax(q_n).item()
            day_idx = torch.argmax(q_d).item()
            shift_idx = torch.argmax(q_s).item()
            
        return (nurse_idx, day_idx, shift_idx)

    def remember(self, state, action, reward, next_state, done):
        """Save experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def update(self):
        """Train the network"""
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        batch = random.sample(self.buffer, self.batch_size)
        
        # Prepare Batch Data
        n_states = torch.FloatTensor(np.array([x[0]['nurse_state'] for x in batch])).to(self.device)
        g_states = torch.FloatTensor(np.array([x[0]['global_state'] for x in batch])).to(self.device)
        
        # Action is tuple (n, d, s)
        actions = np.array([x[1] for x in batch])
        act_n = torch.LongTensor(actions[:, 0]).to(self.device)
        act_d = torch.LongTensor(actions[:, 1]).to(self.device)
        act_s = torch.LongTensor(actions[:, 2]).to(self.device)
        
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        
        next_n_states = torch.FloatTensor(np.array([x[3]['nurse_state'] for x in batch])).to(self.device)
        next_g_states = torch.FloatTensor(np.array([x[3]['global_state'] for x in batch])).to(self.device)
        
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        
        # 1. Current Q
        curr_q_n, curr_q_d, curr_q_s = self.q_net(n_states, g_states)
        
        # Gather Q-values for chosen actions
        # (Batch, N) -> gather -> (Batch)
        q_n = curr_q_n.gather(1, act_n.unsqueeze(1)).squeeze(1)
        q_d = curr_q_d.gather(1, act_d.unsqueeze(1)).squeeze(1)
        q_s = curr_q_s.gather(1, act_s.unsqueeze(1)).squeeze(1)
        
        # Total Q = Average or Sum?
        # 분산된 구조이므로 각각의 Loss를 구해서 합칩니다.
        
        # 2. Target Q
        with torch.no_grad():
            next_q_n, next_q_d, next_q_s = self.target_net(next_n_states, next_g_states)
            target_n = rewards + (1 - dones) * self.gamma * torch.max(next_q_n, dim=1)[0]
            target_d = rewards + (1 - dones) * self.gamma * torch.max(next_q_d, dim=1)[0]
            target_s = rewards + (1 - dones) * self.gamma * torch.max(next_q_s, dim=1)[0]
            
        # 3. Loss Calculation
        loss_n = self.loss_fn(q_n, target_n)
        loss_d = self.loss_fn(q_d, target_d)
        loss_s = self.loss_fn(q_s, target_s)
        
        total_loss = loss_n + loss_d + loss_s
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_loss.item()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

