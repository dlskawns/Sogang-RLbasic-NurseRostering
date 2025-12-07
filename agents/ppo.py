import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, N, D, n_features=5, g_features=3):
        super(ActorCritic, self).__init__()
        self.N = N
        self.D = D
        
        # --- Shared Feature Extractor ---
        # Nurse Encoder
        self.nurse_conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) 
        )
        
        # Global Encoder
        self.global_conv = nn.Sequential(
            nn.Conv1d(g_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        combined_dim = 64 + 32
        
        # --- Actor Heads (Policy) ---
        self.actor_n = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, N))
        self.actor_d = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, D))
        self.actor_s = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 4))
        
        # --- Critic Head (Value) ---
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Scalar Value
        )

    def forward(self, n_state, g_state):
        batch_size = n_state.size(0)
        
        # Feature Extraction
        n_in = n_state.view(-1, self.D, 5).permute(0, 2, 1)
        n_feat = self.nurse_conv(n_in).view(batch_size, self.N, 64)
        n_summary = torch.max(n_feat, dim=1)[0]
        
        g_in = g_state.permute(0, 2, 1)
        g_feat = self.global_conv(g_in).view(batch_size, 32)
        
        combined = torch.cat([n_summary, g_feat], dim=1)
        
        # Outputs
        logits_n = self.actor_n(combined)
        logits_d = self.actor_d(combined)
        logits_s = self.actor_s(combined)
        
        value = self.critic(combined)
        
        return logits_n, logits_d, logits_s, value

class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs # 업데이트 시 반복 횟수
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(env.N, env.D).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
        # Memory
        self.memory = {
            'n_states': [], 'g_states': [],
            'actions': [], 'log_probs': [], 'rewards': [], 'dones': [],
            'values': [] # For GAE (optional, here using simple MC return)
        }

    def select_action(self, obs):
        n_state = torch.FloatTensor(obs['nurse_state']).unsqueeze(0).to(self.device)
        g_state = torch.FloatTensor(obs['global_state']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits_n, logits_d, logits_s, val = self.policy(n_state, g_state)
            
            dist_n = Categorical(logits=logits_n)
            dist_d = Categorical(logits=logits_d)
            dist_s = Categorical(logits=logits_s)
            
            action_n = dist_n.sample()
            action_d = dist_d.sample()
            action_s = dist_s.sample()
            
            log_prob = dist_n.log_prob(action_n) + dist_d.log_prob(action_d) + dist_s.log_prob(action_s)
            
        # Store memory
        self.memory['n_states'].append(obs['nurse_state'])
        self.memory['g_states'].append(obs['global_state'])
        self.memory['actions'].append((action_n.item(), action_d.item(), action_s.item()))
        self.memory['log_probs'].append(log_prob.item())
        self.memory['values'].append(val.item())
        
        return (action_n.item(), action_d.item(), action_s.item())

    def remember_reward(self, reward, done):
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def update(self):
        # 1. Prepare Data
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        
        # Monte Carlo Estimate of State Rewards
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        # Normalizing the returns
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Convert list to tensor
        old_n_states = torch.FloatTensor(np.array(self.memory['n_states'])).to(self.device)
        old_g_states = torch.FloatTensor(np.array(self.memory['g_states'])).to(self.device)
        old_actions = torch.LongTensor(np.array(self.memory['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        
        # Optimize policy for K epochs
        total_loss = 0
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logits_n, logits_d, logits_s, state_values = self.policy(old_n_states, old_g_states)
            state_values = state_values.squeeze()
            
            dist_n = Categorical(logits=logits_n)
            dist_d = Categorical(logits=logits_d)
            dist_s = Categorical(logits=logits_s)
            
            # Recalculate log probs
            new_log_probs = (
                dist_n.log_prob(old_actions[:, 0]) + 
                dist_d.log_prob(old_actions[:, 1]) + 
                dist_s.log_prob(old_actions[:, 2])
            )
            
            # Entropy (for exploration)
            dist_entropy = dist_n.entropy() + dist_d.entropy() + dist_s.entropy()
            
            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Advantage
            advantages = returns - state_values.detach()
            
            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = (
                -torch.min(surr1, surr2) 
                + 0.5 * self.mse_loss(state_values, returns) 
                - 0.01 * dist_entropy
            )
            
            # Backprop
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            total_loss += loss.mean().item()
            
        # Clear memory
        self.memory = {
            'n_states': [], 'g_states': [],
            'actions': [], 'log_probs': [], 'rewards': [], 'dones': [],
            'values': []
        }
        
        return total_loss / self.k_epochs

    def save(self, path: str) -> None:
        """
        현재 Actor-Critic 네트워크의 파라미터를 지정한 경로에 저장합니다.
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """
        저장된 파라미터를 로드하여 Actor-Critic 네트워크를 복원합니다.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)

