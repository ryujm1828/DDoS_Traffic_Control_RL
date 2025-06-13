import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import random

# 파라미터 설정
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
GAMMA = 0.99 # Discount factor
EPS_START = 0.3 # Epsilon
EPS_END = 0.0
EPS_DECAY = 10000

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, action_size)
        )

    def forward(self, x):
        return self.layers(x)    

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size,device):
        self.state_size = state_size
        self.action_size = action_size # 0.0 ~ 0.9, 총 10개
        self.device = device
        
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Adam 옵티마이저 사용
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                # Q-value가 가장 높은 행동 선택
                return self.policy_net(state.to(self.device)).max(1)[1].view(1, 1)
        else:
            # 랜덤 행동 선택
            return torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long,device=self.device)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0]).to(self.device)        
        action_batch = torch.cat(batch[1]).to(self.device)       
        reward_batch = torch.cat(batch[2]).to(self.device)       
        next_state_batch = torch.cat(batch[3]).to(self.device)

        # 현재 Q-value 계산
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 다음 상태의 최대 Q-value 계산 (타겟 네트워크 사용)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        # 기대 Q-value (TD Target) 계산
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 손실 함수 (MSE Loss)
        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())