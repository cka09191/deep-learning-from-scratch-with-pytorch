import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from grid_world import GridWorld
#mps
DEVICE = torch.device('mps')
DTYPE = torch.float32
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(DEVICE)
def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH)
    y, x = state
    idx = y*WIDTH + x
    vec = np.zeros(HEIGHT*WIDTH)
    vec[idx] = 1.0
    return torch.tensor(vec[np.newaxis, :], dtype=DTYPE, device=DEVICE)

class QNet(nn.Module):
    def __init__(self, action_size = 4):
        super().__init__()
        self.l1 = nn.Linear(12, 100, dtype=DTYPE, device=DEVICE)
        self.l2 = nn.Linear(100,action_size, dtype=DTYPE, device=DEVICE)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
    
class QLearningAgent:
    def __init__(self):
        self.gamma = torch.tensor(0.9, dtype=DTYPE, device=DEVICE)
        self.lr = torch.tensor(0.01, dtype=DTYPE, device=DEVICE)
        self.epsilon = torch.tensor(0.1, dtype=DTYPE, device=DEVICE)
        self.action_size = 4
        
        self.qnet = QNet()
        self.optimizer = optim.SGD(self.qnet.parameters(), lr = self.lr)
        
    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()
        
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = torch.tensor(np.zeros(1), dtype=DTYPE, device=DEVICE)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis = 1)[0]
            next_q.detach()
        
        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = nn.MSELoss()(q, target)
        
        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data
    
    
env = GridWorld()
agent = QLearningAgent()

loss_history = []

episodes = 100

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)
        reward = torch.tensor(reward, dtype=DTYPE, device=DEVICE)
        done = torch.tensor(done, dtype=DTYPE, device=DEVICE)
        
        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state
    
    average_loss = total_loss / cnt
    loss_history.append(average_loss)
    if episode % 100 == 0:
        print('episode:', episode, 'loss:', average_loss)
    
plt.xlabel('episode')
plt.ylabel('loss')
#set device to cpu
loss_history = loss_history.cpu().numpy()
plt.plot(range(len(loss_history)), loss_history)
plt.show()

Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)
