"""

    dqn.py - Deep Q Network implementation


    modified to use mps backend
    and use gymnasium instead of OpenAI gym(deprecated).
    
    Orignal code:
    https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/ch08/dqn.py
    and
    https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/pytorch/dqn.py
    
"""
from collections import deque
from torch import nn
import torch
import numpy as np
import gymnasium as gym
import random


class ReplayBuffer:
    """
    class for replay buffer 
    (https://ai.stackexchange.com/questions/42462/what-is-the-purpose-of-a-replay-memory-buffer-in-deep-q-learning-networks)
    
    """
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        """get a batch of samples from the buffer

        Returns:
            state, action, reward, next_state, done: tuple of random samples from the buffer
        """
        data = random.sample(self.buffer, self.batch_size)
        
        state = torch.tensor(np.array([x[0] for x in data]), dtype=torch.float32, device='mps') # array that contains the state of the environment,
        # in this case, a 4-dimensional array that indicates the position and velocity of the cart and the angle and angular velocity of the pole
        action = torch.tensor(np.array([x[1] for x in data]), dtype=torch.int32, device='mps') # 0 or 1 for CartPole
        reward = torch.tensor(np.array([x[2] for x in data]), dtype=torch.float32, device='mps')
        next_state = torch.tensor(np.array([x[3] for x in data]), dtype=torch.float32, device='mps')
        done = torch.tensor(np.array([x[4] for x in data]), dtype=torch.int32, device='mps')
        return state, action, reward, next_state, done
        
        
class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNet, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim, device='mps')
        self.l2 = nn.Linear(hidden_dim, hidden_dim, device='mps')
        self.l3 = nn.Linear(hidden_dim, output_dim, device='mps')
        
    def forward(self, x):
        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class DQNAgent:
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.action_size = 2

        
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.input_dim, self.hidden_dim, self.output_dim) # Q network which is used to estimate Q values
        self.qnet_target = QNet(self.input_dim, self.hidden_dim, self.output_dim) # target network: to stabilize training by fixing target values
        # the whole idea is to use the target network to estimate the target value in the Bellman equation
        # and use the Q network to estimate the current value in the Bellman equation
        # the target network is updated every few steps to match the Q network
        
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32, device='mps') # convert state to tensor
            # np.newaxis is used to add a new axis to the state array(https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis)
            q_values = self.qnet(state)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state) # the Q network is use to determine the current Q value of the state
        q = qs[np.arange(len(action)), action]
        
        next_qs = self.qnet_target(next_state) # the target network is used to estimate the target(the Q value of the next state) in the Bellman equation
        next_q = next_qs.max(1)[0]
        
        next_q.detach() # detach the next_q tensor from the computation graph
        target = reward + self.gamma * next_q * (1 - done) # Bellman equation, if done is True, the next state is terminal, and the reward is zero
        loss = nn.MSELoss()(q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())# update the target network to match the Q network, this is done not every step but every few steps
        
        
        
episodes = 300
sync_interval = 20 # the interval at which the target network is updated
env = gym.make('CartPole-v0')
agent = DQNAgent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state)
        done = terminated or truncated
        
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    if episode % sync_interval == 0:
        agent.sync_qnet()
        
    reward_history.append(total_reward)
    if episode % 10 == 0:
        print(f'Episode : {episode}, Reward : {total_reward}')
