import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple

# Define map elements
EMPTY = 0
WALL = 1
LAVA = 2
TREASURE = 3
EXIT = 4
START = 5
ENEMY = 6
PLAYER = 7

# Define Colors for Visualization
COLOR_MAP = {
    EMPTY: "white",
    WALL: "brown",
    LAVA: "red",
    TREASURE: "yellow",
    EXIT: "green",
    START: "blue",
    ENEMY: "purple",
    PLAYER: "orange",
}

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
    
        grid_batch = torch.cat([e.state[0] for e in experiences])           # [B, 8, 20, 20]
        diff_batch = torch.cat([e.state[1] for e in experiences])           # [B, 1]
        actions = torch.tensor([e.action for e in experiences])
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float)
    
        next_grid_batch = torch.cat([e.next_state[0] for e in experiences])  # [B, 8, 20, 20]
        next_diff_batch = torch.cat([e.next_state[1] for e in experiences])  # [B, 1]
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float)
    
        return grid_batch, diff_batch, actions, rewards, next_grid_batch, next_diff_batch, dones


    def __len__(self):
        return len(self.buffer)
    
###Action not there in input
class CNN_DQNModel(nn.Module):
    def __init__(self, output_dim):
        super(CNN_DQNModel, self).__init__()

        # CNN layers to extract spatial features
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples to 10x10

        # Fully connected layers after flattening
        self.fc1 = nn.Linear(64 * 10 * 10 + 1, 256)  # +1 for difficulty input
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, grid_input, difficulty_input):
        x = F.relu(self.conv1(grid_input))         # [B, 32, 20, 20]
        x = F.relu(self.conv2(x))                  # [B, 64, 20, 20]
        x = self.pool(x)                           # [B, 64, 10, 10]
        x = x.view(x.size(0), -1)                  # Flatten [B, 6400]

        x = torch.cat((x, difficulty_input), dim=1)  # Concatenate difficulty: [B, 6401]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                         # [B, 9]
    
class DungeonAgent:
    def __init__(self, action_size, buffer_size=10000, batch_size=64, gamma=0.99,
                 lr=1e-4, tau=1e-3, epsilon_start=1.0, epsilon_end=0.0000001, epsilon_decay=0.999999):
        self.action_size = action_size
        self.memory = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        # self.epsilon_decay = epsilon_decay
        self.training_losses = []

        # Q-Network (CNN)
        self.qnetwork = CNN_DQNModel(output_dim=action_size)
        self.target_network = CNN_DQNModel(output_dim=action_size)
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork.to(self.device)
        self.target_network.to(self.device)
        self.learn_step_counter = 0
        self.update_every = 5 

    def act(self, grid_tensor, difficulty_tensor, eps_override=None):
        """Choose action based on epsilon-greedy strategy."""
        eps = self.epsilon if eps_override is None else eps_override
        if random.random() < eps:
            return random.choice(range(self.action_size))

        self.qnetwork.eval()
        with torch.no_grad():
            grid_tensor = grid_tensor.to(self.device)
            difficulty_tensor = difficulty_tensor.to(self.device)
            q_values = self.qnetwork(grid_tensor, difficulty_tensor)
        self.qnetwork.train()

        return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        grid_batch, diff_batch, actions, rewards, next_grid_batch, next_diff_batch, dones = \
            self.memory.sample(self.batch_size)

        grid_batch = grid_batch.to(self.device)
        diff_batch = diff_batch.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_grid_batch = next_grid_batch.to(self.device)
        next_diff_batch = next_diff_batch.to(self.device)
        dones = dones.to(self.device)
        
        # No need to re-process anything — just move on to:
        q_values = self.qnetwork(grid_batch.to(self.device), diff_batch.to(self.device))
        q_expected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)   # [B]

        # Get next Q values from target network
        with torch.no_grad():
            q_next = self.target_network(next_grid_batch, next_diff_batch) # [B, 9]
            q_target = rewards + (1 - dones) * self.gamma * q_next.max(1)[0]

        # Loss and backpropagation
        loss = F.smooth_l1_loss(q_expected, q_target)
        self.training_losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_every == 0:
            self.soft_update()

        # Epsilon decay
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.qnetwork.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)