import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# Map elements
EMPTY = 0
WALL = 1
LAVA = 2
TREASURE = 3
EXIT = 4
START = 5
ENEMY = 6

# Environment size parameters
WINDOW_SIZE = 5
NUM_ACTIONS = 4  # up, down, left, right

# Helper to extract 5x5 window around agent
def get_local_view(grid, x, y, goal=None, size=5):
    half = size // 2
    padded = np.pad(grid, pad_width=half, mode='constant', constant_values=WALL)
    x_p, y_p = x + half, y + half
    view = padded[y_p - half:y_p + half + 1, x_p - half:x_p + half + 1]
    
    if goal:
        gx, gy = goal
        gx_rel = gx - x + half
        gy_rel = gy - y + half
        goal_channel = np.zeros_like(view, dtype=np.float32)
        if 0 <= gx_rel < size and 0 <= gy_rel < size:
            goal_channel[gy_rel, gx_rel] = 1.0
        return view, goal_channel
    return view, np.zeros_like(view, dtype=np.float32)

# Simple CNN for local-view Q-learning
class LocalViewDQN(nn.Module):
    def __init__(self, input_channels=8, num_actions=4):
        super(LocalViewDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * WINDOW_SIZE * WINDOW_SIZE, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNSolver:
    def __init__(self, grid, start, goal, difficulty=1, episodes=1000):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.difficulty = difficulty
        self.episodes = episodes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LocalViewDQN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)

        self.gamma = 0.9
        self.batch_size = 32
        self.epsilon_start = 1.0
        self.epsilon_min = 0.00001
        self.epsilon = self.epsilon_start

        if self.episodes > 0:
            self.epsilon_decay_step = (self.epsilon_start - self.epsilon_min) / self.episodes
        else:
            self.epsilon_decay_step = 0


    def encode_local_view(self, view, goal_channel):
        one_hot = np.zeros((8, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
        for i in range(7):
            one_hot[i] = (view == i).astype(np.float32)
        one_hot[7] = goal_channel
        return torch.tensor(one_hot, dtype=torch.float32)

    def select_action(self, state_tensor):
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_vals = self.model(next_states).max(1)[0].detach()
        target_q_vals = rewards + (1 - dones) * self.gamma * next_q_vals

        loss = F.smooth_l1_loss(q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()


    def train(self):
        self.epsilon = 1.0
        total_reward_sum = 0

        self.max_steps = 300
        for ep in range(self.episodes):
            x, y = self.start
            done = False
            total_reward = 0
            steps = 0
            visited = set()
            visit_counts = {}
            grid_copy = self.grid.copy()

            while not done and steps < self.max_steps:
                view, goal_channel = get_local_view(grid_copy, x, y, goal=self.goal)
                state = self.encode_local_view(view, goal_channel)

                action = self.select_action(state)
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
                nx, ny = x + dx, y + dy

                if not (0 <= nx < grid_copy.shape[1] and 0 <= ny < grid_copy.shape[0]):
                    reward = -5
                    next_state = state
                    done = False
                else:
                    tile = grid_copy[ny, nx]
                    
                    prev_dist = abs(x - self.goal[0]) + abs(y - self.goal[1])

                    reward = 1
                    if (x, y) not in visited:
                        reward += 1.0
                        visited.add((x, y))

                    next_dist = abs(nx - self.goal[0]) + abs(ny - self.goal[1])
                    reward += 0.5 * (prev_dist - next_dist)
                    if tile == LAVA:
                        reward -= 100
                        next_state = state
                        done = ep > 250
                    if tile == WALL:
                        wall_visits = visit_counts.get((nx, ny), 0)
                        reward -= 100 if wall_visits == 0 else 20
                        next_state = state
                    if tile == TREASURE:
                        reward += 50
                        grid_copy[ny, nx] = EMPTY
                    if tile == ENEMY:
                        reward -= 20
                    if (nx, ny) == self.goal:
                        reward += 200
                        done = True

                    view_next, goal_channel_next = get_local_view(grid_copy, nx, ny, goal=self.goal)
                    next_state = self.encode_local_view(view_next, goal_channel_next)

                    x, y = nx, ny

                    visit_counts[(x, y)] = visit_counts.get((x, y), 0) + 1
                    if visit_counts[(x, y)] > 3:
                        reward -= 30

                    if visit_counts.get((x, y), 0) <= 1:
                        reward += 0.2


                self.step(state, action, reward, next_state, done)
                total_reward += reward
                steps += 1

            total_reward_sum += total_reward
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_step)

            if ep % 10 == 0:
                print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return total_reward_sum / self.episodes


    def solve(self):
        path = []
        x, y = self.start
        steps = 0
        max_steps = 300
        visit_counts = {}
        total_penalty = 0

        while (x, y) != self.goal and steps < max_steps:
            path.append((x, y))

            view, goal_channel = get_local_view(self.grid, x, y, goal=self.goal)
            state = self.encode_local_view(view, goal_channel).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_vals = self.model(state)

            # Primary action (greedy)
            action = torch.argmax(q_vals).item()
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            nx, ny = x + dx, y + dy

            # If wall, attempt to pick a better alternative
            if not (0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]) or self.grid[ny, nx] == WALL:
                # Try alternate directions with high Q-values
                sorted_qs = torch.argsort(q_vals, descending=True).squeeze().tolist()
                found_valid = False
                for alt_action in sorted_qs:
                    dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][alt_action]
                    ax, ay = x + dx, y + dy
                    if 0 <= ax < self.grid.shape[1] and 0 <= ay < self.grid.shape[0] and self.grid[ay][ax] != WALL:
                        nx, ny = ax, ay
                        found_valid = True
                        break
                if not found_valid:
                    # Fallback: random legal direction
                    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                    random.shuffle(directions)
                    for dx, dy in directions:
                        ax, ay = x + dx, y + dy
                        if 0 <= ax < self.grid.shape[1] and 0 <= ay < self.grid.shape[0] and self.grid[ay][ax] != WALL:
                            nx, ny = ax, ay
                            found_valid = True
                            break
                    if not found_valid:
                        return path, f"No valid move from {(x, y)}, trapped by walls."

            # LAVA: end episode
            if self.grid[ny][nx] == LAVA:
                path.append((nx, ny))
                return path, f"Fell into lava at {(nx, ny)} — game over."

            x, y = nx, ny
            steps += 1
            total_penalty -= 0.1

            if self.grid[y][x] == TREASURE:
                self.grid[y][x] = EMPTY

            visit_counts[(x, y)] = visit_counts.get((x, y), 0) + 1
            if visit_counts[(x, y)] > 3:
                return path, f"Loop at {(x, y)} — breaking"

        path.append((x, y))
        return path, f"Reached {x, y} in {steps} steps, penalty {total_penalty:.2f}"



def solve_with_dqn(grid, start, goal, difficulty, model_path="dqn_solver_general.pth"):
    solver = DQNSolver(grid, start, goal, difficulty, episodes=0)

    # Load pre-trained weights
    solver.model.load_state_dict(torch.load(model_path, map_location=solver.device))
    solver.model.eval()  # set to inference mode

    return solver.solve()

