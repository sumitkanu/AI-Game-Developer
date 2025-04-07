import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import math

# Define map element encoding
EMPTY = 0
WALL = 1
LAVA = 2
TREASURE = 3
EXIT = 4
START = 5

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Memory buffer for experience replay"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        states = torch.stack([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences])
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNModel(nn.Module):
    """Q-Network for predicting action values"""
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Increased from 256
        self.fc2 = nn.Linear(512, 1024)  # Increased from 512
        self.fc3 = nn.Linear(1024, 512)  # Increased from 256
        self.fc4 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DungeonAgent:
    """DQN Agent for dungeon generation"""
    def __init__(self, state_size, action_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        
        # Set random seeds for reproducibility
        self.seed = random.seed(seed)
        torch.manual_seed(seed)
        
        # Q-Networks
        self.qnetwork_local = DQNModel(state_size, action_size)
        self.qnetwork_target = DQNModel(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        
        # Replay memory
        self.memory = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99       # Discount factor
        self.tau = 1e-3         # Soft update parameter
        self.update_every = 4   # Update frequency
        self.step_count = 0
        self.epsilon = 1.0      # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def act(self, state, eval_mode=False):
        """Returns action based on current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if eval_mode or random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and learn if needed"""
        # Convert to tensor format
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        # Save experience in replay memory
        self.memory.push(state_tensor, action, reward, next_state_tensor, done)
        
        # Learn every update_every time steps
        self.step_count = (self.step_count + 1) % self.update_every
        if self.step_count == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class DungeonEnvironment:
    """Environment for dungeon generation with RL"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = None
        self.start_pos = (1, 1)
        self.exit_pos = (width-2, height-2)
        self.max_steps = width * height // 2  # Max steps before termination
        self.current_step = 0
        self.element_types = [EMPTY, WALL, LAVA, TREASURE]  # Possible elements to place
        self.min_rooms = 3  # Minimum number of rooms
        self.max_rooms = 8  # Maximum number of rooms
        self.room_min_size = 3  # Minimum room size
        self.room_max_size = 8  # Maximum room size
        self.corridor_width = 1  # Width of corridors
        
        self.rooms = []


        # Initialize the grid
        self.reset()
    
    def reset(self):
        """Reset the environment to starting state with more complex structure"""
        # Create empty grid
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Set boundary walls
        self.grid[0, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[self.height-1, :] = WALL
        self.grid[:, self.width-1] = WALL
        
        # Generate basic room structure to guide the RL agent
        self._generate_initial_rooms()
        
        # Place start and exit
        self.grid[self.start_pos[1], self.start_pos[0]] = START
        self.grid[self.exit_pos[1], self.exit_pos[0]] = EXIT
        
        self.current_step = 0
        
        return self._get_state()
    
    def _generate_initial_rooms(self):
        """Generate initial room structure to guide the RL agent"""
        self.rooms = []
        num_rooms = random.randint(self.min_rooms, self.max_rooms)
        
        # Try to place rooms
        for _ in range(num_rooms * 3):  # More attempts than needed to ensure we get enough rooms
            if len(self.rooms) >= num_rooms:
                break
                
            # Random room size and position
            w = random.randint(self.room_min_size, self.room_max_size)
            h = random.randint(self.room_min_size, self.room_max_size)
            x = random.randint(1, self.width - w - 1)
            y = random.randint(1, self.height - h - 1)
            
            # Check if this room overlaps with existing ones
            overlaps = False
            for room in self.rooms:
                rx, ry, rw, rh = room
                if (x < rx + rw + 1 and x + w + 1 > rx and
                    y < ry + rh + 1 and y + h + 1 > ry):
                    overlaps = True
                    break
            
            if not overlaps:
                # Add room and carve it into the grid
                self.rooms.append((x, y, w, h))
                
                # Mark room for agent guidance (not actual walls yet)
                for ry in range(y, y + h):
                    for rx in range(x, x + w):
                        self.grid[ry, rx] = EMPTY
        
        # Set start and exit positions in different rooms
        if self.rooms:
            # Start in the first room
            first_room = self.rooms[0]
            self.start_pos = (first_room[0] + first_room[2] // 2, 
                              first_room[1] + first_room[3] // 2)
            
            # Exit in the last room
            last_room = self.rooms[-1]
            self.exit_pos = (last_room[0] + last_room[2] // 2,
                            last_room[1] + last_room[3] // 2)
        else:
            # Fallback if no rooms were placed
            self.start_pos = (1, 1)
            self.exit_pos = (self.width - 2, self.height - 2)


    def _get_state(self):
        """Convert grid to state representation"""
        # Convert to one-hot encoding for each position
        state = []
        for y in range(self.height):
            for x in range(self.width):
                one_hot = [0] * len(self.element_types)
                cell_value = self.grid[y, x]
                
                # Handle special cases (START, EXIT)
                if cell_value == START or cell_value == EXIT:
                    state.extend(one_hot)  # Just zeros for start/exit
                else:
                    # Normal cell (EMPTY, WALL, LAVA, TREASURE)
                    if cell_value < len(one_hot):
                        one_hot[cell_value] = 1
                    state.extend(one_hot)
                    
        return np.array(state, dtype=np.float32)
    
    def _get_action_mask(self):
        """Get mask of valid actions (which cells can be modified)"""
        valid_actions = []
        for y in range(self.height):
            for x in range(self.width):
                # Can't modify boundary, start or exit positions
                if (y == 0 or x == 0 or y == self.height-1 or x == self.width-1 or
                    (x, y) == self.start_pos or (x, y) == self.exit_pos):
                    for _ in self.element_types:
                        valid_actions.append(0)  # Invalid action
                else:
                    for _ in self.element_types:
                        valid_actions.append(1)  # Valid action
        
        return np.array(valid_actions, dtype=np.float32)
    
    def step(self, action):
        """Process an action and return new state, reward, done flag"""
        self.current_step += 1
        
        # Decode action
        action_idx = action % (self.width * self.height * len(self.element_types))
        element_type_idx = action_idx % len(self.element_types)
        position_idx = action_idx // len(self.element_types)
        y = position_idx // self.width
        x = position_idx % self.width
        
        element_type = self.element_types[element_type_idx]
        
        reward = 0
        done = False
        
        # Check if valid action
        if (x == 0 or y == 0 or x == self.width-1 or y == self.height-1 or
            (x, y) == self.start_pos or (x, y) == self.exit_pos):
            reward -= 5  # Penalty for trying to modify unchangeable cells
        else:
            # Apply the action
            self.grid[y, x] = element_type
            
            # Calculate immediate rewards
            if element_type == LAVA:
                # Don't place lava too close to start or exit
                dist_to_start = abs(x - self.start_pos[0]) + abs(y - self.start_pos[1])
                dist_to_exit = abs(x - self.exit_pos[0]) + abs(y - self.exit_pos[1])
                if dist_to_start < 3 or dist_to_exit < 3:
                    reward -= 2
                else:
                    # Reward lava placement in strategic locations
                    reward += self._calculate_strategic_lava_reward(x, y)
            
            elif element_type == TREASURE:
                # Reward treasures that are more difficult to reach
                reward += self._calculate_treasure_reward(x, y)
                
            elif element_type == WALL:
                # Reward walls that create interesting paths but not dead ends
                reward += self._calculate_wall_reward(x, y)
        
        # Check if episode is done
        if self.current_step >= self.max_steps:
            done = True
            
            # Enhanced final evaluation of dungeon quality
            if self._is_accessible():
                reward += 10  # Base reward for accessible map
                
                # Calculate complexity metrics
                path_complexity = self._calculate_path_complexity()
                exploration_value = self._calculate_exploration_value()
                strategic_value = self._calculate_strategic_value()
                
                # Reward for complex maps that encourage exploration
                reward += path_complexity * 5
                reward += exploration_value * 10
                reward += strategic_value * 15
                
                # Penalize maps that are too simple
                if path_complexity < 0.3:
                    reward -= 10
            else:
                reward -= 50  # Severe penalty for inaccessible maps
        
        return self._get_state(), reward, done
    
    def _is_accessible(self):
        """Check if all non-wall areas are accessible from start"""
        # Perform BFS from start position
        visited = np.zeros((self.height, self.width), dtype=bool)
        queue = [self.start_pos]
        visited[self.start_pos[1], self.start_pos[0]] = True
        
        while queue:
            x, y = queue.pop(0)
            
            # Check all four directions
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    not visited[ny, nx] and self.grid[ny, nx] != WALL):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        
        # Check if all non-wall cells are visited
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] != WALL and not visited[y, x]:
                    return False
        
        return True
    
    def _calculate_path_complexity(self):
        """Calculate complexity of the path from start to exit"""
        # Perform BFS to find all paths
        visited = np.zeros((self.height, self.width), dtype=bool)
        queue = [(self.start_pos, 0, [])]  # (position, distance, path)
        visited[self.start_pos[1], self.start_pos[0]] = True
        
        all_paths = []
        
        while queue:
            (x, y), dist, path = queue.pop(0)
            
            # Add current position to path
            current_path = path + [(x, y)]
            
            # Check if reached exit
            if (x, y) == self.exit_pos:
                all_paths.append((dist, current_path))
                continue
            
            # Check all four directions
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    not visited[ny, nx] and self.grid[ny, nx] != WALL):
                    visited[ny, nx] = True
                    queue.append(((nx, ny), dist + 1, current_path))
        
        # Calculate complexity based on number of paths and their lengths
        if not all_paths:
            return 0
        
        # Normalize path length to grid size
        avg_path_length = sum(p[0] for p in all_paths) / len(all_paths)
        normalized_path_length = avg_path_length / (self.width * self.height)
        
        # Calculate path diversity (how different the paths are)
        path_diversity = 0
        if len(all_paths) > 1:
            unique_positions = set()
            for _, path in all_paths:
                for pos in path:
                    unique_positions.add(pos)
            
            # Normalize by total possible positions
            path_diversity = len(unique_positions) / (self.width * self.height)
        
        # Combine metrics
        return (normalized_path_length * 0.6 + path_diversity * 0.4)

    def _calculate_path_length(self):
        """Calculate shortest path length from start to exit"""
        # Perform BFS from start to exit
        visited = np.zeros((self.height, self.width), dtype=bool)
        queue = [(self.start_pos, 0)]  # (position, distance)
        visited[self.start_pos[1], self.start_pos[0]] = True
        
        while queue:
            (x, y), dist = queue.pop(0)
            
            # Check if reached exit
            if (x, y) == self.exit_pos:
                return dist
            
            # Check all four directions
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    not visited[ny, nx] and self.grid[ny, nx] != WALL):
                    visited[ny, nx] = True
                    queue.append(((nx, ny), dist + 1))
        
        return float('inf')  # No path found

    def _calculate_exploration_value(self):
        """Calculate how much the map encourages exploration"""
        # Count treasures and their distribution
        treasure_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == TREASURE:
                    treasure_positions.append((x, y))
        
        if not treasure_positions:
            return 0
        
        # Calculate average distance between treasures
        total_distance = 0
        count = 0
        
        for i in range(len(treasure_positions)):
            for j in range(i+1, len(treasure_positions)):
                x1, y1 = treasure_positions[i]
                x2, y2 = treasure_positions[j]
                distance = abs(x1 - x2) + abs(y1 - y2)
                total_distance += distance
                count += 1
        
        if count == 0:
            return 0
        
        avg_treasure_distance = total_distance / count
        
        # Normalize to grid size
        normalized_distance = avg_treasure_distance / (self.width + self.height)
        
        return normalized_distance
    
    def _calculate_strategic_value(self):
        """Calculate strategic value of the dungeon (risk vs reward)"""
        # Count lava and treasures near each other
        strategic_score = 0
        
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.grid[y, x] == TREASURE:
                    # Check if there's lava nearby
                    lava_nearby = 0
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height and
                                self.grid[ny, nx] == LAVA):
                                lava_nearby += 1
                    
                    # Good strategic placement has some lava nearby but not too much
                    if 1 <= lava_nearby <= 3:
                        strategic_score += 1
        
        # Normalize by number of treasures
        treasure_count = np.sum(self.grid == TREASURE)
        if treasure_count == 0:
            return 0
        
        return strategic_score / treasure_count

    
    def _creates_dead_end(self, x, y):
        """Check if placing a wall at (x,y) creates a dead end"""
        # Temporarily place the wall
        original_value = self.grid[y, x]
        self.grid[y, x] = WALL
        
        # Count accessible directions from neighboring cells
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] != WALL:
                # Count accessible directions from this neighbor
                accessible_dirs = 0
                for ndx, ndy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nnx, nny = nx + ndx, ny + ndy
                    if (0 <= nnx < self.width and 0 <= nny < self.height and 
                        self.grid[nny, nnx] != WALL):
                        accessible_dirs += 1
                
                # If a cell has only one exit, it's a dead end
                if accessible_dirs <= 1:
                    # Restore original value
                    self.grid[y, x] = original_value
                    return True
        
        # Restore original value
        self.grid[y, x] = original_value
        return False
    
    def render(self, mode='text'):
        """Render the current dungeon state"""
        if mode == 'text':
            symbols = {
                EMPTY: '.',
                WALL: '#',
                LAVA: 'L',
                TREASURE: 'T',
                EXIT: 'E',
                START: 'S'
            }
            
            for y in range(self.height):
                row = ''
                for x in range(self.width):
                    row += symbols[self.grid[y, x]] + ' '
                print(row)
            print()
        
        # Could add pygame visualization mode here if needed

    def _calculate_treasure_reward(self, x, y):
        """Calculate reward for treasure placement"""
        reward = 0
        
        # Calculate distance from start
        dist_from_start = abs(x - self.start_pos[0]) + abs(y - self.start_pos[1])
        dist_from_exit = abs(x - self.exit_pos[0]) + abs(y - self.exit_pos[1])
        
        # Normalize distances to grid size
        normalized_start_dist = dist_from_start / (self.width + self.height)
        normalized_exit_dist = dist_from_exit / (self.width + self.height)
        
        # Reward treasures that are farther from start and exit
        reward += normalized_start_dist * 0.5
        reward += normalized_exit_dist * 0.3
        
        # Check spacing from other treasures
        other_treasures_nearby = 0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    self.grid[ny, nx] == TREASURE and
                    (nx != x or ny != y)):
                    # Closer treasures are more problematic
                    distance = abs(dx) + abs(dy)
                    if distance <= 2:
                        other_treasures_nearby += 1
        
        # Penalize treasures that are too close to each other
        if other_treasures_nearby > 0:
            reward -= other_treasures_nearby * 0.2
        else:
            reward += 0.5  # Bonus for well-spaced treasures
        
        return reward

    def _calculate_wall_reward(self, x, y):
        """Calculate reward for wall placement"""
        # Check if wall would create dead ends
        if self._creates_dead_end(x, y):
            return -0.5
        
        # Check if wall creates interesting navigation
        reward = 0
        
        # Get current number of paths from start to exit
        path_count_before = self._count_paths_to_exit()
        
        # Temporarily place wall
        orig_value = self.grid[y, x]
        self.grid[y, x] = WALL
        
        # Check number of paths after wall
        path_count_after = self._count_paths_to_exit()
        
        # Restore original value
        self.grid[y, x] = orig_value
        
        # Reward walls that don't completely block paths but make navigation more complex
        if 0 < path_count_after < path_count_before:
            reward += 0.3
        elif path_count_after == 0:
            reward -= 0.5  # Penalize walls that block all paths
        
        return reward

    def _count_paths_to_exit(self, max_paths=10):
        """Count number of unique paths from start to exit (limited sample)"""
        # Simplified BFS with path tracking (only samples up to max_paths)
        visited = set()
        queue = [(self.start_pos, [self.start_pos])]
        paths_found = 0
        
        while queue and paths_found < max_paths:
            pos, path = queue.pop(0)
            
            if pos == self.exit_pos:
                paths_found += 1
                continue
            
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                new_pos = (nx, ny)
                
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    self.grid[ny, nx] != WALL and
                    new_pos not in path):
                    
                    new_path = path + [new_pos]
                    
                    # Use a string representation of the path for visited check
                    path_key = str(new_path)
                    if path_key not in visited:
                        visited.add(path_key)
                        queue.append((new_pos, new_path))
        
        return paths_found
    
    def _calculate_strategic_lava_reward(self, x, y):
        """Calculate reward for strategic lava placement"""
        reward = 0
        
        # Check if lava creates interesting paths around it
        accessible_neighbors = 0
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and
                self.grid[ny, nx] != WALL and
                self.grid[ny, nx] != LAVA):
                accessible_neighbors += 1
        
        # Lava with many accessible neighbors creates interesting navigation challenges
        if accessible_neighbors >= 3:
            reward += 0.5
        
        # Lava near treasures creates risk/reward scenarios
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    self.grid[ny, nx] == TREASURE):
                    # Closer treasures are more interesting
                    distance = abs(dx) + abs(dy)
                    if distance <= 2:
                        reward += 0.3
                    elif distance <= 3:
                        reward += 0.1
        
        return reward

    # 6. Increase training time and exploration in train_dungeon_generator function:
def train_dungeon_generator(width, height, num_episodes=5000):
        """Train the DQN agent to generate dungeons with more complex maps"""
        env = DungeonEnvironment(width, height)
        
        # Calculate state and action sizes
        num_cell_types = len(env.element_types)
        state_size = width * height * num_cell_types
        action_size = width * height * num_cell_types
        
        agent = DungeonAgent(state_size=state_size, action_size=action_size)
        
        # Modify DQN agent to emphasize exploration for longer
        agent.epsilon_decay = 0.9995  # Slower decay for more exploration
        agent.epsilon_min = 0.05      # Higher minimum exploration
        
        # Training loop
        scores = []
        best_score = float('-inf')
        best_grid = None
        
        for episode in range(1, num_episodes+1):
            state = env.reset()
            score = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
            
            scores.append(score)
            
            # Save best grid
            if score > best_score:
                best_score = score
                best_grid = np.copy(env.grid)
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes}, Average Score: {np.mean(scores[-100:])}")
                
                # Show a sample dungeon
                if episode % 500 == 0:
                    print("Sample dungeon:")
                    env.render()
        
        # Store best grid in agent for later use
        agent.best_grid = best_grid
        
        return agent, env
def train_dungeon_generator(width, height, num_episodes=1000):
        """Train the DQN agent to generate dungeons"""
        env = DungeonEnvironment(width, height)
        
        # Calculate state and action sizes
        num_cell_types = len(env.element_types)
        state_size = width * height * num_cell_types
        action_size = width * height * num_cell_types
        
        agent = DungeonAgent(state_size=state_size, action_size=action_size)
        
        # Training loop
        scores = []
        
        for episode in range(1, num_episodes+1):
            state = env.reset()
            score = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
            
            scores.append(score)
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes}, Average Score: {np.mean(scores[-100:])}")
                
                # Show a sample dungeon
                if episode % 500 == 0:
                    print("Sample dungeon:")
                    env.render()
        
        return agent, env

def generate_dungeon_with_model(agent, width, height, eval_mode=True):
        """Generate a dungeon using a trained RL agent"""
        env = DungeonEnvironment(width, height)
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, eval_mode=eval_mode)
            state, _, done = env.step(action)
        
        # Return the generated grid
        return env.grid

    # Integration with the game
def convert_rl_grid_to_game_grid(rl_grid, GRID_WIDTH, GRID_HEIGHT, Cell):
        """Convert RL generated grid to the game's grid format"""
        game_grid = [[Cell(x, y) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]
        
        # Map RL grid values to game grid cell properties
        for y in range(len(rl_grid)):
            for x in range(len(rl_grid[0])):
                if rl_grid[y, x] == WALL:
                    game_grid[y][x].is_wall = True
                elif rl_grid[y, x] == LAVA:
                    game_grid[y][x].is_lava = True
                elif rl_grid[y, x] == TREASURE:
                    game_grid[y][x].is_treasure = True
                elif rl_grid[y, x] == EXIT:
                    game_grid[y][x].is_exit = True
        
        return game_grid, 1, 1  # Return grid and start position
    


    # Example usage in main game:
    # 
    # # Train model (do this once before game starts)
    # agent, _ = train_dungeon_generator(GRID_WIDTH, GRID_HEIGHT, num_episodes=5000)
    #
    # # In the generate_dungeon function:
    # def generate_dungeon():
    #     # Generate dungeon using RL model
    #     rl_grid = generate_dungeon_with_model(agent, GRID_WIDTH, GRID_HEIGHT)
    #     
    #     # Convert to game format
    #     grid, start_x, start_y = convert_rl_grid_to_game_grid(rl_grid, GRID_WIDTH, GRID_HEIGHT, Cell)
    #     
    #     return grid, start_x, start_y