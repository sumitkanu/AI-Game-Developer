import numpy as np
import random

# Define map elements
EMPTY = 0
WALL = 1
LAVA = 2
TREASURE = 3
EXIT = 4
START = 5
ENEMY = 6

class DungeonSolverQLearning:
    def __init__(self, dungeon, start, goal, alpha=0.1, gamma=0.9, epsilon=0.9, episodes=100000):
        self.original_dungeon = np.copy(dungeon)  # Store original state
        self.dungeon = np.copy(dungeon)  # Working copy
        self.start = start
        self.goal = goal
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.episodes = episodes  
        self.q_table = np.zeros((*dungeon.shape, 4))  # Q-table
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

        # Rewards mapping
        self.rewards = {
            EMPTY: -0.1,
            WALL: -100,
            LAVA: -1000,
            TREASURE: 50,
            EXIT: 100,
            START: 0,
            ENEMY: -20
        }

    def train(self):
        """Train the Q-learning agent to solve the dungeon."""
        for episode in range(self.episodes):
            self.dungeon = np.copy(self.original_dungeon)  # ✅ Reset dungeon
            x, y = self.start
            health = 100  

            while (x, y) != self.goal:
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(range(4))  # Explore
                else:
                    action = np.argmax(self.q_table[x, y])  # Exploit learned values

                dx, dy = self.actions[action]
                nx, ny = x + dx, y + dy

                if 0 <= nx < self.dungeon.shape[0] and 0 <= ny < self.dungeon.shape[1]:
                    cell_type = self.dungeon[nx, ny]

                    if cell_type == WALL:
                        continue  

                    if cell_type == LAVA:
                        self.q_table[x, y, action] = -1000  
                        break  

                    if cell_type == ENEMY:
                        health -= 20  
                        if health <= 0:
                            break  

                    reward = self.rewards.get(cell_type, -1)

                    if cell_type == TREASURE:
                        health += 50  
                        reward += 100  
                        self.dungeon[nx, ny] = EMPTY  

                    max_future_q = np.max(self.q_table[nx, ny]) if 0 <= nx < self.q_table.shape[0] and 0 <= ny < self.q_table.shape[1] else 0
                    self.q_table[x, y, action] = (1 - self.alpha) * self.q_table[x, y, action] + \
                                                 self.alpha * (reward + self.gamma * max_future_q)

                    x, y = nx, ny

            self.epsilon = max(0.01, self.epsilon * 0.995)

    def solve(self):
        """Use the trained Q-table to find an optimal path."""
        path = []
        x, y = self.start
        health = 100  

        # ✅ Reset dungeon before solving to properly track treasures
        self.dungeon = np.copy(self.original_dungeon)

        while (x, y) != self.goal:
            path.append((x, y))
            possible_moves = {}  
            treasure_moves = []  

            for i, (dx, dy) in enumerate(self.actions):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.dungeon.shape[0] and 0 <= ny < self.dungeon.shape[1]:
                    if self.dungeon[nx, ny] != WALL:
                        if self.dungeon[nx, ny] == TREASURE:
                            treasure_moves.append(i)
                        possible_moves[i] = self.q_table[x, y, i]

            if not possible_moves:
                return path, "No valid moves left, game over."

            action = random.choice(treasure_moves) if treasure_moves else max(possible_moves, key=possible_moves.get)
            dx, dy = self.actions[action]
            nx, ny = x + dx, y + dy
            cell_type = self.dungeon[nx, ny]

            if cell_type == LAVA:
                return path, "Fell into lava, game over."

            if cell_type == ENEMY:
                health -= 20
                if health <= 0:
                    return path, "Defeated by enemies, game over."
            
            # ✅ Ensure treasure is collected and health is updated
            if cell_type == TREASURE and self.dungeon[nx, ny] == TREASURE:
                health += 50  
                self.dungeon[nx, ny] = EMPTY  

            x, y = nx, ny

        path.append(self.goal)
        return path, f"Reached exit with {health} health."

# Function to solve a generated level
def solve_generated_level(dungeon, start, goal):
    solver = DungeonSolverQLearning(dungeon, start, goal)
    solver.train()
    path, result = solver.solve()
    return path, result

# Example input dungeon (provided by level generator)
dungeon = np.array([
    # [5,  0,  1,  0,  3],
    # [0,  1,  0,  1,  6],
    # [0,  0,  6,  3,  3],
    # [1,  0,  1,  3,  3],
    # [0,  0,  0,  0,  4]
    [5, 0, 1, 0, 3],
    [0, 1, 0, 1, 6],
    [0, 0, 6, 3, 3],
    [1, 0, 1, 3, 3],
    [0, 0, 0, 0, 4]
])

start = (0, 2)  # Start position
goal = (4,4)   # Exit position

path, outcome = solve_generated_level(dungeon, start, goal)
print("Solved Path:", path)
print("Outcome:", outcome)
