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
        self.original_dungeon = np.copy(dungeon)
        self.dungeon = np.copy(dungeon)
        self.start = start
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((*dungeon.shape, 4))
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

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
        max_steps = 200  # Limit per episode

        for episode in range(self.episodes):
            self.dungeon = np.copy(self.original_dungeon)
            x, y = self.start
            health = 100
            steps = 0

            while (x, y) != self.goal and steps < max_steps:
                steps += 1

                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(range(4))  # Explore
                else:
                    action = np.argmax(self.q_table[x, y])  # Exploit

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
                        self.dungeon[nx, ny] = EMPTY  # Mark as collected

                    max_future_q = np.max(self.q_table[nx, ny])
                    self.q_table[x, y, action] = (1 - self.alpha) * self.q_table[x, y, action] + \
                                                 self.alpha * (reward + self.gamma * max_future_q)

                    x, y = nx, ny

            self.epsilon = max(0.01, self.epsilon * 0.995)

    def solve(self):
        path = []
        x, y = self.start
        health = 100
        self.dungeon = np.copy(self.original_dungeon)
        visited = set()
        steps = 0
        max_steps = 200

        while (x, y) != self.goal and steps < max_steps:
            path.append((x, y))
            visited.add((x, y))
            steps += 1

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

            if (nx, ny) in visited and cell_type == ENEMY:
                return path, "Defeated by enemies (looped into enemy tile)."

            if cell_type == LAVA:
                return path, "Fell into lava, game over."

            if cell_type == ENEMY:
                health -= 20
                if health <= 0:
                    return path, "Defeated by enemies, game over."

            if cell_type == TREASURE:
                print(f"Collected treasure at {(nx, ny)}")
                health += 50
                self.dungeon[nx, ny] = EMPTY

            x, y = nx, ny

        path.append((x, y))

        if (x, y) == self.goal:
            return path, f"Reached exit with {health} health."
        else:
            return path, "Did not reach exit, ran out of steps."

def solve_generated_level(dungeon, start, goal):
    solver = DungeonSolverQLearning(dungeon, start, goal)
    solver.train()
    return solver.solve()
