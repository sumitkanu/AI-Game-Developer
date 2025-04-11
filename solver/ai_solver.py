import numpy as np
import random
from collections import deque
from numba import njit

# Map elements
EMPTY = 0
WALL = 1
LAVA = 2
TREASURE = 3
EXIT = 4
START = 5
ENEMY = 6

DEBUG = False  # Set to True to print Q-values and moves

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class DungeonSolverQLearning:
    def __init__(self, dungeon, start, goal, alpha=0.1, gamma=0.9, epsilon=0.9, episodes=50000):
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
            EMPTY: -1,
            WALL: -1000,
            LAVA: -1000,
            TREASURE: 50,
            EXIT: 5000,
            START: 0,
            ENEMY: -250
        }

    def train(self):
        rewards_array = np.array([
            self.rewards[EMPTY],    # 0
            self.rewards[WALL],     # 1
            self.rewards[LAVA],     # 2
            self.rewards[TREASURE], # 3
            self.rewards[EXIT],     # 4
            self.rewards[START],    # 5
            self.rewards[ENEMY],    # 6
        ], dtype=np.float32)

        self.dungeon = np.copy(self.original_dungeon)
        self.q_table = q_learning_train_loop(
            self.dungeon, self.q_table, self.start, self.goal,
            rewards_array, self.alpha, self.gamma,
            self.episodes, max_steps=500,
            epsilon_start=1.0, epsilon_end=0.1
        )

    # def train(self):
    #     max_steps = 500
    #     gx, gy = self.goal

    #     for episode in range(self.episodes):
    #         self.dungeon = np.copy(self.original_dungeon)
    #         self.dungeon[gx, gy] = EXIT

    #         x, y = self.start
    #         health = 100
    #         steps = 0

    #         while (x, y) != self.goal and steps < max_steps:
    #             steps += 1

    #             if random.uniform(0, 1) < self.epsilon:
    #                 action = random.choice(range(4))
    #             else:
    #                 action = np.argmax(self.q_table[x, y])

    #             dx, dy = self.actions[action]
    #             nx, ny = x + dx, y + dy

    #             if 0 <= nx < self.dungeon.shape[0] and 0 <= ny < self.dungeon.shape[1]:
    #                 cell_type = self.dungeon[nx, ny]

    #                 if cell_type == WALL:
    #                     continue

    #                 if cell_type == LAVA:
    #                     self.q_table[x, y, action] = -1000
    #                     break

    #                 if cell_type == ENEMY:
    #                     reward = self.rewards[ENEMY]
    #                     self.q_table[x, y, action] = (1 - self.alpha) * self.q_table[x, y, action] + \
    #                         self.alpha * (reward + self.gamma * 0)
    #                     health -= 20
    #                     if health <= 0:
    #                         break

    #                 reward = self.rewards.get(cell_type, -1)

    #                 if cell_type == TREASURE:
    #                     health += 50
    #                     reward += 100
    #                     self.dungeon[nx, ny] = EMPTY

    #                 max_future_q = np.max(self.q_table[nx, ny])
    #                 self.q_table[x, y, action] = (1 - self.alpha) * self.q_table[x, y, action] + \
    #                     self.alpha * (reward + self.gamma * max_future_q)

    #                 x, y = nx, ny
    #         self.epsilon = 1.0
    #         self.epsilon = max(0.01, self.epsilon * 0.995)

    def solve(self):
        path = []
        x, y = self.start
        health = 100
        self.dungeon = np.copy(self.original_dungeon)
        visited = {(x, y): 1}
        visited_treasures = set()
        recent_positions = deque(maxlen=10)
        steps = 0
        max_steps = 200
        last_position = None

        while (x, y) != self.goal and steps < max_steps:
            path.append((x, y))
            visited[(x, y)] = visited.get((x, y), 0) + 1
            recent_positions.append((x, y))

            if visited[(x, y)] > 5:
                print("Loop detected. Recent path:", path[-5:])
                return path, f"Stuck in a loop at {(x, y)}, game over."

            steps += 1
            possible_moves = {}
            treasure_moves = []

            for i, (dx, dy) in enumerate(self.actions):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.dungeon.shape[0] and 0 <= ny < self.dungeon.shape[1]:
                    if (nx, ny) == last_position:
                        continue

                    cell = self.dungeon[nx, ny]
                    if (nx, ny) in visited_treasures and cell == TREASURE:
                        cell = EMPTY

                    if cell != WALL:
                        nnx, nny = nx + dx, ny + dy
                        future_penalty = 0
                        if 0 <= nnx < self.dungeon.shape[0] and 0 <= nny < self.dungeon.shape[1]:
                            future_cell = self.dungeon[nnx, nny]
                            if future_cell == LAVA:
                                future_penalty -= 200
                            elif future_cell == WALL:
                                future_penalty -= 100
                            elif future_cell == TREASURE:
                                future_penalty += 50

                        score = self.q_table[x, y, i] + future_penalty

                        if cell == TREASURE and (nx, ny) not in visited_treasures and (nx, ny) not in recent_positions:
                            treasure_moves.append(i)

                        if cell == START and (nx, ny) != self.start:
                            continue
                        if (nx, ny) == self.start and (nx, ny) != (x, y):
                            score -= 10

                        possible_moves[i] = score

            if not possible_moves:
                return path, "No valid moves left, game over."

            if treasure_moves:
                action = random.choice(treasure_moves)
            else:
                best_q = max(possible_moves.values())
                best_actions = [i for i in possible_moves if possible_moves[i] == best_q]
                if len(best_actions) > 1:
                    best_actions.sort(key=lambda i: manhattan((x + self.actions[i][0], y + self.actions[i][1]), self.goal))
                action = best_actions[0]

            dx, dy = self.actions[action]
            nx, ny = x + dx, y + dy
            cell_type = self.dungeon[nx, ny]

            if (nx, ny) in visited_treasures and cell_type == TREASURE:
                cell_type = EMPTY

            if cell_type == LAVA:
                return path, "Fell into lava, game over."

            if cell_type == ENEMY:
                health -= 20
                if health <= 0:
                    return path, "Defeated by enemies, game over."

            if cell_type == TREASURE and (nx, ny) not in visited_treasures:
                health += 50
                visited_treasures.add((nx, ny))
                self.dungeon[nx, ny] = EMPTY

            last_position = (x, y)
            x, y = nx, ny

        path.append((x, y))

        if (x, y) == self.goal:
            return path, f"Reached exit at {self.goal} with {health} health."
        else:
            return path, "Did not reach exit, ran out of steps."

@njit
def q_learning_train_loop(dungeon, q_table, start, goal, rewards, alpha, gamma,
                          episodes, max_steps, epsilon_start, epsilon_end):

    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    gx, gy = goal
    dungeon_shape = dungeon.shape

    for ep in range(episodes):
        
        epsilon = epsilon_start - ((epsilon_start - epsilon_end) * (ep / episodes))
        x, y = start
        health = 100

        for step in range(max_steps):
            if (x, y) == goal:
                break

            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(q_table[x, y])

            dx, dy = actions[action]
            nx, ny = x + dx, y + dy

            if not (0 <= nx < dungeon_shape[0] and 0 <= ny < dungeon_shape[1]):
                continue

            cell_type = dungeon[nx, ny]
            if cell_type == 1:  # WALL
                continue
            if cell_type == 2:  # LAVA
                q_table[x, y, action] = -1000
                break

            reward = rewards[cell_type]

            if cell_type == 6:  # ENEMY
                health -= 20
                if health <= 0:
                    break

            if cell_type == 3:  # TREASURE
                health += 50
                reward += 100
                dungeon[nx, ny] = 0  # clear treasure

            max_future_q = np.max(q_table[nx, ny])
            q_table[x, y, action] = (1 - alpha) * q_table[x, y, action] + \
                                    alpha * (reward + gamma * max_future_q)

            x, y = nx, ny

    return q_table

def solve_generated_level(dungeon, start, goal):
    solver = DungeonSolverQLearning(dungeon, start, goal)
    solver.train()
    return solver.solve()
