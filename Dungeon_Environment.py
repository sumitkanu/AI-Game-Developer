from Generator_Agent import EMPTY, WALL, LAVA, TREASURE, EXIT, START, ENEMY
import numpy as np
import torch
import random

class DungeonEnvironment:
    def __init__(self, difficulty=5, board_size=20):
        self.board_size = board_size
        self.difficulty = difficulty
        self.max_enemies = 4

        self.element_types = [EMPTY, WALL, LAVA, TREASURE, EXIT, START, ENEMY]
        self.placement_types = [LAVA, TREASURE, ENEMY, WALL, EMPTY]

        self.grid = np.zeros((board_size, board_size), dtype=np.int8)
        self.cursor = None
        self.start_pos = None
        self.exit_pos = None
        self.steps = 0
        self.max_steps = 600
        self._setup_playable_area()
        self.visited_tiles = set()

    def _setup_playable_area(self):
        """Define and fill the playable area based on difficulty."""
        # Scale playable area size with difficulty
        min_playable = 5
        max_playable = 18
        playable_size = min_playable + (self.difficulty - 1) * (max_playable - min_playable) // 9
        self.playable_width = self.playable_height = playable_size

        # Center the playable area
        self.playable_start_x = (self.board_size - self.playable_width) // 2
        self.playable_start_y = (self.board_size - self.playable_height) // 2
        self.playable_end_x = self.playable_start_x + self.playable_width - 1
        self.playable_end_y = self.playable_start_y + self.playable_height - 1

        self.object_counts = {k: 0 for k in self.placement_types}

    def reset(self):
        self.grid = np.zeros((self.board_size, self.board_size), dtype=int)
        self.steps = 0
        self._setup_playable_area()
        self.visited_tiles = set()

        # Fill outer border with walls
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (
                    x < self.playable_start_x or x > self.playable_end_x or
                    y < self.playable_start_y or y > self.playable_end_y
                ):
                    self.grid[y][x] = WALL

        #  Generate maze before placing START/EXIT
        self._generate_maze_layout()

        #  Place START
        self.start_pos = self._find_random_empty_cell()
        self.grid[self.start_pos[1]][self.start_pos[0]] = START

        # Place EXIT far away
        # Check find random empty cell
        min_dist = (self.playable_width + self.playable_height) * 0.75
        self.exit_pos = self._find_random_empty_cell(exclude=[self.start_pos], avoid_near=self.start_pos, min_distance=min_dist)
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = EXIT

        #self._carve_simple_path(self.start_pos, self.exit_pos)

        # Check lava and wall free path
        safe_path = self._calculate_path_complexity(return_path=True)
        if not safe_path:
            return self.reset()
        self.safe_path = set(safe_path)

        self.cursor = list(self.start_pos)
        self.object_counts = {k: 0 for k in self.placement_types}
        self.visited = {tuple(self.cursor)}

        # Determine playable interior area
        playable_area = (self.playable_width - 2) * (self.playable_height - 2)

        # --- Suggested object limits based on size + difficulty ---
        self.suggested_treasure = max(1, int(playable_area * 0.04) - self.difficulty)
        self.suggested_lava = int(playable_area * (0.01 + 0.01 * self.difficulty))

        #  New logic for treasure limit
        max_treasure = max(1, int(0.25 * (playable_area ** 0.5)) - (self.difficulty // 2))
        max_treasure = min(max_treasure, 10)

        #  Lava and wall density updates
        lava_density = 0.01 + 0.005 * self.difficulty
        wall_density = 0.2 + 0.02 * self.difficulty

        max_lava = int(playable_area * lava_density)

        #  Enemy tier rules
        if self.difficulty <= 4:
            self.allowed_enemy_limit = 1
        elif self.difficulty <= 7:
            self.allowed_enemy_limit = random.randint(2, 3)
        else:
            self.allowed_enemy_limit = 5

        #  Place all elements after maze and START/EXIT
        self._place_random_objects(TREASURE, max_treasure)
        self._place_random_objects(LAVA, max_lava)
        self._place_random_objects(ENEMY, self.allowed_enemy_limit)

        return self._get_state()

    def _find_random_empty_cell(self, exclude=[], avoid_near=None, min_distance=0):
        """Find an empty cell that avoids proximity and overlaps."""
        candidates = [
            (x, y)
            for y in range(self.playable_start_y + 1, self.playable_end_y)
            for x in range(self.playable_start_x + 1, self.playable_end_x)
            if self.grid[y][x] == EMPTY and (x, y) not in exclude
        ]

        if avoid_near and min_distance > 0:
            candidates = [
                (x, y)
                for (x, y) in candidates
                if abs(x - avoid_near[0]) + abs(y - avoid_near[1]) >= min_distance
            ]

        return random.choice(candidates) if candidates else (self.playable_start_x + 1, self.playable_start_y + 1)

    def _place_random_objects(self, tile, count):
        placed = 0
        attempts = 0
        max_attempts = count * 30
        objects = random.randint(0, count)
        existing_positions = [
            (x, y)
            for y in range(self.playable_start_y + 1, self.playable_end_y)
            for x in range(self.playable_start_x + 1, self.playable_end_x)
            if self.grid[y][x] == tile
        ]

        # Generate a noise map for more natural placement
        noise = np.random.rand(self.board_size, self.board_size)

        while placed < objects and attempts < max_attempts:
            x = random.randint(self.playable_start_x + 1, self.playable_end_x - 1)
            y = random.randint(self.playable_start_y + 1, self.playable_end_y - 1)

            if (x, y) in [self.start_pos, self.exit_pos]:
                attempts += 1
                continue

            if self.grid[y][x] != EMPTY:
                attempts += 1
                continue

            if tile == TREASURE:
                # Keep treasures away from START
                if abs(x - self.start_pos[0]) + abs(y - self.start_pos[1]) < 5:
                    attempts += 1
                    continue

                # Avoid placing next to another treasure
                neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx != 0 or dy != 0)]
                near_treasure = any(
                    0 <= nx < self.board_size and 0 <= ny < self.board_size and self.grid[ny][nx] == TREASURE
                    for nx, ny in neighbors
                )
                if near_treasure:
                    attempts += 1
                    continue

                # Soft-distribution using noise map
                if noise[y][x] < 0.3:  # require higher randomness for spacing
                    attempts += 1
                    continue


            #  Don’t place lava on the safe path
            if tile == LAVA and (x, y) in getattr(self, 'safe_path', set()):
                attempts += 1
                continue

            # Use noise as a soft priority (prefer high noise values)
            if noise[y][x] < 0.5:
                attempts += 1
                continue

            self.grid[y][x] = tile
            existing_positions.append((x, y))
            placed += 1
            if tile in self.object_counts:
                self.object_counts[tile] += 1

            attempts += 1

    def step(self, action):
        self.steps += 1
        reward = 0
        done = False
    
        if action < 4:
            # Movement: up, down, left, right
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_x = min(max(self.cursor[0] + dx, self.playable_start_x + 1), self.playable_end_x - 1)
            new_y = min(max(self.cursor[1] + dy, self.playable_start_y + 1), self.playable_end_y - 1)
            self.cursor = [new_x, new_y]
            self.visited_tiles.add(tuple(self.cursor))
            reward -= 0.1  # discourage wandering
        else:
            prev_path_exists = self._calculate_path_complexity() > 0
            tile = self.placement_types[action - 4]
            cx, cy = self.cursor
            current_tile = self.grid[cy][cx]
    
            if (cx, cy) in [self.start_pos, self.exit_pos]:
                reward -= 15
            else:
                if current_tile in self.object_counts:
                    self.object_counts[current_tile] = max(0, self.object_counts[current_tile] - 1)
    
                self.grid[cy][cx] = tile
                
                if tile in self.object_counts:
                    self.object_counts[tile] += 1
    
                if tile == WALL:
                    reward += 2
                elif tile == TREASURE:
                    if self.object_counts[TREASURE] > self.suggested_treasure:
                        reward -= 5
                    else:
                        dist_to_start = abs(cx - self.start_pos[0]) + abs(cy - self.start_pos[1])
                        reward += 3 if dist_to_start >= 4 else -3
    
                        nearby = sum(
                            1 for dx in range(-2, 3) for dy in range(-2, 3)
                            if 0 <= cx + dx < self.board_size and 0 <= cy + dy < self.board_size and
                            self.grid[cy + dy][cx + dx] == TREASURE
                        )
                        if nearby > 2:
                            reward -= 2
    
                elif tile == ENEMY:
                    if self.object_counts[ENEMY] > self.allowed_enemy_limit:
                        reward -= 10
                    else:
                        reward += 4
    
                elif tile == LAVA:
                    if self.object_counts[LAVA] > self.suggested_lava:
                        reward -= 4
                    else:
                        reward += 2
    
                elif tile == EMPTY:
                    reward -= 1
    
                new_path_exists = self._calculate_path_complexity() > 0
                if new_path_exists:
                    reward += 10
                if prev_path_exists and not new_path_exists:
                    reward -= 50
                elif not prev_path_exists and new_path_exists:
                    reward += 50
    
        # ✅ End of episode logic
        if self.steps >= self.max_steps:
            done = True
            complexity = self._calculate_path_complexity()
            exploration_bonus = len(self.visited_tiles)  # Each new tile visited gives +0.2
            reward += exploration_bonus
    
            if complexity > 0:
                target_complexity = min(1.0, 0.3 + 0.07 * self.difficulty)
                reward += (1 - abs(complexity - target_complexity)) * 20 + complexity * 100
            else:
                reward -= 100
    
        return self._get_state(), reward, done


    def _get_state(self):
        """
        Returns:
            - grid_tensor: [1, 8, 20, 20] — for CNN input
            - difficulty_tensor: [1, 1] — appended in FC layer
        """
        one_hot_grid = np.zeros((7, self.board_size, self.board_size), dtype=np.float32)
        for i in range(7):
            one_hot_grid[i] = (self.grid == i).astype(np.float32)
    
        # Cursor as binary channel
        cursor_channel = np.zeros((1, self.board_size, self.board_size), dtype=np.float32)
        cursor_x, cursor_y = self.cursor
        cursor_channel[0, cursor_y, cursor_x] = 1.0
    
        # Stack to form 8-channel input
        combined = np.concatenate([one_hot_grid, cursor_channel], axis=0)  # [8, 20, 20]
    
        grid_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  # [1, 8, 20, 20]
        difficulty_tensor = torch.tensor([[self.difficulty / 10.0]], dtype=torch.float32)  # [1, 1]
    
        return grid_tensor, difficulty_tensor

    def _calculate_path_complexity(self, return_path=False):
        allowed_tiles = {EMPTY, START, EXIT, ENEMY, TREASURE, LAVA}
        visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        queue = [(self.start_pos, 0, [self.start_pos])]
        visited[self.start_pos[1]][self.start_pos[0]] = True
    
        best_path = []
        best_score = 0
    
        while queue:
            (x, y), dist, path = queue.pop(0)
    
            if (x, y) == self.exit_pos:
                # ✅ Measure complexity components
                num_special_tiles = sum(
                    1 for (px, py) in path
                    if self.grid[py][px] in {ENEMY, TREASURE, LAVA}
                )
    
                normalized_length = dist / (self.playable_width + self.playable_height)
                complexity_score = normalized_length + 0.05 * num_special_tiles
    
                # Optionally keep most complex valid path
                if complexity_score > best_score:
                    best_score = complexity_score
                    best_path = path
    
                continue
    
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (self.playable_start_x <= nx <= self.playable_end_x and
                    self.playable_start_y <= ny <= self.playable_end_y and
                    not visited[ny, nx] and self.grid[ny][nx] in allowed_tiles):
                    visited[ny, nx] = True
                    queue.append(((nx, ny), dist + 1, path + [(nx, ny)]))
    
        if return_path:
            return best_path
        return best_score  # Returns a score based on length + meaningful objects


    def _generate_maze_layout(self):
        """Prim's Algorithm to generate maze walls inside playable area"""
        maze = np.ones((self.playable_height, self.playable_width), dtype=np.int8)

        def in_bounds(x, y):
            return 0 <= x < self.playable_width and 0 <= y < self.playable_height

        def neighbors(x, y):
            dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            result = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny):
                    result.append((nx, ny))
            return result

        start_x = random.randrange(1, self.playable_width, 2)
        start_y = random.randrange(1, self.playable_height, 2)
        maze[start_y][start_x] = 0
        frontier = [(start_x, start_y)]

        while frontier:
            x, y = frontier.pop(random.randint(0, len(frontier) - 1))
            for nx, ny in neighbors(x, y):
                if maze[ny][nx] == 1:
                    between_x = (x + nx) // 2
                    between_y = (y + ny) // 2
                    maze[ny][nx] = 0
                    maze[between_y][between_x] = 0
                    frontier.append((nx, ny))

        # Convert maze to global coordinates
        for y in range(self.playable_height):
            for x in range(self.playable_width):
                global_x = self.playable_start_x + x
                global_y = self.playable_start_y + y
                if maze[y][x] == 1:
                    self.grid[global_y][global_x] = WALL