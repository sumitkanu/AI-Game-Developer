import pygame
import random
import math
import sys

# Initial pygame
pygame.init()

# Set up game window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Dungeons")

# Color define
DEEP_BLUE = (0, 0, 139)  # Dark blue for background
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
BROWN = (165, 42, 42)

# Game parameters 
CELL_SIZE = 40
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
PLAYER_SPEED = 5
ENEMY_SPEED = 3
MAX_HEALTH = 100
HEALTH_DECREASE = 10
TREASURE_SCORE = 50
FPS = 60

# Game status
score = 0
health = MAX_HEALTH
fell_into_lava = False
game_over = False
level_complete = False  # Track level completion status

# Clock control
clock = pygame.time.Clock()

# Font
font = pygame.font.SysFont('Arial', 24)

# Load door image
door_image = pygame.image.load('Door.png')
door_image = pygame.transform.scale(door_image, (CELL_SIZE, CELL_SIZE))

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_wall = False
        self.is_black_wall = False 
        self.is_lava = False       
        self.is_treasure = False
        self.is_visited = False
        self.is_exit = False  # Add exit door flag

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = CELL_SIZE - 10
        self.height = CELL_SIZE - 10
        self.vel = PLAYER_SPEED
        self.rect = pygame.Rect(x, y, self.width, self.height)
    
    def draw(self):
        pygame.draw.rect(screen, GREEN, self.rect)
    
    # Added a new check function, specifically used to check whether the player is on magma
    def check_lava(self, grid):
        grid_x = int(self.x // CELL_SIZE)
        grid_y = int(self.y // CELL_SIZE)
        
        # Make sure the coordinates are within the valid range
        if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
            # Check center point
            if grid[grid_y][grid_x].is_lava:
                return True
                
            # Check all four corners for greater accuracy
            corners = [
                (self.x, self.y),  # left up
                (self.x + self.width, self.y),  # right up
                (self.x, self.y + self.height),  # left down
                (self.x + self.width, self.y + self.height)  # right down
            ]
            
            for corner_x, corner_y in corners:
                cx = int(corner_x // CELL_SIZE)
                cy = int(corner_y // CELL_SIZE)
                if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT and grid[cy][cx].is_lava:
                    return True
        
        return False
    
    # Fix: indent move method to be part of the Player class
    def move(self, dx, dy, grid):
        global game_over, level_complete
        # Calculate new position first
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Create a temporary rectangle to check the new position
        temp_rect = pygame.Rect(new_x, new_y, self.width, self.height)
        
        # Check if the new location is within the bounds and not a wall
        can_move = True
        
        # Get all grid cells covered by the temporary rectangle
        grid_positions = []
        for x in range(new_x, new_x + self.width + 1, max(1, self.width // 2)):
            for y in range(new_y, new_y + self.height + 1, max(1, self.height // 2)):
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    grid_x = int(x // CELL_SIZE)
                    grid_y = int(y // CELL_SIZE)
                    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                        grid_positions.append((grid_x, grid_y))
        
        # Check for lava
        for grid_x, grid_y in grid_positions:
            if grid[grid_y][grid_x].is_lava:
                global health, fell_into_lava
                health = 0
                fell_into_lava = True
                return False
        
        # Check if hits a wall
        for grid_x, grid_y in grid_positions:
            if grid[grid_y][grid_x].is_wall:
                can_move = False
                break

        # Check if reached exit door
        for grid_x, grid_y in grid_positions:
            if grid[grid_y][grid_x].is_exit:
                game_over = True
                level_complete = True
                return True
        
        # If moveable, update location
        if can_move:
            self.x = new_x
            self.y = new_y
            self.rect.x = new_x
            self.rect.y = new_y
            
            # Check for treasure collection
            for grid_x, grid_y in grid_positions:
                if grid[grid_y][grid_x].is_treasure:
                    global score
                    score += TREASURE_SCORE
                    grid[grid_y][grid_x].is_treasure = False
        
        return can_move

def is_path_accessible(grid, start_x, start_y, end_x, end_y):
    """Use BFS to check if there is a feasible path from the starting point to the end point"""
    if start_x == end_x and start_y == end_y:
        return True
        
    # Marking visited cells
    visited = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    visited[start_y][start_x] = True
    
    # BFS search using queues
    queue = [(start_x, start_y)]
    
    # Define four directions of movement: up, down, left, right
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    while queue:
        current_x, current_y = queue.pop(0)
        
        # Check four directions
        for dx, dy in directions:
            next_x, next_y = current_x + dx, current_y + dy
            
            # Check to see if it's within the grid
            if not (0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT):
                continue
                
            # Skip if already visited or walled in
            if visited[next_y][next_x] or grid[next_y][next_x].is_wall:
                continue
                
            # Returns True if the end point is reached
            if next_x == end_x and next_y == end_y:
                return True
                
            # Mark as visited and add to queue
            visited[next_y][next_x] = True
            queue.append((next_x, next_y))
    
    # If the queue is empty, the path was not found
    return False

def test_map_accessibility(grid):
    """Test whether the map can be reached from any point to another point"""
    # Find all walkable grids
    walkable_cells = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if not grid[y][x].is_wall:
                walkable_cells.append((x, y))
    
    # If there are no walkable grids or only one, return directly to the
    if len(walkable_cells) <= 1:
        return True
    
    # Choose the first grid as the starting point
    start_x, start_y = walkable_cells[0]
    
    # Check if all other walkable grids can be reached from the starting point
    for end_x, end_y in walkable_cells[1:]:
        if not is_path_accessible(grid, start_x, start_y, end_x, end_y):
            return False
    
    return True

def find_treasure_farthest_from_start(grid, start_x, start_y):
    """Find the farthest accessible location from the start point"""
    # Initialize distances and visited cells
    distances = [[-1 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    distances[start_y][start_x] = 0
    
    # Use queue for BFS search
    queue = [(start_x, start_y)]
    
    # Define movement directions (up, right, down, left)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    max_distance = 0
    farthest_point = (start_x, start_y)
    
    while queue:
        current_x, current_y = queue.pop(0)
        current_distance = distances[current_y][current_x]
        
        # Check all directions
        for dx, dy in directions:
            next_x, next_y = current_x + dx, current_y + dy
            
            # Check if within grid bounds
            if not (0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT):
                continue
                
            # Skip if visited or is wall
            if distances[next_y][next_x] != -1 or grid[next_y][next_x].is_wall:
                continue
                
            # Update distance
            distances[next_y][next_x] = current_distance + 1
            
            # Update farthest point if found
            if distances[next_y][next_x] > max_distance:
                max_distance = distances[next_y][next_x]
                farthest_point = (next_x, next_y)
                
            # Add to queue
            queue.append((next_x, next_y))
    
    return farthest_point, max_distance


class Enemy:
   def __init__(self, x, y):
       self.x = x
       self.y = y
       self.width = CELL_SIZE - 10
       self.height = CELL_SIZE - 10
       self.vel = ENEMY_SPEED
       self.rect = pygame.Rect(x, y, self.width, self.height)
       # Load monster icon
       self.monster_icon = pygame.image.load('monster.png')  
       self.monster_icon = pygame.transform.scale(self.monster_icon, (self.width, self.height))
   
   def draw(self):
       screen.blit(self.monster_icon, self.rect)  # Draw monster icon
   
   def move_towards_player(self, player, grid):
       # Calculate target position (aligned to grid)
       target_grid_x = int(player.x // CELL_SIZE)
       target_grid_y = int(player.y // CELL_SIZE)
       current_grid_x = int(self.x // CELL_SIZE)
       current_grid_y = int(self.y // CELL_SIZE)
       
       # Initialize movement
       dx = 0
       dy = 0
       
       # Determine movement direction based on grid position
       if current_grid_x < target_grid_x:
           dx = self.vel
       elif current_grid_x > target_grid_x:
           dx = -self.vel
           
       if current_grid_y < target_grid_y:
           dy = self.vel
       elif current_grid_y > target_grid_y:
           dy = -self.vel
       
       # Try horizontal movement first
       new_x = self.x + dx
       new_y = self.y
       if self.check_valid_move(new_x, new_y, grid):
           self.x = new_x
           self.rect.x = new_x
       
       # Try vertical movement
       new_x = self.x
       new_y = self.y + dy
       if self.check_valid_move(new_x, new_y, grid):
           self.y = new_y
           self.rect.y = new_y
   
   def check_valid_move(self, new_x, new_y, grid):
       # Get grid positions for all corners of the enemy
       corners = [
           (new_x, new_y),  # Top-left
           (new_x + self.width, new_y),  # Top-right
           (new_x, new_y + self.height),  # Bottom-left
           (new_x + self.width, new_y + self.height)  # Bottom-right
       ]
       
       # Check each corner
       for corner_x, corner_y in corners:
           grid_x = int(corner_x // CELL_SIZE)
           grid_y = int(corner_y // CELL_SIZE)
           
           # Check if position is valid
           if not (0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT):
               return False
           
           # Check if position contains wall or lava
           if grid[grid_y][grid_x].is_wall or grid[grid_y][grid_x].is_lava:
               return False
       
       return True
    
def generate_dungeon():
    """Generate a valid dungeon map automatically"""
    while True:  # Keep trying until we generate a valid map
        # Create grid
        grid = [[Cell(x, y) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]
        
        # Set starting point
        start_x, start_y = 1, 1
        
        # Add walls
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Border walls
                if x == 0 or y == 0 or x == GRID_WIDTH - 1 or y == GRID_HEIGHT - 1:
                    grid[y][x].is_wall = True
                # Random walls
                elif random.random() < 0.3 and not (x == start_x and y == start_y):
                    grid[y][x].is_wall = True
        
        # Create safe starting area
        safe_radius = 4
        for dy in range(-safe_radius, safe_radius + 1):
            for dx in range(-safe_radius, safe_radius + 1):
                nx, ny = start_x + dx, start_y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    grid[ny][nx].is_wall = False
                    grid[ny][nx].is_visited = True
        
        # Add connecting passages
        for i in range(2):
            # Horizontal passage
            h_y = random.randint(GRID_HEIGHT // 4, 3 * GRID_HEIGHT // 4)
            for x in range(1, GRID_WIDTH - 1):
                if random.random() < 0.7:
                    grid[h_y][x].is_wall = False
            
            # Vertical passage
            v_x = random.randint(GRID_WIDTH // 4, 3 * GRID_WIDTH // 4)
            for y in range(1, GRID_HEIGHT - 1):
                if random.random() < 0.7:
                    grid[y][v_x].is_wall = False
        
        # Test map connectivity
        if not test_map_accessibility(grid):
            continue  # Skip to next iteration if map is not fully accessible
        
        # Add lava
        lava_count = 0
        max_lava = 5
        attempts = 0
        max_attempts = 100
        
        while lava_count < max_lava and attempts < max_attempts:
            attempts += 1
            x = random.randint(1, GRID_WIDTH - 2)
            y = random.randint(1, GRID_HEIGHT - 2)
            
            manhattan_dist = abs(x - start_x) + abs(y - start_y)
            
            if (manhattan_dist > safe_radius + 2 and 
                not grid[y][x].is_wall and 
                not grid[y][x].is_lava and 
                not grid[y][x].is_visited):
                
                has_walkable_neighbor = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and 
                            not grid[ny][nx].is_wall and 
                            not grid[ny][nx].is_lava):
                            has_walkable_neighbor = True
                            break
                    if has_walkable_neighbor:
                        break
                
                if has_walkable_neighbor:
                    grid[y][x].is_lava = True
                    lava_count += 1
        
        # Add treasures
        far_point, distance = find_treasure_farthest_from_start(grid, start_x, start_y)
        treasures_count = 0
        max_treasures = 10
        
        # Place treasure at farthest point
        if distance > 10:
            far_x, far_y = far_point
            if (not grid[far_y][far_x].is_wall and 
                not grid[far_y][far_x].is_lava and
                not (far_x == start_x and far_y == start_y)):
                grid[far_y][far_x].is_treasure = True
                treasures_count += 1
        
        # Add remaining treasures
        while treasures_count < max_treasures:
            x = random.randint(1, GRID_WIDTH - 2)
            y = random.randint(1, GRID_HEIGHT - 2)
            
            if (not grid[y][x].is_wall and 
                not grid[y][x].is_lava and 
                not grid[y][x].is_treasure and
                not (x == start_x and y == start_y)):
                grid[y][x].is_treasure = True
                treasures_count += 1
            
            # Place exit door at the bottom right area
            exit_x = GRID_WIDTH - 2
            exit_y = GRID_HEIGHT - 2
            grid[exit_y][exit_x].is_exit = True
            grid[exit_y][exit_x].is_wall = False
            grid[exit_y][exit_x].is_lava = False
            grid[exit_y][exit_x].is_treasure = False
        
        # If we get here, we have a valid map
        return grid, start_x, start_y
    
    # If a valid map cannot be generated after several attempts, a simple map is created
    print("Unable to generate complex maps, create simple maps")
    grid = [[Cell(x, y) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]
    
    # Adding only boundary walls
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if x == 0 or y == 0 or x == GRID_WIDTH - 1 or y == GRID_HEIGHT - 1:
                grid[y][x].is_wall = True
    
    start_x, start_y = 1, 1
    
    # Add some treasure
    for _ in range(10):
        x = random.randint(1, GRID_WIDTH - 2)
        y = random.randint(1, GRID_HEIGHT - 2)
        if not (x == start_x and y == start_y):
            grid[y][x].is_treasure = True
    
    return grid, start_x, start_y

def spawn_enemies(grid, player_x, player_y, count=3):
    enemies = []
    safe_radius = 5  # Safe distance from players
    
    attempts = 0
    max_attempts = 50  # Maximum number of attempts to prevent a dead loop
    
    while len(enemies) < count and attempts < max_attempts:
        attempts += 1
        x = random.randint(1, GRID_WIDTH - 2)
        y = random.randint(1, GRID_HEIGHT - 2)
        
        # Calculate the distance to the player
        distance = math.sqrt((x*CELL_SIZE - player_x)**2 + (y*CELL_SIZE - player_y)**2)
        
        # Ensure that enemies do not spawn near the player, on walls, lava or black walls
        if (not grid[y][x].is_wall and
            not grid[y][x].is_lava and
            distance > safe_radius * CELL_SIZE):
            
            # Check the surrounding area for walkable paths
            has_path = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and 
                        not grid[ny][nx].is_wall and 
                        not grid[ny][nx].is_black_wall and 
                        not grid[ny][nx].is_lava):
                        has_path = True
                        break
                if has_path:
                    break
            
            if has_path:
                enemy_x = x * CELL_SIZE + 5
                enemy_y = y * CELL_SIZE + 5
                enemies.append(Enemy(enemy_x, enemy_y))
    
    # Make sure there's at least one enemy.
    if not enemies and count > 0:
        # If the generation fails, force a generation at a distance from the player
        # Choose one of the four corners of the map
        corners = [
            (1, 1), 
            (1, GRID_HEIGHT-2), 
            (GRID_WIDTH-2, 1), 
            (GRID_WIDTH-2, GRID_HEIGHT-2)
        ]
        # Find the farthest corner from the player
        max_dist = 0
        best_corner = None
        for cx, cy in corners:
            dist = math.sqrt((cx*CELL_SIZE - player_x)**2 + (cy*CELL_SIZE - player_y)**2)
            if dist > max_dist and not grid[cy][cx].is_wall and not grid[cy][cx].is_lava:
                max_dist = dist
                best_corner = (cx, cy)
        
        if best_corner:
            cx, cy = best_corner
            enemy_x = cx * CELL_SIZE + 5
            enemy_y = cy * CELL_SIZE + 5
            enemies.append(Enemy(enemy_x, enemy_y))
    
    return enemies

def draw_grid(grid):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if grid[y][x].is_wall:
                pygame.draw.rect(screen, BROWN, rect)
            elif grid[y][x].is_lava:
                pygame.draw.rect(screen, BLACK, rect)
                pygame.draw.rect(screen, RED, rect.inflate(-4, -4))  
            elif grid[y][x].is_exit:
                screen.blit(door_image, rect)  # Draw door image instead of green rectangle
            else:
                pygame.draw.rect(screen, BLACK, rect)
                pygame.draw.rect(screen, WHITE, rect, 1) 
        
            if grid[y][x].is_treasure:
                treasure_rect = pygame.Rect(x * CELL_SIZE + 10, y * CELL_SIZE + 10, 
                                          CELL_SIZE - 20, CELL_SIZE - 20)
                pygame.draw.rect(screen, YELLOW, treasure_rect)

def draw_hud():
    # Plotting fractions
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

def show_lava_death_screen():
    """Showing death by falling into lava"""
    # Clear screen
    screen.fill(BLACK)
    
    # Drawing a red background to indicate magma
    lava_rect = pygame.Rect(20, 20, WIDTH - 40, HEIGHT - 40)
    pygame.draw.rect(screen, RED, lava_rect)
    
    # Displaying a death message
    death_font = pygame.font.SysFont('SimHei', 48, bold=True)  
    message_font = pygame.font.SysFont('SimHei', 36) 
    
    # Draw a text background block to add contrast
    death_text = death_font.render("You fall into the lava, you die!", True, WHITE)
    message_text = message_font.render("Press R to restart the game", True, WHITE)
    
    death_rect = death_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
    message_rect = message_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 30))
    
    # Add a black background to make the text more visible
    pygame.draw.rect(screen, BLACK, death_rect)
    pygame.draw.rect(screen, BLACK, message_rect)
    
    # display text
    screen.blit(death_text, death_rect)
    screen.blit(message_text, message_rect)
    
    # Make sure the screen is up to date
    pygame.display.flip()
    
    # Adds a very short delay on death to ensure rendering is complete
    pygame.time.delay(50)

# Modified main function to wait for user confirmation before restarting
def main():
    global health, score, fell_into_lava, game_over, level_complete
    running = True
    
    while running:  # Outer loop for game restart
        # Initialize game state
        health = MAX_HEALTH
        score = 0
        fell_into_lava = False
        game_over = False
        level_complete = False  
        show_death_screen = False
        waiting_for_restart = False  # New: flag to indicate waiting for user confirmation
        
        # Generate map (only once at game start or restart)
        grid, start_x, start_y = generate_dungeon()
        
        # Create player
        player_x = start_x * CELL_SIZE + 5
        player_y = start_y * CELL_SIZE + 5
        player = Player(player_x, player_y)
        
        # Generate enemies
        enemies = spawn_enemies(grid, player_x, player_y)
        
        # Game main loop
        while running:  # Inner loop for actual gameplay
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    # When waiting for restart, press R to restart the game
                    if waiting_for_restart and event.key == pygame.K_r:
                        waiting_for_restart = False
                        game_over = False
                        break
            
            # If waiting for restart, skip game logic and continue showing end screen
            if waiting_for_restart:
                if fell_into_lava:
                    show_lava_death_screen()
                else:
                    # Keep showing game over/level complete screen
                    screen.fill(BLACK)
                    draw_grid(grid)
                    player.draw()
                    for enemy in enemies:
                        enemy.draw()
                    draw_hud()
                    
                    if level_complete:
                        text = f"Level Complete! Final score: {score}. Press R to restart"
                    else:
                        text = "Game Over! Press R to restart"
                    
                    game_over_text = font.render(text, True, WHITE)
                    text_rect = game_over_text.get_rect(center=(WIDTH//2, HEIGHT//2))
                    screen.blit(game_over_text, text_rect)
                
                pygame.display.flip()
                clock.tick(FPS)
                continue
            
            # If game is over but not yet waiting for restart
            if game_over:
                waiting_for_restart = True
                continue
                
            # Check if fallen into lava
            if fell_into_lava and not show_death_screen:
                show_death_screen = True
                show_lava_death_screen()
                game_over = True
                waiting_for_restart = True
                continue
            
            if show_death_screen:
                show_lava_death_screen()
                pygame.display.flip()
                continue
            
            # Player movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player.move(-player.vel, 0, grid)
            if keys[pygame.K_RIGHT]:
                player.move(player.vel, 0, grid)
            if keys[pygame.K_UP]:
                player.move(0, -player.vel, grid)
            if keys[pygame.K_DOWN]:
                player.move(0, player.vel, grid)
            
            # Enemy movement and collision
            for enemy in enemies:
                enemy.move_towards_player(player, grid)
                if player.rect.colliderect(enemy.rect):
                    health -= HEALTH_DECREASE                        
                    # Knockback player
                    dx = player.x - enemy.x
                    dy = player.y - enemy.y
                    distance = max(1, math.sqrt(dx*dx + dy*dy))
                    knockback = 50
                    player.move(knockback * dx / distance, knockback * dy / distance, grid)
                    if health <= 0:
                        game_over = True
            
            # Drawing
            screen.fill(BLACK)
            draw_grid(grid)
            player.draw()
            for enemy in enemies:
                enemy.draw()
            draw_hud()
            
            # Display health
            health_text = font.render(f"Health: {health}", True, WHITE)
            screen.blit(health_text, (10, 40))
            
            pygame.display.flip()
            clock.tick(FPS)
            
        # If we break out of the inner loop but not the outer one
        if not waiting_for_restart and game_over:
            break
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()  # Fix: remove the recursive call to main()