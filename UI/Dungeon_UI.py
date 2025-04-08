import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from PIL import Image, ImageTk
import os
import torch
import torch.nn.functional as F
import random
from solver.ai_solver import solve_generated_level


# Direct imports from source files to avoid circular imports
from generator.Generator_Agent import DungeonAgent, EMPTY, WALL, LAVA, TREASURE, EXIT, START, ENEMY, COLOR_MAP
from generator.Dungeon_Environment import DungeonEnvironment

def one_hot_encode_grid_with_cursor(grid, cursor, num_tile_types=7):
    """
    Converts a 2D grid and cursor position into an 8-channel tensor [8, 20, 20]
    """
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    one_hot = F.one_hot(grid_tensor, num_classes=num_tile_types)  # [20, 20, 7]
    one_hot = one_hot.permute(2, 0, 1).float()  # [7, 20, 20]

    # Add 8th channel for cursor
    cursor_channel = torch.zeros((1, 20, 20), dtype=torch.float32)
    cursor_x, cursor_y = cursor
    cursor_channel[0, cursor_y, cursor_x] = 1.0

    full_tensor = torch.cat([one_hot, cursor_channel], dim=0)  # [8, 20, 20]
    return full_tensor

def encode_difficulty(difficulty):
    """
    Converts a scalar difficulty (1–10) into a normalized tensor [B, 1]
    """
    return torch.tensor([[difficulty / 10.0]], dtype=torch.float32)  # shape [1, 1]

def load_trained_agent(model_path="dungeon_rl_model.pth"):
    """Load the trained model weights into a new agent."""
    action_size = 9

    agent = DungeonAgent(action_size=action_size)
    try:
        agent.qnetwork.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.qnetwork.eval()
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained agent instead")
    
    return agent

def generate_dungeon_with_model(agent, difficulty):
    """Generate a dungeon using the trained agent."""
    env = DungeonEnvironment(difficulty)
    env.reset()
    done = False

    # Prepare the initial state
    grid_tensor = one_hot_encode_grid_with_cursor(env.grid, env.cursor).unsqueeze(0)
    difficulty_tensor = encode_difficulty(env.difficulty)
    state = (grid_tensor, difficulty_tensor)

    while not done:
        action = agent.act(*state, eps_override=0.0)

        # Add some randomness for variety
        if random.random() < 0.2:
            action = random.randint(0, 8)

        # Environment transition
        next_state, _, done = env.step(action)
        
        # Update state
        state = next_state

    return env.grid

def export_dungeon(grid):
    """
    Inspect a dungeon grid and return:
        - its size (width, height)
        - the start position (START tile)
        - the exit position (EXIT tile)
        - the grid itself
    """
    rows, cols = grid.shape
    start_pos = None
    exit_pos = None

    for row in range(rows):
        for col in range(cols):
            if grid[row, col] == START:
                start_pos = (col, row)  # Note: using (col, row) format for x,y coordinates
            elif grid[row, col] == EXIT:
                exit_pos = (col, row)

    return (rows, cols), start_pos, exit_pos, grid

class DungeonGeneratorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dungeon Generator")
        self.root.geometry("750x600")
        self.last_solver_path = None
        self.last_solver_outcome = ""
        self.solver_outcome_var = tk.StringVar(value="Solver: Not yet run")
        
        # Load the trained agent
        self.agent = load_trained_agent()
        
        # Image paths for dungeon elements - using existing images from your project
        self.element_images = {
            EMPTY: self.load_image_or_color("UI/Floor.png", COLOR_MAP[EMPTY]),
            WALL: self.load_image_or_color("UI/Wall.png", COLOR_MAP[WALL]),
            LAVA: self.load_image_or_color("UI/Lava.png", COLOR_MAP[LAVA]),
            TREASURE: self.load_image_or_color("UI/Treasure.png", COLOR_MAP[TREASURE]),
            EXIT: self.load_image_or_color("UI/Door.png", COLOR_MAP[EXIT]),
            START: self.load_image_or_color("UI/Start.png", COLOR_MAP[START]),
            ENEMY: self.load_image_or_color("UI/monster.png", COLOR_MAP[ENEMY])
        }
        
        # Current dungeon data
        self.current_grid = None
        self.current_start_pos = None
        self.current_exit_pos = None
        self.current_treasure_count = 0
        
        # Create UI elements
        self.create_widgets()
        
        # Generate initial dungeon
        self.generate_dungeon()
    
    def load_image_or_color(self, image_path, color):
        """Load image if exists, otherwise return the color"""
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((25, 25))  # Resize to fit grid cells
                return ImageTk.PhotoImage(img)
            print(f"Image not found: {image_path}, using color instead")
            return color
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return color
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side (dungeon display)
        dungeon_frame = ttk.Frame(main_frame)
        dungeon_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for dungeon grid
        self.canvas_size = 500
        self.canvas = tk.Canvas(dungeon_frame, width=self.canvas_size, height=self.canvas_size,
                              bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.pack(padx=10, pady=10)
        
        # Right side (controls)
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Level selection
        level_frame = ttk.LabelFrame(control_frame, text="Level", padding=10)
        level_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(level_frame, text="Difficulty:").pack(anchor=tk.W)
        
        self.difficulty_var = tk.IntVar(value=5)
        difficulty_scale = ttk.Scale(level_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                   variable=self.difficulty_var, command=self.on_difficulty_change)
        difficulty_scale.pack(fill=tk.X)
        
        self.difficulty_label = ttk.Label(level_frame, text="5")
        self.difficulty_label.pack()
        
        # Generate button
        generate_button = ttk.Button(level_frame, text="Generate Dungeon", command=self.generate_dungeon)
        generate_button.pack(fill=tk.X, pady=10)

        # Solve button
        solve_button = ttk.Button(level_frame, text="Solve Dungeon", command=self.solve_current_dungeon)
        solve_button.pack(fill=tk.X, pady=5)

        # Rewards display
        rewards_frame = ttk.LabelFrame(control_frame, text="Rewards", padding=10)
        rewards_frame.pack(fill=tk.X, pady=10)
        
        self.treasure_var = tk.StringVar(value="Treasures: 0")
        ttk.Label(rewards_frame, textvariable=self.treasure_var).pack(anchor=tk.W)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.size_var = tk.StringVar(value="Size: 20x20")
        self.start_var = tk.StringVar(value="Start: (0,0)")
        self.exit_var = tk.StringVar(value="Exit: (0,0)")
        
        ttk.Label(stats_frame, textvariable=self.size_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.start_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.exit_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.solver_outcome_var).pack(anchor=tk.W)
    
    def on_difficulty_change(self, event):
        difficulty = int(float(self.difficulty_var.get()))
        self.difficulty_label.configure(text=str(difficulty))
    
    def generate_dungeon(self):
        self.last_solver_path = None
        self.solver_outcome_var.set("Solver: Not yet run")
        difficulty = int(self.difficulty_var.get())
        
        # Generate dungeon using the model
        generated_dungeon = generate_dungeon_with_model(self.agent, difficulty)
        
        # Export dungeon data
        (rows, cols), start_pos, exit_pos, grid = export_dungeon(generated_dungeon)
        
        self.current_grid = grid
        self.current_start_pos = start_pos
        self.current_exit_pos = exit_pos
        
        # Calculate treasure count
        self.current_treasure_count = np.sum(grid == TREASURE)
        
        # Update UI
        self.update_dungeon_display()
        self.update_stats()
    
    def update_dungeon_display(self):
        self.canvas.delete("all")
        
        if self.current_grid is None:
            return
        
        rows, cols = self.current_grid.shape
        cell_size = min(self.canvas_size // rows, self.canvas_size // cols)
        
        # Center the grid in the canvas
        x_offset = (self.canvas_size - (cols * cell_size)) // 2
        y_offset = (self.canvas_size - (rows * cell_size)) // 2
        
        # Draw each cell
        for y in range(rows):
            for x in range(cols):
                cell_value = self.current_grid[y, x]
                
                # Calculate position
                x1 = x_offset + x * cell_size
                y1 = y_offset + y * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Draw cell
                if isinstance(self.element_images[cell_value], str):
                    # It's a color
                    self.canvas.create_rectangle(x1, y1, x2, y2, 
                                            fill=self.element_images[cell_value],
                                            outline="black")
                else:
                    # It's an image
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
                    self.canvas.create_image(x1 + cell_size//2, y1 + cell_size//2, 
                                        image=self.element_images[cell_value])
        
        # Draw solver path if available - 绘制在地图上方而不是替换地图
        if self.last_solver_path:
            for step in self.last_solver_path:
                px, py = step
                px1 = x_offset + px * cell_size
                py1 = y_offset + py * cell_size
                px2 = px1 + cell_size
                py2 = py1 + cell_size
                # 只绘制路径标记，不替换原始单元格
                self.canvas.create_rectangle(px1, py1, px2, py2, outline="red", width=2)

    
    def update_stats(self):
        if self.current_grid is not None:
            rows, cols = self.current_grid.shape
            self.size_var.set(f"Size: {cols}x{rows}")
            
            if self.current_start_pos:
                self.start_var.set(f"Start: ({self.current_start_pos[0]},{self.current_start_pos[1]})")
            
            if self.current_exit_pos:
                self.exit_var.set(f"Exit: ({self.current_exit_pos[0]},{self.current_exit_pos[1]})")
            
            self.treasure_var.set(f"Treasures: {self.current_treasure_count}")

    def solve_current_dungeon(self):
        if self.current_grid is None or self.current_start_pos is None or self.current_exit_pos is None:
            print("No dungeon loaded.")
            return

        # Call solver
        try:
            path, outcome = solve_generated_level(self.current_grid, self.current_start_pos, self.current_exit_pos)
            self.last_solver_path = path
            self.last_solver_outcome = outcome
            self.solver_outcome_var.set(f"Solver: {outcome}")

            self.update_dungeon_display()
        except Exception as e:
            print("Solver error:", e)


def launch_ui():
    root = tk.Tk()
    app = DungeonGeneratorUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_ui()