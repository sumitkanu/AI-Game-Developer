import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
from generator.Dungeon_Environment import DungeonEnvironment
from generator.Generator_Agent import DungeonAgent, COLOR_MAP, EXIT, START, TREASURE
from solver.ai_solver import solve_generated_level

def one_hot_encode_grid_with_cursor(grid, cursor, num_tile_types=7):
    """
    Converts a 2D grid and cursor position into an 8-channel tensor [8, 20, 20]
    """
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    one_hot = F.one_hot(grid_tensor, num_classes=num_tile_types)
    one_hot = one_hot.permute(2, 0, 1).float()

    # Add 8th channel for cursor
    cursor_channel = torch.zeros((1, 20, 20), dtype=torch.float32)
    cursor_x, cursor_y = cursor
    cursor_channel[0, cursor_x, cursor_y] = 1.0

    full_tensor = torch.cat([one_hot, cursor_channel], dim=0)
    return full_tensor

def encode_difficulty(difficulty):
    """
    Converts a scalar difficulty (1–10) into a normalized tensor [B, 1]
    """
    return torch.tensor([[difficulty / 10.0]], dtype=torch.float32)

def train_dungeon_generator(num_episodes=1000, model_path="dungeon_rl_model.pth"):
    """Train a single RL model to generate levels with CNN and solver-based feedback."""
    envs = {d: DungeonEnvironment(d) for d in range(1, 11)}
    tile_types = len(envs[1].element_types)
    action_size = 9

    agent = DungeonAgent(action_size=action_size)
    episode_rewards = []
    solver_scores = []

    t0 = time.time()

    for episode in range(num_episodes):
        difficulty = min(1 + episode // 200, 10)
        env = envs[difficulty]
        env.reset()

        grid_tensor = one_hot_encode_grid_with_cursor(env.grid, env.cursor).unsqueeze(0)
        difficulty_tensor = encode_difficulty(env.difficulty)
        state = (grid_tensor, difficulty_tensor)

        done = False
        total_reward = 0
        losses_before = len(agent.training_losses)

        while not done:
            action = agent.act(*state)
            _, reward, done = env.step(action)

            next_grid_tensor = one_hot_encode_grid_with_cursor(env.grid, env.cursor).unsqueeze(0)
            next_difficulty_tensor = encode_difficulty(env.difficulty)
            next_state = (next_grid_tensor, next_difficulty_tensor)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # ====== SOLVER FEEDBACK INTEGRATION ======
        final_grid = env.grid
        (rows, cols), start, goal, _ = export_dungeon(final_grid)

        solver_reward = 0
        solver_success = False

        try:
            solve_start = time.time()
            path, outcome = solve_generated_level(final_grid, start, goal)
            solve_time = time.time() - solve_start

            solver_success = "Reached exit" in outcome

            if solver_success:
                expected_time = 20 + 5 * difficulty
                time_penalty = max(0, solve_time - expected_time)
                time_score = max(0, 100 - int(time_penalty))

                treasure_count = sum(1 for (x, y) in path if final_grid[y][x] == TREASURE)
                solver_reward = time_score + (5 * treasure_count)
            else:
                solver_reward = -100  # Unsolvable penalty
        except Exception as e:
            print(f"[Solver Error] Episode {episode} — skipping solver reward:", str(e))
            solver_reward = -100

        total_reward += solver_reward
        solver_scores.append(solver_reward)

        episode_rewards.append(total_reward)

        # === Logging ===
        if episode % 50 == 0 and episode > 0:
            elapsed = time.time() - t0
            avg_reward = sum(episode_rewards[-50:]) / 50
            avg_loss = (sum(agent.training_losses[losses_before:]) /
                        max(1, len(agent.training_losses) - losses_before))
            avg_solver = sum(solver_scores[-50:]) / 50
            print(f"Episode {episode} | Reward: {avg_reward:.2f} | Solver: {avg_solver:.2f} | "
                  f"Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f} | Time: {elapsed:.2f}s")
            t0 = time.time()

    # Save model
    torch.save(agent.qnetwork.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Plot diagnostics
    plot_training_diagnostics(episode_rewards, agent.training_losses, solver_scores)

    return agent, envs

def plot_training_diagnostics(rewards, losses, solver_scores=None):
    num_plots = 3 if solver_scores else 2
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    axs[0].plot(rewards)
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(losses)
    axs[1].set_title("Q-Network Losses")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Loss")

    if solver_scores:
        axs[2].plot(solver_scores)
        axs[2].set_title("Solver Feedback Scores")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Score")

    plt.tight_layout()
    plt.show()


def generate_dungeon_with_model(agent, difficulty):
    env = DungeonEnvironment(difficulty)
    state = env.reset()  # returns just the grid
    done = False

    while not done:
        grid_tensor, difficulty_tensor = state  # since env.reset() returns (grid_tensor, difficulty_tensor)
        difficulty_tensor = torch.tensor([[difficulty / 10.0]], dtype=torch.float32)

        action = agent.act(grid_tensor, difficulty_tensor, eps_override=0.0)

        if random.random() < 0.2:
            action = random.randint(0, 8)

        state, _, done = env.step(action)

    return env.grid

def visualize_dungeon(grid):
    """Display dungeon grid using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            rect = plt.Rectangle(
                (y, grid.shape[0] - x - 1), 1, 1,
                facecolor=COLOR_MAP[grid[x, y]],
                edgecolor='black'
            )
            ax.add_patch(rect)

    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Generated Dungeon")
    plt.show()

def export_dungeon(grid):
    """
    Inspect a dungeon grid and return:
        - the grid itself
        - its size (width, height)
        - the start position (START tile)
        - the exit position (EXIT tile)
    """
    rows, cols = grid.shape
    start_pos = None
    exit_pos = None

    for row in range(rows):
        for col in range(cols):
            if grid[row, col] == START:
                start_pos = (row, col)
            elif grid[row, col] == EXIT:
                exit_pos = (row, col)

    return (rows, cols), start_pos, exit_pos, grid

def load_trained_agent(model_path="dungeon_rl_model.pth"):
    """Load the trained model weights into a new agent."""
    grid_size = 20 * 20
    tile_types = len(COLOR_MAP)
    state_size = grid_size * tile_types + 2
    action_size = 9

    agent = DungeonAgent(action_size=action_size)
    agent.qnetwork.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.qnetwork.eval()

    print(f"Model loaded from: {model_path}")
    return agent
