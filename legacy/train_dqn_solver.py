import os
import torch
from generator.Generator_Trainer import load_trained_agent, generate_dungeon_with_model, export_dungeon
from legacy.dqn_solver import DQNSolver

def train_general_solver(
    model_path="dqn_solver_general.pth",
    num_difficulties=10,
    levels_per_difficulty=5,
    episodes_per_level=200
):
    # Load the pretrained generator agent
    gen_agent = load_trained_agent()

    # Initialize DQN solver with dummy values, model will be reused
    dqn_solver = DQNSolver(grid=None, start=None, goal=None, difficulty=1, episodes=episodes_per_level)

    # Load existing weights if present
    if os.path.exists(model_path):
        dqn_solver.model.load_state_dict(torch.load(model_path, map_location=dqn_solver.device))
        print(f"✅ Loaded existing solver model from {model_path}")

    for difficulty in range(1, num_difficulties + 1):
        print(f"\n🧠 Training on difficulty {difficulty}...")
        rewards = []

        for _ in range(levels_per_difficulty + 2*difficulty):
            # Generate a new level for the current difficulty
            dungeon = generate_dungeon_with_model(gen_agent, difficulty)
            size, start_pos, exit_pos, grid = export_dungeon(dungeon)

            # Update solver environment and difficulty
            dqn_solver.grid = grid
            dqn_solver.start = start_pos[::-1]
            dqn_solver.goal = exit_pos[::-1]
            dqn_solver.difficulty = difficulty

            # Recompute epsilon decay for current difficulty
            decay_steps = int(100 + difficulty * 10)
            dqn_solver.epsilon_decay = (dqn_solver.epsilon_min / dqn_solver.epsilon_start) ** (1 / decay_steps)

            # Train and collect reward info
            total_reward = dqn_solver.train()
            rewards.append(total_reward)

        avg_reward = sum(rewards) / len(rewards)
        print(f"➡️  Avg Reward @ Difficulty {difficulty}: {avg_reward:.2f}")

    # Save the trained general solver
    torch.save(dqn_solver.model.state_dict(), model_path)
    print(f"\n✅ General DQN solver saved to {model_path}")
