import random
from generator.Dungeon_Environment import DungeonEnvironment
from generator.Generator_Trainer import encode_difficulty, is_path_valid, one_hot_encode_grid_with_cursor, train_dungeon_generator, generate_dungeon_with_model, visualize_dungeon, export_dungeon, load_trained_agent

def train_generator(num_episodes=1000):
    train_dungeon_generator(num_episodes)

def generate_dungeon(difficulty=8):
    agent = load_trained_agent()
    generated_dungeon = generate_dungeon_with_model(agent, difficulty)
    gen_size, gen_start_pos, gen_end_pos, gen_grid = export_dungeon(generated_dungeon)
    return gen_size, gen_start_pos, gen_end_pos, gen_grid

def evaluate_generator_success_rate(agent, num_trials_per_difficulty=100):
    from collections import defaultdict

    results = defaultdict(lambda: {'attempts': 0, 'successes': 0})

    for difficulty in range(1, 11):
        print(f"\nEvaluating difficulty {difficulty}...")
        for _ in range(num_trials_per_difficulty):
            attempts = 0
            while True:
                attempts += 1
                env = DungeonEnvironment(difficulty)
                env.reset()
                done = False

                # Run agent
                grid_tensor = one_hot_encode_grid_with_cursor(env.grid, env.cursor).unsqueeze(0)
                difficulty_tensor = encode_difficulty(env.difficulty)
                state = (grid_tensor, difficulty_tensor)

                while not done:
                    action = agent.act(*state, eps_override=0.0)
                    if random.random() < 0.2:
                        action = random.randint(0, 8)
                    next_state, _, done = env.step(action)
                    state = next_state

                _, start, goal, grid = export_dungeon(env.grid)
                results[difficulty]['attempts'] += 1

                if is_path_valid(grid, start, goal):
                    results[difficulty]['successes'] += 1
                    break  # count only first success

    # Print report
    print("\n=== Generator Success Rate ===")
    for d in range(1, 11):
        a = results[d]['attempts']
        s = results[d]['successes']
        r = 100 * s / a if a > 0 else 0
        print(f"  Difficulty {d}: {s}/{a} valid ({r:.2f}%)")

    return results

if __name__ == "__main__":
    agent = load_trained_agent("dungeon_rl_model_5000.pth")
    evaluate_generator_success_rate(agent, num_trials_per_difficulty=50)
