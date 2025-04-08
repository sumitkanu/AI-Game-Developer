from generator.Generator_Trainer import train_dungeon_generator, generate_dungeon_with_model, visualize_dungeon, export_dungeon, load_trained_agent

def train_generator(num_episodes=1000):
    train_dungeon_generator(num_episodes)

def generate_dungeon(difficulty=8):
    agent = load_trained_agent()
    generated_dungeon = generate_dungeon_with_model(agent, difficulty)
    gen_size, gen_start_pos, gen_end_pos, gen_grid = export_dungeon(generated_dungeon)
    return gen_size, gen_start_pos, gen_end_pos, gen_grid