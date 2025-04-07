from Generator_Trainer import train_dungeon_generator, generate_dungeon_with_model, visualize_dungeon, export_dungeon, load_trained_agent

#### Train model
# trained_agent, trained_envs = train_dungeon_generator(num_episodes=50)
# generated_dungeon = generate_dungeon_with_model(trained_agent, 7)
# visualize_dungeon(generated_dungeon)

# difficulty = 8
# print(f"Difficulty: {difficulty}")
# generated_dungeon = generate_dungeon_with_model(trained_agent, difficulty)
# visualize_dungeon(generated_dungeon)
# gen_size, gen_start_pos, gen_end_pos, gen_grid = export_dungeon(generated_dungeon)

#### Generate new levels from saved weights
difficulty = 8
agent = load_trained_agent()
generated_dungeon = generate_dungeon_with_model(agent, difficulty)
gen_size, gen_start_pos, gen_end_pos, gen_grid = export_dungeon(generated_dungeon)
visualize_dungeon(generated_dungeon)