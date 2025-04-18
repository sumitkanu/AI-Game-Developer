# AI Dungeon Generator & Solver

This project implements a **dual-agent system** for automated game level design. One agent—a CNN-based Deep Q-Network (DQN)—**generates 2D dungeon levels**, while a second agent—a Q-learning solver—**validates playability** by attempting to solve the generated map. The generator learns from both environment-level shaping and solver-based feedback, forming a closed training loop that produces playable, difficulty-scaled dungeons over time.

The system includes:
- A **level generator** trained via reinforcement learning (DQN with CNN encoder)
- A **Q-learning-based solver** to verify level solvability
- A **Tkinter UI** to visualize, solve, and play dungeons interactively

---

## How to Run

### Requirements

Ensure you have **Python 3.8+** and the following Python packages installed:

```bash
pip install torch numpy matplotlib pillow numba
```

### Run the Visual UI

Launch the interactive dungeon generator + player interface:
```bash
python main.py --mode ui
```
This allows you to:
- Generate a level using the trained model
- Visualize the solver’s path
- Play through the dungeon using arrow keys
- View your score based on items collected and obstacles avoided

### Train the Generator

To train the generator from scratch:
```bash
python main.py --mode train
```
By default, it runs for 1000 episode. To train over a longer period:
```bash
# Edit inside Generator_Main.py or main.py:
train_generator(num_episodes=5000)
```
Training logs and weights are saved to:
- training_metrics.csv — logs reward, loss, tile counts, and solver success
- dungeon-rl-model.pth - saves weights of trained models

## How It Works

- The Generator Agent uses a CNN-based Deep Q-Network to place tiles across a 20x20 dungeon grid.
- The Dungeon Environment gives rewards based on path complexity, entropy, and structural constraints.
- The Solver Agent, a Q-learning policy, attempts to solve the generated dungeon.
- Success or failure from the solver is fed back to the generator, closing the training loop.
- A curriculum scheduler ensures higher difficulty levels receive more training exposure.
- The Tkinter UI lets users test levels, visualize solver paths, and play through dungeons manually.
