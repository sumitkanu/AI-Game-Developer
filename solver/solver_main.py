import numpy as np
from solver.ai_solver import solve_generated_level

def parse_tuple(line):
    return tuple(map(int, line.strip().strip('()').split(',')))

def read_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse size, start, and goal
    size = parse_tuple(lines[0])  # Not used, but could validate shape
    start = parse_tuple(lines[1])
    goal = parse_tuple(lines[2])

    # Read grid lines as rows
    dungeon_rows = []
    for line in lines[3:]:
        line = line.strip().replace('[', '').replace(']', '')
        if line:
            dungeon_rows.append(list(map(int, line.split())))

    dungeon = np.array(dungeon_rows)

    # Optional: Validate shape matches declared size
    if dungeon.shape != size:
        raise ValueError(f"Declared size {size} doesn't match actual dungeon shape {dungeon.shape}")

    return dungeon, start, goal

def write_output(filename, path, outcome):
    with open(filename, 'w') as f:
        f.write("Path:\n")
        for step in path:
            f.write(f"{step}\n")
        f.write("\nOutcome:\n")
        f.write(outcome)

if __name__ == "__main__":
    dungeon, start, goal = read_input("input.txt")
    path, outcome = solve_generated_level(dungeon, start, goal)
    write_output("output.txt", path, outcome)